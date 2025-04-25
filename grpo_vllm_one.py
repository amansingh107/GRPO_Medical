from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Path configuration - adjust these to match your remote server's paths
model_paths = [
    "/data2/Qwen/Qwen2.5-3B",
    "/home/surajracha/aman/models/Qwen2.5-3B",  # Alternative path
    "./models/Qwen2.5-3B"                       # Local relative path
]

# Find first valid model path
model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        print(f"Using model path: {model_path}")
        break

if model_path is None:
    print("No valid model path found. Will attempt to download from Hugging Face.")
    model_path = "Qwen/Qwen2.5-3B"

# Choose available GPU for generation
gen_device = 0  # Default to first GPU
# Check available GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    if num_gpus > 1:
        gen_device = 1  # Use second GPU if available
    else:
        gen_device = 0  # Use the only GPU

print(f"Using GPU {gen_device} for generation")

beta = 0.04
all_steps = 1000
Q_batch_size = 5
num_pre_Q = 8
train_batch_size = 8
gen_update_steps = 16
save_steps = 200
compute_gen_logps = True
clip_param = 0.2

# Reference server configuration
ref_server = "http://localhost:59875"
# Also try alternative port if needed
alternative_ref_server = "http://localhost:59876"

# Import utility functions
try:
    from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
except ImportError:
    print("Could not import from ref_server.py. Make sure it's in the same directory.")
    sys.exit(1)

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

# Create save directory if it doesn't exist
os.makedirs("./checkpoints", exist_ok=True)

def get_batch():
    global ref_server, alternative_ref_server
    
    # Try primary server
    try:
        r = requests.get(f"{ref_server}/get", timeout=5).content
        if r == b'empty': return None
    except requests.exceptions.RequestException:
        print(f"Could not connect to {ref_server}")
        # Try alternative server
        try:
            print(f"Trying alternative server {alternative_ref_server}")
            r = requests.get(f"{alternative_ref_server}/get", timeout=5).content
            if r == b'empty': return None
            # If successful, switch to alternative server
            ref_server = alternative_ref_server
            print(f"Switched to alternative server: {ref_server}")
        except:
            print("Could not connect to any reference server")
            return None
    
    try:
        dd = bytes_list_to_list(r)
        data = json.loads(dd[0]) 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        data['refs'] = bytes_to_tensor(dd[3])
        if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
        return data
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    
    # Import vLLM here to ensure it uses the correct GPU
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Installing now...")
        os.system("pip install vllm")
        from vllm import LLM, SamplingParams
    
    # Try loading the model with vLLM
    try:
        vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.5)
    except Exception as e:
        print(f"Error loading model with vLLM: {e}")
        print("Trying with alternative parameters...")
        try:
            vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.3, max_model_len=2048)
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            print("Trying to download model from Hugging Face...")
            try:
                vllm_gen = LLM(model="Qwen/Qwen2.5-3B", gpu_memory_utilization=0.3)
            except Exception as e3:
                print(f"All attempts failed: {e3}")
                print("Exiting generation worker.")
                return
    
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=700)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    # Load medical dataset
    try:
        from datasets import load_dataset
        print("Loading medical dataset...")
        dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
        QAs = [{'Q': item['Patient'], 'A': item['Doctor']} for item in dataset]
        print(f"Loaded {len(QAs)} QA pairs")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating a small fallback dataset")
        # Fallback dataset in case of loading issues
        QAs = [
            {'Q': 'I have a headache and fever', 'A': 'You may have the flu. Rest and hydrate.'},
            {'Q': 'My blood pressure is 140/90', 'A': 'Your blood pressure is slightly elevated.'},
            {'Q': 'I have chest pain', 'A': 'Seek immediate medical attention.'},
            {'Q': 'I have a cough for 2 weeks', 'A': 'You should see a doctor for persistent cough.'},
            {'Q': 'My temperature is 101.5', 'A': 'You have a fever.'}
        ]
    
    # Medical system prompt
    system_prompt = """You are a medical assistant with expertise in healthcare. A conversation between Patient and Doctor. The patient asks a medical question, and the Doctor answers it. The Doctor first thinks about the diagnostic reasoning process and then provides the patient with the answer.
    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
    
    def gen_answers(prompts):
        try:
            tip_text = []
            for x in prompts:
                tip_text.append(tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
            voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
            answers = [];  ans_token_ids = []
            for v in voutputs:
                for z in v.outputs: 
                    answers.append(z.text)
                    ans_token_ids.append(z.token_ids)
            return answers, ans_token_ids
        except Exception as e:
            print(f"Error in gen_answers: {e}")
            # Return empty results to avoid crashing
            return [""] * len(prompts) * num_pre_Q, [[0]] * len(prompts) * num_pre_Q

    try:
        from math_verify import parse, verify, ExprExtractionConfig
    except ImportError:
        print("math_verify module not found. Using simple reward functions.")
        # Simple fallback reward functions
        def reward_correct(item, answer):
            if item["A"].lower() in answer.lower():
                return 1
            return -1
            
        def reward_format(item, answer):
            pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
            think_count = answer.count("<think>") + answer.count("</think>")
            answer_count = answer.count("<answer>") + answer.count("</answer>")
            return 1.25 if re.search(pattern, answer, re.DOTALL) and think_count==2 and answer_count==2 else -1
    else:
        # Original reward functions if module is available
        def reward_correct(item, answer):
            pattern = r'\d+\.\d+|\d+/\d+|\d+'
            nums = re.findall(pattern, answer) 
            if len(nums) == 0: return -1.0
            lastnum = nums[-1]
            try:
                ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
                ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
                return 1 if verify(ans, ground_truth) else -1
            except Exception as e:
                print(f"Error in reward_correct: {e}")
                return -1
                
        def reward_format(item, answer):
            pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
            think_count = answer.count("<think>") + answer.count("</think>")
            answer_count = answer.count("<answer>") + answer.count("</answer>")
            return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else -1

    def gen_samples(inputs):
        prompts = [x["Q"] for x in inputs]
        answers, ans_token_ids = gen_answers(prompts)
        rewards = []
        for i, inp in enumerate(inputs):
            for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                rewards.append(reward_correct(inp, a) + reward_format(inp, a))
        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] receiving new model ...')
            try:
                llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(new_state_dict.items())
                print('[VLLM PROC] model updated')
            except Exception as e:
                print(f'[VLLM PROC] model update failed: {e}')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence
    
    # Setup connection to both reference servers
    global ref_server, alternative_ref_server
    
    for it in range(999999999):
        if it % 3 == 0: try_update_model()
        
        # Handle potential errors during iteration
        try:
            inputs = random.sample(QAs, min(Q_batch_size, len(QAs)))
            tic = time.time()
            prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
            print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
            if it % 5 == 0 and answers: print('answers:', answers[0])

            for i, pp in enumerate(prompt_inputs):
                prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
                plen = prompt_ids.shape[1]
                curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
                if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

                # Try both ref server versions
                servers_to_try = [ref_server, alternative_ref_server]
                
                for current_server in servers_to_try:
                    try:
                        if ref_server_ver == 'tensor':
                            curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                            for ii in range(0, num_pre_Q, train_batch_size):
                                sub_rewards = curr_rewards[ii:ii+train_batch_size]
                                sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                                tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                                output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                                Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                                merged_ids = torch.cat([Qrep, output_ids], dim=1)
                                data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

                                if compute_gen_logps:
                                    try:
                                        zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                                        data.append(tensor_to_bytes(gen_logps))
                                    except Exception as e:
                                        print(f"Error computing gen_logps: {e}")
                                        compute_gen_logps = False

                                xdata = make_bytes_list(data)
                                r = requests.post(f"{current_server}/upload", data=xdata, timeout=5)
                                if r.content == b'string': ref_server_ver = 'string'
                                # Update active server if successful
                                ref_server = current_server
                                break
                        elif ref_server_ver == 'string':
                            xdata = make_bytes_list([json.dumps({"Q": pp, "As": curr_answers}).encode(), 
                                                    tensor_to_bytes(curr_rewards)])
                            r = requests.post(f"{current_server}/upload", data=xdata, timeout=5)
                            if r.content == b'tensor': ref_server_ver = 'tensor'
                            # Update active server if successful
                            ref_server = current_server
                            break
                    except Exception as e:
                        print(f"Error with server {current_server}: {e}")
                        # Try next server
                        continue
        except Exception as e:
            print(f"Error in generation loop: {e}")
            # Sleep a bit before retrying
            time.sleep(5)
            continue


# Load tokenizer with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading tokenizer from {model_path}: {e}")
    print("Trying to load from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    except Exception as e2:
        print(f"Failed to load tokenizer: {e2}")
        sys.exit(1)

if __name__ == '__main__':
    import deepspeed
    try:
        deepspeed.init_distributed()
    except Exception as e:
        print(f"Error initializing distributed training: {e}")
        print("Falling back to single-GPU mode")
        # Set environment variables for single-GPU mode
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        try:
            deepspeed.init_distributed()
        except Exception as e2:
            print(f"Still couldn't initialize distributed environment: {e2}")
            print("Will attempt to continue without distributed training")

    # Set up multiprocessing queue and launch generation worker
    # Note: Only set the start method once - this is the fix to the error
    mp.set_start_method('spawn', force=True)
    
    try:
        if dist.is_initialized() and dist.get_rank() == 0:
            print('\nSTART vLLM generation...\n')
            Q = mp.Queue()
            p = mp.Process(target=gen_worker, args=(Q, gen_device))
            p.start()
    except Exception as e:
        print(f"Error setting up generation process: {e}")
        print("Starting generation process directly")
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()

    # Load model with error handling
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Trying to load from Hugging Face...")
        try:
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", 
                    torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            sys.exit(1)

    try:
        engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                    model_parameters=model.parameters())
    except Exception as e:
        print(f"Error initializing DeepSpeed: {e}")
        sys.exit(1)

    try:
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    except:
        is_main_process = True

    progress = range(1, all_steps+1)
    if is_main_process: 
        progress = tqdm(progress)
        
    for step in progress:
        try:
            batch = get_batch()
            wait_count = 0
            while batch is None:
                wait_count += 1
                if wait_count % 30 == 0:  # Every 30 seconds
                    print(f'Still waiting for batch... Waited {wait_count} seconds')
                time.sleep(1)
                batch = get_batch()

            loss = GRPO_step(batch)
            engine.backward(loss)
            engine.step()

            if is_main_process:
                progress.set_description(f"Loss: {loss.item():.6f}")

            # Make sure dist is initialized before using barriers
            if dist.is_initialized():
                if step % gen_update_steps == 0:
                    dist.barrier()
                    if dist.get_rank() == 0:
                        print('[TRAINING PROC] sending latest state_dict ...')
                        state_dict = engine.module.state_dict()
                        Q.put(state_dict)
                        print('[TRAINING PROC] send state_dict ok!')
                    dist.barrier()

                if step % save_steps == 0:
                    dist.barrier()
                    if dist.get_rank() == 0:
                        print('saving model')
                        save_name = f"./checkpoints/step_{step}"
                        state_dict = engine.module.state_dict()
                        state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                        engine.module.save_pretrained(save_name, state_dict=state_dict)
                        tokenizer.save_pretrained(save_name)
                    dist.barrier()
            else:
                # Handle non-distributed case
                if step % gen_update_steps == 0:
                    print('[TRAINING PROC] sending latest state_dict ...')
                    state_dict = engine.module.state_dict()
                    Q.put(state_dict)
                    print('[TRAINING PROC] send state_dict ok!')

                if step % save_steps == 0:
                    print('saving model')
                    save_name = f"./checkpoints/step_{step}"
                    state_dict = engine.module.state_dict()
                    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                    engine.module.save_pretrained(save_name, state_dict=state_dict)
                    tokenizer.save_pretrained(save_name)
                    
        except Exception as e:
            print(f"Error in training loop at step {step}: {e}")
            # Sleep a bit before continuing
            time.sleep(5)
            continue