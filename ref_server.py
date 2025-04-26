import json, os, shutil, re, random, io, time
import torch

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)

def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

if __name__ == '__main__':   
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn as nn

    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    # Path to the model - adjust this to your remote machine's path
    model_path = "/data2/Qwen/Qwen2.5-7B"
    
    # Try alternative paths if the primary path doesn't exist
    alternative_paths = [
        "/home/surajracha/aman/models/Qwen2.5-3B",  # Example alternative path
        "./models/Qwen2.5-3B",                      # Local relative path
    ]
    
    # Find the first valid path or download from Hugging Face if none exist
    valid_path = None
    if os.path.exists(model_path):
        valid_path = model_path
    else:
        for path in alternative_paths:
            if os.path.exists(path):
                valid_path = path
                break
    
    try:
        if valid_path:
            print(f"Loading model from local path: {valid_path}")
            ref_model = AutoModelForCausalLM.from_pretrained(
                valid_path,
                torch_dtype=torch.bfloat16,
                _attn_implementation="sdpa",
                trust_remote_code=True,
                local_files_only=True
            ).to('cuda')
        else:
            print("Local model not found. Attempting to download from Hugging Face...")
            ref_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B",  # This is the HF model ID
                torch_dtype=torch.bfloat16,
                _attn_implementation="sdpa",
                trust_remote_code=True
            ).to('cuda')
            print("Model successfully downloaded from Hugging Face")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying with explicit file path...")
        try:
            # Try with explicit file:// protocol
            ref_model = AutoModelForCausalLM.from_pretrained(
                "file://" + (valid_path or model_path),
                torch_dtype=torch.bfloat16,
                _attn_implementation="sdpa",
                trust_remote_code=True,
                local_files_only=True
            ).to('cuda')
        except Exception as nested_e:
            print(f"Second attempt failed: {nested_e}")
            raise

    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids):
        logits = ref_model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V)
        input_ids = input_ids[:, 1:]  # (B, L-1)
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()

    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        if len(dd) not in (3, 4): return b'tensor'
        data = {'base': json.loads(dd[0])}
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        if len(dd) == 4: data['gen_logps'] = bytes_to_tensor(dd[3])
        raw_queue.put(data)
        print('receive', data['inputs'].shape, data['rewards'], 
              data['gen_logps'].shape if 'gen_logps' in data else '')
        return b'tensor'

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        return result_queue.get()

    def run_server(): 
        try:
            bottle.run(app, host='0.0.0.0', port=59875, server='tornado')
        except Exception as e:
            print(f"Error starting server: {e}")
            # Try alternative port if 59875 is in use
            try:
                print("Attempting to use alternative port 59876...")
                bottle.run(app, host='0.0.0.0', port=59876, server='tornado')
            except Exception as e2:
                print(f"Failed to start server on alternative port: {e2}")

    print("Starting reference server...")
    threading.Thread(target=run_server, daemon=False).start()
    print("Server thread started")

    while True:
        try:
            d = raw_queue.get()
            prompt_length = d['base']['plen']
            with torch.inference_mode():
                per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
            per_token_logps = per_token_logps[:, prompt_length - 1:]
            data = [json.dumps(d['base']).encode(), tensor_to_bytes(d['inputs']),
                    tensor_to_bytes(d['rewards']), tensor_to_bytes(per_token_logps)]
            if 'gen_logps' in d: data.append(tensor_to_bytes(d['gen_logps']))
            xdata = make_bytes_list(data)
            result_queue.put(xdata)
        except Exception as e:
            print(f"Error processing queue item: {e}")
            # Continue processing even if one item fails
            continue