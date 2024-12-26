import torch
from model import GPTConfig, GPT
import pickle
import os
import time
init_from='resume'
device='cpu'
start_time=time.time()
checkpoint=torch.load('/home/guest/nanoGPT/out-shakespeare-char/ckpt.pt',weights_only=True)

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict,strict=True)
model.eval()
model.to(device)
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

start='hello'
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])



for k in range(10):
            y = model.generate(x, 200, temperature=0.8, top_k=200)
            print(decode(y[0].tolist()))
            print('---------------')
stop=time.time()
print("The time of execution of above program is :",
      ((stop-start_time) * 10**3)/1000, "ms")


##PS:comment out the weight pruning method when inferring``

