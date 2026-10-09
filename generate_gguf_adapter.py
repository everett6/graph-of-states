"""Run a custom GoS adapter GGUF with its original local NF4 Gemma backbone."""
import argparse
from pathlib import Path
import json
import gguf
import torch
from local_gemma import load_local_gemma
from experiment_support import load_gos_gguf


def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--base');p.add_argument('--max-new-tokens',type=int,default=128);a=p.parse_args()
    reader=gguf.GGUFReader(a.adapter);field=reader.fields['gos.metadata']
    meta=json.loads(field.parts[field.data[0]].tobytes().decode())
    base,tokenizer=load_local_gemma(a.base or meta['base_path']);model,_=load_gos_gguf(base,a.adapter);model.eval()
    text=tokenizer.apply_chat_template([{'role':'user','content':a.prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer(text,return_tensors='pt',add_special_tokens=False).to(model.device)
    if not 1<=a.max_new_tokens or inputs.input_ids.shape[1]+a.max_new_tokens>4096:raise ValueError('Total tokens must fit 4096')
    with torch.no_grad():out=model.generate(**inputs,use_cache=True,max_new_tokens=a.max_new_tokens,do_sample=False,logits_to_keep=1)
    print(tokenizer.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True))

if __name__=='__main__':main()
