"""Controlled adapters, resumable state, and faithful custom adapter GGUF I/O."""
import json, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from gos_gemma import attach_gos, inner_lm


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32=False


class MatchedMLP(nn.Module):
    """Two-layer residual MLP with exact used-parameter budget via partial biases."""
    def __init__(self, base, hidden, budget):
        super().__init__(); self.base=base
        width,remainder=divmod(budget,2*hidden)
        if width<1: raise ValueError('Budget too small for a two-layer MLP')
        self.down=nn.Linear(hidden,width,bias=False)
        self.up=nn.Linear(width,hidden,bias=False)
        self.input_bias=nn.Parameter(torch.zeros(min(remainder,hidden)))
        self.output_bias=nn.Parameter(torch.zeros(max(0,remainder-hidden)))
        nn.init.zeros_(self.up.weight)
    @property
    def weight(self): return self.base.weight
    def forward(self, hidden, **kwargs):
        x=hidden.to(self.down.weight.dtype)
        x=x+torch.nn.functional.pad(self.input_bias,(0,x.shape[-1]-self.input_bias.numel()))
        delta=self.up(torch.tanh(self.down(x)))
        delta=delta+torch.nn.functional.pad(self.output_bias,(0,x.shape[-1]-self.output_bias.numel()))
        return self.base(hidden+delta.to(hidden.dtype))


def trainable_count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure(base, variant, width=32, rounds=2, enhancements=None, lora_rank=2, lora_targets='qv'):
    base.requires_grad_(False)
    if variant=='frozen': return base
    if variant=='gos': return attach_gos(base,width,4,rounds,enhancements)
    if variant=='mlp':
        # Obtain the exact graph budget without allocating a second backbone.
        from gos_gemma import TokenGraph
        graph=TokenGraph(base.config.hidden_size,width,4,rounds,enhancements)
        budget=sum(p.numel() for p in graph.parameters())+width*base.config.hidden_size
        head=MatchedMLP(base.get_output_embeddings(),base.config.hidden_size,budget)
        head.to(device=base.device); base.set_output_embeddings(head)
        return base
    if variant=='lora':
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(base,LoraConfig(r=lora_rank,lora_alpha=2*lora_rank,lora_dropout=0,
            # 'all-linear' adapts attention and MLP projections (attention-only LoRA underperforms in RL).
            target_modules={'qv':['q_proj','v_proj'],'all-linear':'all-linear'}[lora_targets],task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        return model
    raise ValueError('Unknown variant')


def save_checkpoint(path,model,optimizer,step,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'format_version':1,'step':step,'metadata':metadata,
        'parameters':{n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
        'optimizer':optimizer.state_dict() if optimizer else None,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(state,temporary);os.replace(temporary,path)


def load_checkpoint(path,model,optimizer,metadata):
    state=torch.load(path,map_location='cpu',weights_only=False) # Only trusted local checkpoints.
    if state['format_version']!=1 or state['metadata']!=metadata: raise ValueError('Checkpoint configuration/data mismatch')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if parameters.keys()!=state['parameters'].keys() or any(parameters[n].shape!=v.shape for n,v in state['parameters'].items()):
        raise ValueError('Checkpoint parameter mismatch')
    with torch.no_grad():
        for n,v in state['parameters'].items(): parameters[n].copy_(v)
    if optimizer is not None: optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    if state['cuda_rng']: torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']


def export_gos_gguf(model,path,metadata):
    """Custom nonlinear adapter format; NOT a standalone llama.cpp model."""
    import gguf
    writer=gguf.GGUFWriter(str(path),'gos_adapter')
    topology={k:getattr(inner_lm(model).config,k,None) for k in ('hidden_size','vocab_size','num_hidden_layers','num_attention_heads')}
    writer.add_string('gos.metadata',json.dumps({'format_version':1,'base_topology':topology,**metadata},sort_keys=True))
    for name,p in model.named_parameters():
        if p.requires_grad: writer.add_tensor(name,p.detach().float().cpu().numpy())
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()


def load_gos_gguf(base,path):
    import gguf
    reader=gguf.GGUFReader(str(path))
    field=reader.fields['gos.metadata'];meta=json.loads(field.parts[field.data[0]].tobytes().decode())
    if meta['format_version']!=1: raise ValueError('Unsupported adapter GGUF version')
    topology={k:getattr(base.config,k,None) for k in meta['base_topology']}
    if topology!=meta['base_topology']: raise ValueError('Adapter GGUF backbone mismatch')
    if Path(meta.get('base_path','')).is_dir() and str(base.config._name_or_path)!=meta['base_path']:
        raise ValueError('Adapter GGUF requires the original backbone path')
    model=configure(base,'gos',meta['width'],meta['rounds'],meta['enhancements'])
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    tensors={t.name:t.data for t in reader.tensors}
    if parameters.keys()!=tensors.keys() or any(tuple(parameters[n].shape)!=tuple(v.shape) for n,v in tensors.items()):
        raise ValueError('Adapter GGUF architecture mismatch')
    with torch.no_grad():
        for name,value in tensors.items():parameters[name].copy_(torch.from_numpy(value.copy()))
    return model,meta
