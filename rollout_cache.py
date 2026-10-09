"""Bounded CPU LRU for deterministic frozen features, never adapter logits."""
from collections import OrderedDict
import hashlib
import torch
from gos_gemma import frozen_features, inner_lm


class FrozenFeatureCache:
    def __init__(self,entries=8,max_bytes=64*1024*1024):
        if entries<1 or max_bytes<1: raise ValueError('Invalid cache budget')
        self.entries,self.max_bytes=entries,max_bytes
        self.items=OrderedDict();self.bytes=0;self.hits=0;self.misses=0

    def clear(self):
        self.items.clear();self.bytes=0

    def get(self,model,ids,mask,max_length):
        backbone=inner_lm(model).base_model
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError('Feature caching requires a fully frozen backbone')
        key=(id(backbone),str(next(backbone.parameters()).dtype),max_length,tuple(ids.shape),
             hashlib.sha256(ids.detach().cpu().numpy().tobytes()+mask.detach().cpu().numpy().tobytes()).digest())
        if key in self.items:
            self.hits+=1;value=self.items.pop(key);self.items[key]=value
            return value.to(ids.device)
        self.misses+=1
        value=frozen_features(model,ids,mask,max_length)
        size=value.numel()*value.element_size()
        if size<=self.max_bytes:
            while self.items and (len(self.items)>=self.entries or self.bytes+size>self.max_bytes):
                _,old=self.items.popitem(last=False);self.bytes-=old.numel()*old.element_size()
            self.items[key]=value.detach().cpu().clone();self.bytes+=size
        return value
