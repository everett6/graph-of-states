"""Read local legacy Gemma4 unified text weights without remote code/downloads."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, Gemma4TextConfig, Gemma4ForCausalLM, BitsAndBytesConfig


def load_local_gemma(path, nf4=True):
    path = Path(path)
    raw = json.loads((path / 'config.json').read_text())
    text = dict(raw.get('text_config', raw))
    for key in ('model_type', 'architectures', 'dtype', 'torch_dtype'):
        text.pop(key, None)
    config = Gemma4TextConfig(**text)
    kwargs = dict(config=config, local_files_only=True, dtype=torch.bfloat16,
                  attn_implementation='sdpa')
    if raw['model_type'] == 'gemma4_unified':
        kwargs['key_mapping'] = {r'^model\.language_model\.': 'model.'}
    if nf4:
        kwargs.update(device_map={'': torch.cuda.current_device()}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['lm_head']))
    model, info = Gemma4ForCausalLM.from_pretrained(path, output_loading_info=True, **kwargs)
    missing = [key for key in info.get('missing_keys', []) if not (key == 'lm_head.weight' and config.tie_word_embeddings)]
    if missing or info.get('mismatched_keys'):
        raise RuntimeError(f'Incompatible local text weights: missing={missing}, mismatched={info.get("mismatched_keys")}')
    model.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return model, tokenizer


def load_causal_lm(path, nf4=True):
    """Local Gemma 4 via load_local_gemma; any other local causal LM (e.g. a Qwen-based prover) via Auto*."""
    path = Path(path)
    raw = json.loads((path / 'config.json').read_text())
    if raw.get('model_type', '').startswith('gemma4'):
        return load_local_gemma(path, nf4)
    from transformers import AutoModelForCausalLM
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    if nf4:
        kwargs.update(device_map={'': torch.cuda.current_device()}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return model, tokenizer
