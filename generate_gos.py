"""Generate with a locally cached NF4 backbone and a trained GoS adapter."""
import argparse
import json
from pathlib import Path

import torch

from gos_gemma import load_gos_adapter
from gos_h2o import h2o_generate
from train_unified import UnifiedConfig, load_nf4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--cache', choices=['none', 'h2o', 'sliding'], default='h2o')
    parser.add_argument('--heavy-tokens', type=int, default=256)
    parser.add_argument('--recent-tokens', type=int, default=256)
    parser.add_argument('--kv-quantization', choices=['none', 'int4'], default='int4')
    parser.add_argument('--kv-kernel', choices=['torch', 'triton'], default='triton')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--tools', action='store_true', help='Enable actuator-gated container tool turns')
    parser.add_argument('--tool-turns', type=int, default=4)
    parser.add_argument('--tool-threshold', type=float, default=0.85)
    parser.add_argument('--allow-model-download', action='store_true')
    args = parser.parse_args()
    config = UnifiedConfig(**json.loads((Path(args.adapter) / 'gos_config.json').read_text())).validate()
    base, tokenizer = load_nf4(config, args.allow_model_download)
    model = load_gos_adapter(base, args.adapter).eval()
    if args.tools:
        from gos_tools import gated_tool_turns
        from gos_gemma import frozen_features, inner_lm
        def encode(messages, generation=True):
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation,
                return_tensors='pt').to(next(model.parameters()).device)
        @torch.no_grad()
        def generate(messages):
            ids = encode(messages)
            if ids.shape[1] + args.max_new_tokens > config.max_seq_length:
                raise ValueError('Tool feedback exceeds the context budget; reduce tool-turns or completion size')
            if args.cache in ('h2o', 'sliding'):
                output, _ = h2o_generate(model, ids, args.max_new_tokens, args.heavy_tokens, args.recent_tokens,
                    max_seq_length=config.max_seq_length, temperature=args.temperature,
                    kv_quantization=args.kv_quantization, kernel=args.kv_kernel, policy=args.cache)
            else:
                kwargs = {'do_sample': args.temperature > 0}
                if args.temperature > 0:
                    kwargs.update(temperature=args.temperature, top_k=0, top_p=1.0)
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens,
                    use_cache=False, logits_to_keep=1, **kwargs)
            return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
        @torch.no_grad()
        def gate(messages, text):
            # Same final-assistant-position supervision convention as tool data.
            ids = encode(messages + [{'role': 'assistant', 'content': text}], generation=False)
            hidden = frozen_features(model, ids, torch.ones_like(ids), config.max_seq_length)
            return inner_lm(model).get_output_embeddings().tool_logits(hidden[:, -1]).sigmoid().item()
        text, _, records = gated_tool_turns([{'role': 'user', 'content': args.prompt}], generate, gate,
            threshold=args.tool_threshold, max_turns=args.tool_turns)
        print(text)
        print(json.dumps({'tool_executions': len(records), 'outcomes': records}))
        return
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors='pt').to(next(model.parameters()).device)
    if ids.shape[1] + args.max_new_tokens > config.max_seq_length:
        parser.error('Prompt + completion exceeds the configured sequence limit')
    if args.max_new_tokens < 1 or args.temperature < 0:
        parser.error('Use positive max-new-tokens and nonnegative temperature')
    with torch.no_grad():
        if args.cache in ('h2o', 'sliding'):
            output, stats = h2o_generate(model, ids, args.max_new_tokens, args.heavy_tokens,
                args.recent_tokens, max_seq_length=config.max_seq_length, temperature=args.temperature,
                kv_quantization=args.kv_quantization, kernel=args.kv_kernel, policy=args.cache)
        else:
            kwargs = {'do_sample': args.temperature > 0}
            if args.temperature > 0:
                kwargs.update(temperature=args.temperature, top_k=0, top_p=1.0)
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens, use_cache=False, logits_to_keep=1, **kwargs)
            stats = {'cache_policy': 'none'}
    print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
