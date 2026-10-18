"""Keep verified-task rows whose rendered prompt + answer fit a training sequence budget.

Uses the same rendering as benchmark_gos (Gemma chat template, thinking off, answer + EOS), so a filtered
file never trips its "Verified reference exceeds sequence budget" check. Rows are dropped, never truncated.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-path', required=True, help='Model directory with the tokenizer')
    parser.add_argument('--budget', type=int, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_path)
    kept = dropped = 0
    with open(args.input) as source, open(args.output, 'w') as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
            length = (len(tokenizer.encode(rendered, add_special_tokens=False)) +
                      len(tokenizer.encode(row['answer'], add_special_tokens=False)) + 1)
            if length <= args.budget:
                sink.write(line if line.endswith('\n') else line + '\n')
                kept += 1
            else:
                dropped += 1
    Path(args.output + '.manifest.json').write_text(json.dumps(
        {'input': args.input, 'budget': args.budget, 'kept': kept, 'dropped_over_budget': dropped}, indent=2) + '\n')
    print(json.dumps({'kept': kept, 'dropped_over_budget': dropped}))


if __name__ == '__main__':
    main()
