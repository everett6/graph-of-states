"""Uniformly sample assistant-supervised Fable 5.1 windows for a bounded experiment."""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rows', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    rng = random.Random(args.seed)
    reservoir = []
    eligible = 0
    windows = 0
    with open(args.input) as source:
        for line in source:
            row = json.loads(line)
            windows += 1
            if row.get('window_index', 0) != 0 or sum(label != -100 for label in row['labels'][1:]) < 32:
                continue
            eligible += 1
            item = {'input_ids': row['input_ids'], 'labels': row['labels'],
                    'source': 'fable', 'trace_id': row.get('trace_id', f'row:{windows}'), 'window_index': 0}
            if len(reservoir) < args.rows:
                reservoir.append(item)
            else:
                slot = rng.randrange(eligible)
                if slot < args.rows:
                    reservoir[slot] = item
    rng.shuffle(reservoir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in reservoir))
    print(json.dumps({'input_windows': windows, 'eligible_windows': eligible, 'sampled_rows': len(reservoir),
                      'seed': args.seed, 'output': str(output)}))


if __name__ == '__main__':
    main()
