"""End-to-end LRR CLI: prepare, train/resume, evaluate, generate, and offline demo."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from gos_engine_architecture import LRRConfig, LatentRecursiveReconstruction
from lrr_data import (ByteTokenizer, BuiltinFrozenBackbone, FeatureDataset, collate_features,
                      create_cache, fingerprint, make_backbone, read_records, split_records)


def device_name(requested):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable to this process. Check GPU access or select --device cpu.')
    return device


def emit(record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def prepare(args):
    if min(args.max_prompt_tokens, args.max_answer_tokens) < 1:
        raise ValueError('Token limits must be positive')
    records, stats = read_records(args.data)
    train_records, val_records = split_records(records, args.validation_fraction, args.seed)
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Output directory must be empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    device = device_name(args.device)
    if args.backbone == 'builtin':
        spec = {'kind': 'builtin', 'width': 64, 'seed': args.seed}
    else:
        spec = {'kind': 'huggingface', 'model_id': args.backbone, 'revision': args.revision,
                'local_files_only': args.local_files_only, 'quantize_4bit': args.quantize_4bit}
    backbone = make_backbone(spec, device)
    if spec['kind'] == 'huggingface':
        spec['revision'] = backbone.revision
    for name, subset in [('train', train_records), ('val', val_records)]:
        if subset:
            create_cache(subset, root / name, backbone, spec, args.max_prompt_tokens,
                         args.max_answer_tokens, args.truncate_prompts)
            (root / f'{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in subset), encoding='utf-8')
    report = {**stats, 'train_examples': len(train_records), 'val_examples': len(val_records),
              'device': device, 'output': str(root)}
    (root / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    emit(report)


def loader(dataset, batch_size, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), collate_fn=collate_features)


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def precision_context(device, precision):
    if precision == 'fp32':
        return nullcontext()
    if not device.startswith('cuda'):
        raise ValueError('Mixed precision requires CUDA; use --precision fp32 on CPU')
    return torch.autocast('cuda', dtype={'bf16': torch.bfloat16, 'fp16': torch.float16}[precision])


@torch.no_grad()
def evaluate_model(model, dataset, device, batch_size=8, max_new_tokens=128,
                   dynamic_halt=False, prediction_path=None):
    was_training = model.training
    model.eval()
    tokenizer = ByteTokenizer()
    loss_sum, tokens, correct_tokens, exact, examples, rounds = 0.0, 0, 0, 0, 0, 0
    predictions = []
    started = time.perf_counter()
    try:
        for batch in loader(dataset, batch_size):
            batch = to_device(batch, device)
            logits, _ = model(batch['hidden'], batch['mask'], batch['inputs'])
            labels = batch['labels']
            loss_sum += F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(),
                                        ignore_index=-100, reduction='sum').item()
            valid = labels.ne(-100)
            tokens += valid.sum().item()
            correct_tokens += ((logits.argmax(-1) == labels) & valid).sum().item()
            generated, diagnostics = model.generate(batch['hidden'], batch['mask'], 1, 2,
                                                      max_new_tokens, dynamic_halt=dynamic_halt)
            rounds += diagnostics['steps'].sum().item()
            for prompt, answer, ids in zip(batch['prompts'], batch['answers'], generated.tolist()):
                prediction = tokenizer.decode(ids)
                matched = prediction == answer
                exact += matched
                examples += 1
                predictions.append({'prompt': prompt, 'answer': answer, 'prediction': prediction,
                                    'exact_match': matched, 'ended_with_eos': 2 in ids})
    finally:
        model.train(was_training)
    metrics = {'loss': loss_sum / tokens, 'perplexity': math.exp(min(loss_sum / tokens, 50)),
               'token_accuracy': correct_tokens / tokens, 'exact_match': exact / examples,
               'examples': examples, 'mean_graph_rounds': rounds / examples,
               'elapsed_seconds': time.perf_counter() - started}
    if prediction_path:
        path = Path(prediction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in predictions), encoding='utf-8')
    return metrics


def check_features(metadata, other):
    if metadata['feature_signature'] != other['feature_signature']:
        raise ValueError('Feature caches use different backbones, tokenizers, or prompt settings')


def load_checkpoint(path, device):
    # Optimizer/RNG tensors are loaded on CPU first; model moves explicitly.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Expected a full checkpoint written by the current train command')
    model = LatentRecursiveReconstruction(LRRConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model_state'])
    return model, checkpoint


def train(args):
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.width, args.nodes,
           args.rounds, args.max_new_tokens) < 1 or args.learning_rate <= 0 or args.warmup_steps < 0:
        raise ValueError('Training sizes/rate must be positive and warmup nonnegative')
    if args.weight_decay < 0 or args.clip_grad_norm <= 0:
        raise ValueError('Weight decay must be nonnegative and gradient clipping positive')
    device = device_name(args.device)
    if args.precision != 'fp32' and not device.startswith('cuda'):
        raise ValueError('Choose --precision fp32 on CPU')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    dataset = FeatureDataset(args.features)
    validation = FeatureDataset(args.validation_features) if args.validation_features else None
    if validation:
        check_features(dataset.metadata, validation.metadata)
        if dataset.metadata['records_digest'] == validation.metadata['records_digest']:
            raise ValueError('Training and validation datasets must differ')
        train_prompts = {dataset[index]['prompt'].strip() for index in range(len(dataset))}
        if any(validation[index]['prompt'].strip() in train_prompts for index in range(len(validation))):
            raise ValueError('Prompt leakage between training and validation datasets')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Training output must be empty unless --resume is supplied')
    output.mkdir(parents=True, exist_ok=True)
    model = LatentRecursiveReconstruction(LRRConfig(backbone_dim=dataset.metadata['backbone_dim'],
        vocab_size=dataset.metadata['vocab_size'], width=args.width, nodes=args.nodes, rounds=args.rounds,
        min_rounds=min(2, args.rounds), checkpoint_rounds=args.checkpoint_rounds)).to(device)
    settings = {key: getattr(args, key) for key in ('batch_size', 'accumulation_steps',
        'learning_rate', 'weight_decay', 'warmup_steps', 'seed', 'precision', 'clip_grad_norm')}
    validation_digest = validation.metadata['records_digest'] if validation else None
    start_epoch, global_step, best_loss = 0, 0, float('inf')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.startswith('cuda') and args.precision == 'fp16'))
    builtin_state = None
    if dataset.metadata['backbone']['kind'] == 'builtin':
        builtin_state = torch.load(dataset.root / 'backbone.pt', weights_only=True, map_location='cpu')
    if args.resume:
        restored, checkpoint = load_checkpoint(args.resume, device)
        check_features(dataset.metadata, checkpoint['metadata'])
        if dataset.metadata['records_digest'] != checkpoint['metadata']['records_digest']:
            raise ValueError('Resume requires the same training dataset')
        if settings != checkpoint['training_settings']:
            raise ValueError('Resume requires the same optimizer/batch/precision settings')
        if validation_digest != checkpoint.get('validation_digest'):
            raise ValueError('Resume requires the same validation dataset')
        model = restored
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch, global_step, best_loss = checkpoint['epoch'], checkpoint['global_step'], checkpoint['best_loss']
        torch.set_rng_state(checkpoint['rng_state'])
        if device.startswith('cuda') and checkpoint['cuda_rng_state']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        source_best = Path(args.resume).parent / 'best.pt'
        if output.resolve() != Path(args.resume).parent.resolve() and source_best.exists():
            shutil.copyfile(source_best, output / 'best.pt')
    if args.epochs <= start_epoch:
        raise ValueError('--epochs is the total target and must exceed the checkpoint epoch')
    run_config = {**{key: value for key, value in vars(args).items() if key != 'function'},
                  'device': device, 'config': asdict(model.config),
                  'trainable_parameters': sum(p.numel() for p in model.parameters())}
    (output / 'run.json').write_text(json.dumps(run_config, indent=2) + '\n')
    emit({'event': 'training_started', 'device': device, 'train_examples': len(dataset),
          'parameters': run_config['trainable_parameters'], 'start_epoch': start_epoch})
    for epoch in range(start_epoch, args.epochs):
        model.train()
        batches = iter(loader(dataset, args.batch_size, shuffle=True, seed=args.seed + epoch))
        total_loss, total_tokens = 0.0, 0
        started = time.perf_counter()
        while True:
            window = []
            for _ in range(args.accumulation_steps):
                batch = next(batches, None)
                if batch is None:
                    break
                window.append(batch)
            if not window:
                break
            token_count = sum(batch['labels'].ne(-100).sum().item() for batch in window)
            optimizer.zero_grad(set_to_none=True)
            for cpu_batch in window:
                batch = to_device(cpu_batch, device)
                with precision_context(device, args.precision):
                    logits, _ = model(batch['hidden'], batch['mask'], batch['inputs'])
                    loss_sum = F.cross_entropy(logits.flatten(0, 1).float(), batch['labels'].flatten(),
                                               ignore_index=-100, reduction='sum')
                if not torch.isfinite(loss_sum):
                    raise RuntimeError('Non-finite loss; retry fp32 or a lower learning rate')
                scaler.scale(loss_sum / token_count).backward()
                total_loss += loss_sum.detach().item()
                total_tokens += batch['labels'].ne(-100).sum().item()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite gradient; checkpoint from previous epoch is intact')
            rate = args.learning_rate * min(1.0, (global_step + 1) / max(1, args.warmup_steps))
            for group in optimizer.param_groups:
                group['lr'] = rate
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
        record = {'epoch': epoch + 1, 'global_step': global_step, 'training_loss': total_loss / total_tokens,
                  'learning_rate': rate, 'elapsed_seconds': time.perf_counter() - started}
        if validation:
            record['validation'] = evaluate_model(model, validation, device, args.batch_size,
                                                   args.max_new_tokens)
        selected_loss = record['validation']['loss'] if validation else record['training_loss']
        improved = selected_loss < best_loss
        best_loss = min(best_loss, selected_loss)
        if device.startswith('cuda'):
            record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        checkpoint = {'format_version': 1, 'config': asdict(model.config), 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'scaler_state': scaler.state_dict(),
            'epoch': epoch + 1, 'global_step': global_step, 'best_loss': best_loss,
            'metadata': dataset.metadata, 'builtin_backbone_state': builtin_state,
            'validation_digest': validation_digest,
            'training_settings': settings, 'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
            'metrics': record}
        atomic_save(checkpoint, output / 'last.pt')
        if improved:
            atomic_save(checkpoint, output / 'best.pt')
        with (output / 'metrics.jsonl').open('a') as history:
            history.write(json.dumps(record) + '\n')
        emit(record)
    return output / 'last.pt'


def evaluate(args):
    device = device_name(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = FeatureDataset(args.features)
    check_features(checkpoint['metadata'], dataset.metadata)
    report = evaluate_model(model, dataset, device, args.batch_size, args.max_new_tokens,
                            args.dynamic_halt, args.predictions)
    report['checkpoint'] = str(args.checkpoint)
    emit(report)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def generate(args):
    device = device_name(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    metadata = checkpoint['metadata']
    spec = dict(metadata['backbone'])
    if args.backbone_device:
        backbone_device = device_name(args.backbone_device)
    else:
        backbone_device = device
    backbone = make_backbone(spec, backbone_device, checkpoint['builtin_backbone_state'])
    hidden, mask = backbone.extract([args.prompt], metadata['max_prompt_tokens'],
                                    truncate=metadata['truncate_prompts'])
    # Release backbone before graph generation; cached path can train without it.
    del backbone
    if backbone_device.startswith('cuda'):
        torch.cuda.empty_cache()
    # Match cache storage precision before feeding the graph.
    hidden, mask = hidden.to(torch.float16).float().to(device), mask.to(device)
    ids, diagnostics = model.generate(hidden, mask, 1, 2, args.max_new_tokens,
                                       dynamic_halt=args.dynamic_halt)
    emit({'prompt': args.prompt, 'answer': ByteTokenizer().decode(ids[0].tolist()),
          'graph_rounds': int(diagnostics['steps'][0]), 'ended_with_eos': 2 in ids[0].tolist()})


def demo(args):
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Demo output directory must be empty')
    root.mkdir(parents=True, exist_ok=True)
    # A deterministic, locally generated truth-table task. Held-out prompts have
    # unseen wording; this tests the pipeline, not broad reasoning competence.
    records = []
    for operation in ('AND', 'OR', 'XOR'):
        for left in (0, 1):
            for right in (0, 1):
                answer = {'AND': left & right, 'OR': left | right, 'XOR': left ^ right}[operation]
                for template in ('{a} {op} {b} =', 'Compute {a} {op} {b}.', 'Solve: {a} {op} {b}', 'Result of {a} {op} {b}?'):
                    records.append({'prompt': template.format(a=left, b=right, op=operation), 'answer': str(answer)})
    data = root / 'logic.jsonl'
    data.write_text(''.join(json.dumps(record) + '\n' for record in records))
    prepared = root / 'features'
    prepare(argparse.Namespace(data=str(data), output=str(prepared), backbone='builtin',
        revision=None, local_files_only=True, quantize_4bit=False, device=args.device,
        validation_fraction=0.2, seed=17, max_prompt_tokens=64, max_answer_tokens=8, truncate_prompts=False))
    training_args = build_parser().parse_args(['train', '--features', str(prepared / 'train'),
        '--validation-features', str(prepared / 'val'), '--output', str(root / 'training'),
        '--epochs', str(args.epochs), '--width', '64', '--rounds', '3', '--batch-size', '8',
        '--learning-rate', '0.003', '--max-new-tokens', '4', '--device', args.device,
        '--precision', args.precision])
    checkpoint = train(training_args)
    checkpoint = checkpoint.parent / 'best.pt'
    evaluate(argparse.Namespace(checkpoint=str(checkpoint), features=str(prepared / 'val'),
        device=args.device, batch_size=8, max_new_tokens=4, dynamic_halt=False,
        predictions=str(root / 'predictions.jsonl'), output=str(root / 'evaluation.json')))
    generate(argparse.Namespace(checkpoint=str(checkpoint), device=args.device, backbone_device=None,
        prompt='1 AND 0 =', max_new_tokens=4, dynamic_halt=False))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=positive_int, default=4, help='CPU worker threads (default: 4)')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', aliases=['cache'], help='Validate, split, and cache JSONL features')
    p.add_argument('--data', required=True)
    p.add_argument('--backbone', default='builtin', help='builtin, local model path, or Hugging Face model ID')
    p.add_argument('--revision')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--quantize-4bit', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--validation-fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-prompt-tokens', type=positive_int, default=256)
    p.add_argument('--max-answer-tokens', type=positive_int, default=256)
    p.add_argument('--truncate-prompts', action='store_true')
    p.set_defaults(function=prepare)
    t = commands.add_parser('train')
    t.add_argument('--features', required=True)
    t.add_argument('--validation-features')
    t.add_argument('--output', required=True)
    t.add_argument('--resume', help='Full checkpoint; --epochs is the total target epoch count')
    t.add_argument('--device', default='auto')
    t.add_argument('--width', type=positive_int, default=128)
    t.add_argument('--nodes', type=positive_int, default=4)
    t.add_argument('--rounds', type=positive_int, default=4)
    t.add_argument('--epochs', type=positive_int, default=20)
    t.add_argument('--batch-size', type=positive_int, default=4)
    t.add_argument('--accumulation-steps', type=positive_int, default=1)
    t.add_argument('--learning-rate', type=float, default=0.001)
    t.add_argument('--weight-decay', type=float, default=0.01)
    t.add_argument('--warmup-steps', type=int, default=0)
    t.add_argument('--clip-grad-norm', type=float, default=1.0)
    t.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    t.add_argument('--seed', type=int, default=7)
    t.add_argument('--checkpoint-rounds', action='store_true')
    t.add_argument('--max-new-tokens', type=positive_int, default=128)
    t.set_defaults(function=train)
    e = commands.add_parser('evaluate')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--features', required=True)
    e.add_argument('--device', default='auto')
    e.add_argument('--batch-size', type=positive_int, default=8)
    e.add_argument('--max-new-tokens', type=positive_int, default=128)
    e.add_argument('--dynamic-halt', action='store_true')
    e.add_argument('--predictions')
    e.add_argument('--output')
    e.set_defaults(function=evaluate)
    g = commands.add_parser('generate')
    g.add_argument('--checkpoint', required=True)
    g.add_argument('--prompt', required=True)
    g.add_argument('--device', default='auto')
    g.add_argument('--backbone-device', help='Optional separate device for feature extraction')
    g.add_argument('--max-new-tokens', type=positive_int, default=128)
    g.add_argument('--dynamic-halt', action='store_true')
    g.set_defaults(function=generate)
    d = commands.add_parser('demo', help='Run an offline logic dataset through the entire workflow')
    d.add_argument('--output', default='runs/demo')
    d.add_argument('--device', default='auto')
    d.add_argument('--epochs', type=positive_int, default=20)
    d.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    d.set_defaults(function=demo)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    try:
        args.function(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
