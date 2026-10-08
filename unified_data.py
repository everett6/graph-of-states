"""Lazy dataset sources, conversation grouping, and bounded assistant-only windows."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


DEFAULT_SOURCES = [
    {'name': 'stratos', 'repo': 'bespokelabs/Bespoke-Stratos-17k', 'split': 'train', 'weight': 2},
    {'name': 'fable', 'repo': 'MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x', 'split': 'full', 'weight': 1},
    {'name': 'feedback', 'repo': 'm-a-p/CodeFeedback-Filtered-Instruction', 'split': 'train', 'weight': 1},
]


def content_text(content):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError('Expected text or a list of content blocks')
    pieces = []
    for block in content:
        kind = block.get('type')
        if kind in ('text', 'input_text', 'output_text'):
            pieces.append(block.get('text', ''))
        elif kind in ('thinking', 'reasoning'):
            pieces.append('<think>\n' + block.get('thinking', block.get('text', '')) + '\n</think>')
        elif kind in ('tool_use', 'tool_call', 'tool_result'):
            pieces.append(json.dumps(block, sort_keys=True, ensure_ascii=False))
        else:
            raise ValueError(f'Unsupported non-text content block: {kind!r}')
    return '\n'.join(pieces)


def normalize_row(row):
    raw = row.get('messages', row.get('conversations'))
    if raw is None and row.get('tool_interactions'):
        raw = [{'role': 'user', 'content': row.get('query', row.get('prompt', ''))}]
        for interaction in row['tool_interactions']:
            command = ({'name': 'python', 'code': interaction['code']} if isinstance(interaction.get('code'), str)
                       else {'name': interaction.get('latent_action_intent', 'EXECUTE_SANDBOX_COMPILER'),
                             'command': interaction.get('tool_command', '')})
            raw.append({'role': 'assistant', 'content': '<tool_call>' + json.dumps(command) + '</tool_call>', 'tool_action': True})
            result = {'role': 'tool', 'content': interaction.get('sandbox_raw_return', '')}
            for key in ('success', 'is_error'):
                if key in interaction:
                    result[key] = interaction[key]
            raw.append(result)
        answer = row.get('answer', row.get('response'))
        if isinstance(answer, str) and answer:
            raw.append({'role': 'assistant', 'content': answer, 'tool_action': False})
    if raw is None:
        prompt = row.get('query', row.get('prompt', row.get('instruction')))
        answer = row.get('answer', row.get('response', row.get('output')))
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError('Row needs messages, conversations, or a prompt/answer pair')
        if row.get('input'):
            prompt += '\n' + content_text(row['input'])
        raw = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
    messages = []
    if isinstance(row.get('system'), str) and row['system']:
        messages.append({'role': 'system', 'content': row['system']})
    aliases = {'human': 'user', 'gpt': 'assistant', 'model': 'assistant', 'function': 'tool'}
    for item in raw:
        role = item.get('role', item.get('from'))
        role = aliases.get(role, role)
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError(f'Unsupported role {role!r}')
        text = content_text(item.get('content', item.get('value', '')))
        reasoning = item.get('reasoning_content', item.get('reasoning'))
        if reasoning and role == 'assistant':
            text = '<think>\n' + content_text(reasoning) + '\n</think>\n' + text
        if item.get('tool_calls'):
            text += '\n[Tool calls]\n' + json.dumps(item['tool_calls'], sort_keys=True, ensure_ascii=False)
        message = {'role': role, 'content': text}
        if type(item.get('tool_action')) is bool:
            message['tool_action'] = item['tool_action']
        elif item.get('tool_calls'):
            message['tool_action'] = True
        # Only explicit outcomes supervise friction; no syntax/substring guesses.
        if type(item.get('is_error')) is bool:
            message['failure'] = float(item['is_error'])
        elif type(item.get('success')) is bool:
            message['failure'] = float(not item['success'])
        elif isinstance(item.get('content'), list):
            outcomes = [block['is_error'] for block in item['content']
                        if block.get('type') == 'tool_result' and type(block.get('is_error')) is bool]
            if outcomes:
                message['failure'] = float(any(outcomes))
                if all(block.get('type') == 'tool_result' for block in item['content']):
                    message['role'] = 'tool'
        messages.append(message)
    if not any(m['role'] == 'user' and m['content'].strip() for m in messages):
        raise ValueError('Missing user task')
    if not any(m['role'] == 'assistant' and m['content'].strip() for m in messages):
        raise ValueError('Missing assistant target')
    return messages


def task_key(messages):
    # Source-independent task grouping: all windows, revisions, and tool turns
    # with the same initial user task stay in one split, even across datasets.
    first = next(message['content'] for message in messages if message['role'] == 'user')
    canonical = ' '.join(first.split())
    return hashlib.sha256(canonical.encode()).hexdigest()


def task_split(key, seed):
    bucket = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def source_rows(source, allow_remote=False):
    """No source is opened until iteration. Streaming still transfers remote bytes."""
    if source.get('path'):
        with Path(source['path']).open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    if not allow_remote:
        raise ValueError('Remote streaming is disabled. Set local source paths or explicitly pass --allow-remote-data.')
    from datasets import load_dataset
    yield from load_dataset(source['repo'], name=source.get('config'), split=source.get('split', 'train'),
                            revision=source.get('revision'), streaming=True)


def template_messages(messages):
    # Tool calls/results remain verbatim text. No native tool invocation or code
    # execution is implied, and plain Gemma chat templates can consume the trace.
    result = []
    for item in messages:
        role = 'user' if item['role'] == 'tool' else item['role']
        text = ('[Tool result]\n' if item['role'] == 'tool' else '') + item['content']
        if result and result[-1]['role'] == role:
            result[-1]['content'] += '\n' + text
        else:
            result.append({'role': role, 'content': text})
    return result


def assistant_tokens(messages, tokenizer):
    normalized = template_messages(messages)
    rendered = tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    spans = []
    # Prefix validation avoids assuming that arbitrary role markers tokenize to
    # fixed IDs. It fails explicitly for templates that rewrite previous turns.
    for index, message in enumerate(normalized):
        if message['role'] != 'assistant':
            continue
        prefix = tokenizer.apply_chat_template(normalized[:index], tokenize=False, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(normalized[:index+1], tokenize=False, add_generation_prompt=False)
        if not rendered.startswith(prefix) or not rendered.startswith(through):
            raise ValueError('Chat template is not prefix-stable; provide a compatible text-only template')
        spans.append((len(prefix), len(through)))
    labels = [token if any(end > start and start >= a and end <= b for a, b in spans) else -100
              for token, (start, end) in zip(ids, offsets)]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError('No assistant tokens survived tokenization')
    # Outcome targets attach to the last token of the attempt preceding an
    # explicitly labeled tool result. Other tokens remain unsupervised.
    friction = [-100.0] * len(ids)
    assistant_index = -1
    for index, message in enumerate(messages):
        if message['role'] == 'assistant':
            assistant_index = index
        if message['role'] == 'tool' and 'failure' in message and assistant_index >= 0:
            prior = template_messages(messages[:assistant_index+1])
            through = tokenizer.apply_chat_template(prior, tokenize=False, add_generation_prompt=False)
            if rendered.startswith(through):
                positions = [i for i, (_, end) in enumerate(offsets)
                             if end <= len(through) and labels[i] != -100]
                if positions:
                    friction[positions[-1]] = message['failure']
    return ids, labels, friction


def actuator_targets(messages, tokenizer, ids, labels):
    targets = [-100.0] * len(ids)
    rendered = tokenizer.apply_chat_template(template_messages(messages), tokenize=False, add_generation_prompt=False)
    offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    for index, message in enumerate(messages):
        if message['role'] != 'assistant' or 'tool_action' not in message:
            continue
        through = tokenizer.apply_chat_template(template_messages(messages[:index+1]), tokenize=False, add_generation_prompt=False)
        if not rendered.startswith(through):
            raise ValueError('Non-prefix-stable tool trace')
        positions = [i for i, (_, end) in enumerate(offsets) if end <= len(through) and labels[i] != -100]
        if positions:
            targets[positions[-1]] = float(message['tool_action'])
    return targets


def token_windows(ids, labels, friction, length=4096, overlap=256, extra=None):
    if not 2 <= length <= 4096 or not 0 <= overlap < length - 1:
        raise ValueError('Invalid context/overlap or context above the hard cap of 4096')
    for start in range(0, len(ids), length - overlap):
        stop = min(start + length, len(ids))
        target = labels[start:stop].copy()
        risks = friction[start:stop].copy()
        prefix = min(max(1, overlap) if start else 1, len(target))
        target[:prefix] = [-100] * prefix
        risks[:prefix] = [-100.0] * prefix
        extras = {key: values[start:stop] for key, values in (extra or {}).items()}
        if 'tool_targets' in extras:
            extras['tool_targets'][:prefix] = [-100.0] * prefix
        if any(value != -100 for value in target[1:]):
            yield {'input_ids': ids[start:stop], 'attention_mask': [1] * (stop-start),
                   'labels': target, 'friction_targets': risks, **extras}
        if stop == len(ids):
            break


def grpo_examples(messages, tokenizer, prompt_limit):
    for index, message in enumerate(messages):
        if message['role'] != 'assistant' or not message['content'].strip():
            continue
        prior = template_messages(messages[:index])
        if not any(item['role'] == 'user' for item in prior):
            continue
        prompt = tokenizer.apply_chat_template(prior, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > prompt_limit:
            # A dropped task invalidates rewards. Skip rather than silently
            # left-truncating a long agent transcript into a different task.
            continue
        yield {'prompt': prompt, 'answer': message['content']}


def mixed_examples(sources, tokenizer, split, seed=17, mode='sft', max_seq_length=4096,
                   overlap=256, prompt_limit=3072, limit=None, allow_remote=False, stats=None):
    """Exact source ratios per schedule cycle; hash splits are 80/10/10 in expectation."""
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split')
    stats = stats if stats is not None else Counter()
    schedule = [source['name'] for source in sources for _ in range(source['weight'])]
    by_name = {source['name']: source for source in sources}
    if not schedule or len(by_name) != len(sources):
        raise ValueError('Sources need unique names and positive integer weights')

    def examples(source):
        seen = set()
        for row in source_rows(source, allow_remote):
            stats['rows_scanned'] += 1
            try:
                messages = normalize_row(row)
                key = task_key(messages)
                if task_split(key, seed) != split:
                    continue
                identity = {'messages': messages}
                if 'modal_states' in row:
                    identity.update(modal_states=row['modal_states'], modal_mask=row.get('modal_mask'))
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                if digest in seen:
                    stats['duplicates'] += 1
                    continue
                seen.add(digest)
                if mode == 'sft':
                    ids, labels, friction = assistant_tokens(messages, tokenizer)
                    extra = {'tool_targets': actuator_targets(messages, tokenizer, ids, labels)}
                    if 'modal_states' in row:
                        if len(row['modal_states']) != len(ids) or len(row.get('modal_mask', [])) != len(ids):
                            raise ValueError('Modal features must align with the rendered/tokenized conversation')
                        extra.update(modal_states=row['modal_states'], modal_mask=row['modal_mask'])
                    prepared = token_windows(ids, labels, friction, max_seq_length, overlap, extra)
                else:
                    prepared = grpo_examples(messages, tokenizer, prompt_limit)
                for example in prepared:
                    yield {**example, 'task_key': key, 'source': source['name'],
                           'tests': row.get('tests')}
            except (ValueError, TypeError, KeyError) as error:
                stats['rejected'] += 1
                # A missing template is configuration failure, not bad data.
                if getattr(tokenizer, 'chat_template', None) is None:
                    raise ValueError('Tokenizer must have a compatible chat template') from error

    streams = {name: iter(examples(source)) for name, source in by_name.items()}
    count = 0
    while limit is None or count < limit:
        for name in schedule:
            example = next(streams[name], None)
            if example is None:
                if split != 'train':
                    return  # Finite evaluation never repeats examples.
                streams[name] = iter(examples(by_name[name]))
                example = next(streams[name], None)
                if example is None:
                    raise ValueError(f'No usable {split} examples for source {name}; check split/schema/limits')
            stats[f'emitted_{name}'] += 1
            count += 1
            yield example
            if limit is not None and count >= limit:
                return


class UnifiedCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        length = max(len(item['input_ids']) for item in features)
        if length > 4096:
            raise ValueError('Sequence exceeds hard cap')
        pads = {'input_ids': self.pad_id, 'attention_mask': 0, 'labels': -100, 'friction_targets': -100.0}
        result = {key: torch.tensor([item[key] + [pad] * (length-len(item[key])) for item in features],
                                  dtype=torch.float32 if key == 'friction_targets' else torch.long)
                for key, pad in pads.items()}
        if any('tool_targets' in item for item in features):
            result['tool_targets'] = torch.tensor([item.get('tool_targets', [-100.0] * len(item['input_ids'])) +
                [-100.0] * (length - len(item['input_ids'])) for item in features], dtype=torch.float32)
        if any('modal_states' in item for item in features):
            exemplar = next(item for item in features if 'modal_states' in item)
            sample = torch.as_tensor(exemplar['modal_states'], dtype=torch.float32)
            if sample.ndim != 3 or not torch.isfinite(sample).all():
                raise ValueError('modal_states must be finite [tokens, modalities, hidden_width]')
            result['modal_states'] = torch.zeros((len(features), length, *sample.shape[1:]))
            result['modal_mask'] = torch.zeros((len(features), length, sample.shape[1]), dtype=torch.bool)
            for index, item in enumerate(features):
                if 'modal_states' in item:
                    values = torch.as_tensor(item['modal_states'], dtype=torch.float32)
                    masks = torch.as_tensor(item['modal_mask'], dtype=torch.bool)
                    if values.shape != (len(item['input_ids']), *sample.shape[1:]) or masks.shape != values.shape[:2] or not torch.isfinite(values).all():
                        raise ValueError('Inconsistent modality shapes or nonfinite states')
                    result['modal_states'][index, :len(values)] = values
                    result['modal_mask'][index, :len(values)] = masks
        return result
