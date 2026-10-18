import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from self_distill import distill  # noqa: E402


def test_keeps_shortest_verified_solution_that_fits():
    rows = [{'trace_id': 'a', 'prompt': 'p1', 'task_spec': {'kind': 'math', 'expected': '4'}, 'source': 'math_train'},
            {'trace_id': 'b', 'prompt': 'p2', 'task_spec': '{"kind": "math", "expected": "9"}'},
            {'trace_id': 'c', 'prompt': 'p3', 'task_spec': {'kind': 'math', 'expected': '1'}}]
    outputs = {'p1': [('long work so \\boxed{4}', 50), ('short \\boxed{4}', 10), ('wrong \\boxed{5}', 5)],
               'p2': [('too long \\boxed{9}', 900)], 'p3': [('\\boxed{1}', 3)]}
    calls = []

    def generate(messages, count, sample):
        calls.append(messages[0]['content'])
        assert sample and count == 3
        return outputs[messages[0]['content']]

    results = dict(distill(rows, generate, 3, lambda prompt, answer: 'too long' not in answer, done={'c'}))
    assert calls == ['p1', 'p2'] and set(results) == {'a', 'b'}
    assert results['b'] is None
    assert results['a']['answer'] == 'short \\boxed{4}' and results['a']['correct_samples'] == 2
    assert results['a']['source'] == 'selfdistill_math_train' and results['a']['task_spec']['expected'] == '4'
