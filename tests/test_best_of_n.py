import json

import pytest

import best_of_n
from best_of_n import evaluate_row, run
from eval_search import majority_vote, pass_at_k, row_metrics, summarize, wilson
from verified_rewards import format_only_reward, random_control_reward


def test_pass_at_k_matches_closed_form():
    assert pass_at_k(4, 0, 1) == 0.0
    assert pass_at_k(4, 4, 4) == 1.0
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    assert pass_at_k(5, 2, 3) == pytest.approx(1 - 1 / 10)
    with pytest.raises(ValueError):
        pass_at_k(2, 3, 1)


def test_majority_vote_clusters_equivalent_answers():
    assert majority_vote(['1/2', '3', r'\frac{2}{4}', '0.5', '3']) == 0
    assert majority_vote(['7', None, '8', '8']) == 2
    assert majority_vote([None, '']) is None
    assert majority_vote(['a', 'b']) == 0  # tie -> earliest class


def math_row(answer='2', ident='m1'):
    return {'id': ident, 'source': 's', 'cutoff_status': 'post_cutoff', 'prompt': 'q',
            'task_spec': json.dumps({'kind': 'math', 'expected': answer})}


def fake(outputs):
    calls = []
    def generate(messages, count, sample):
        calls.append((len(messages), count, sample))
        return [(outputs.pop(0), 5) for _ in range(count)]
    return generate, calls


def test_math_rows_score_pass_and_majority():
    generate, _ = fake([r'\boxed{3}', r'\boxed{2}', r'\boxed{3}', r'\boxed{4/2}'])
    record = evaluate_row(math_row(), generate, 4)
    metrics = row_metrics(record, [1, 2, 4])
    assert metrics['pass@1'] == pytest.approx(0.5)
    assert metrics['pass@4'] == 1.0
    assert metrics['maj@1'] == 0.0 and metrics['maj@4'] == 0.0  # 3 vs {2, 4/2}: tie, earliest wins
    assert metrics['tokens'] == 20


def test_coding_repair_only_after_all_samples_fail():
    spec = {'kind': 'coding', 'verifier': 'restricted_python', 'function': 'solve',
            'cases': [{'args': [2], 'expected': 4}]}
    row = {'id': 'c1', 'source': 'c', 'cutoff_status': 'post_cutoff', 'prompt': 'double it', 'task_spec': spec}
    generate, calls = fake(['def solve(x):\n return x', 'def solve(x):\n return x+1',
                            '```python\ndef solve(x):\n    return 2*x\n```'])
    record = evaluate_row(row, generate, 2)
    assert record['repair']['correct'] and calls[-1] == (3, 1, False)
    assert row_metrics(record, [1, 2])['any_or_repair'] == 1.0
    generate, calls = fake(['def solve(x):\n return 2*x', 'def solve(x):\n return x'])
    assert 'repair' not in evaluate_row(row, generate, 2) and len(calls) == 1


def test_run_resumes_and_summarizes(tmp_path):
    rows = [math_row('2', 'a'), math_row('5', 'b')]
    generate, calls = fake([r'\boxed{2}', r'\boxed{2}', r'\boxed{1}', r'\boxed{5}'])
    summary = run(rows, generate, 2, [1, 2], tmp_path)
    assert summary['overall']['rows'] == 2 and summary['overall']['pass@2'] == 1.0
    assert summary['overall']['maj@1'] == 0.5 and summary['source=s']['rows'] == 2
    generate, calls = fake([])
    assert run(rows, generate, 2, [1, 2], tmp_path)['overall']['rows'] == 2 and not calls


def test_wilson_and_summary_intervals():
    low, high = wilson(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8
    records = [{'kind': 'math', 'source': 's', 'samples': [{'correct': c, 'answer': None, 'tokens': 1}]}
               for c in (True, False)]
    assert summarize(records, [1])['overall']['pass@1_ci95'] == wilson(1, 2)


def test_control_rewards_ignore_correctness():
    texts = [r'\boxed{2}', r'\boxed{999}', 'no box', '```python\ndef f(:\n```']
    specs = [json.dumps({'kind': 'math', 'expected': '2'})] * 3 + [json.dumps({'kind': 'coding', 'tests': 'x'})]
    assert format_only_reward(texts, task_spec=specs) == [1.0, 1.0, 0.0, 0.0]
    first = random_control_reward(texts)
    assert first == random_control_reward(texts) and set(first) <= {0.0, 1.0}
    many = random_control_reward([f'answer {i}' for i in range(400)])
    assert 0.4 < sum(many) / 400 < 0.6


def test_benchmark_heldout_rejects_training_overlap(tmp_path, monkeypatch):
    import benchmark_gos
    path = tmp_path / 'heldout.jsonl'
    path.write_text(json.dumps(math_row() | {'prompt': 'Compute the area of the unit square exactly please.'}) + '\n')
    config = {'heldout_tasks': [str(path)], 'seed': 1, 'heldout_samples': 1}
    monkeypatch.setattr(best_of_n, 'hf_generator', lambda *a, **k: fake([r'\boxed{2}'])[0])
    monkeypatch.setattr(benchmark_gos, 'seed_all', lambda seed: None)
    class Model:
        def eval(self):
            return self
    with pytest.raises(RuntimeError):
        benchmark_gos.heldout_evaluation(Model(), None, config,
                                         [{'prompt': 'Compute the area of the unit square exactly please.'}], tmp_path)
    summary = benchmark_gos.heldout_evaluation(Model(), None, config, [{'prompt': 'unrelated text here'}], tmp_path)
    assert summary['overall']['pass@1'] == 1.0
