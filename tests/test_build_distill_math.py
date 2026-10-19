import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from build_distill_math import accepted, candidates, to_gemma  # noqa: E402


def test_to_gemma_moves_reasoning_into_thought_channel():
    assert to_gemma('<think>\nlet x=2\n</think>\n\nSo \\boxed{2}') == '<|channel>thought\nlet x=2<channel|>So \\boxed{2}'
    assert to_gemma('no tags \\boxed{2}') is None
    assert to_gemma('<think>\n\n</think> \\boxed{2}') is None
    assert to_gemma('<think>a</think>   ') is None


def test_candidates_keep_verified_complete_traces_shortest_first():
    rows = [{'question_type': 'math-word-problem', 'answer': '2',
             'generations': ['<think>long reasoning here</think>\\boxed{2}', '<think>short</think>\\boxed{2}',
                             '<think>w</think>\\boxed{3}', '<think>cut'],
             'correctness_math_verify': [True, True, False, True], 'is_reasoning_complete': [True, True, True, False]},
            {'question_type': 'proof', 'answer': 'x', 'generations': ['<think>p</think>done'],
             'correctness_math_verify': [True], 'is_reasoning_complete': [True]},
            {'question_type': 'math-word-problem', 'answer': '1', 'generations': ['<think>a</think>\\boxed{1}'],
             'correctness_math_verify': None, 'is_reasoning_complete': None}]
    found = list(candidates(rows))
    assert len(found) == 1
    assert found[0][1] == ['<|channel>thought\nshort<channel|>\\boxed{2}',
                           '<|channel>thought\nlong reasoning here<channel|>\\boxed{2}']


def test_accepted_drops_unparseable_references():
    spec = {'kind': 'math', 'expected': '2'}
    assert accepted({'answer': '<|channel>thought\nx<channel|>\\boxed{2}', 'task_spec': spec})
    assert not accepted({'answer': '\\boxed{3}', 'task_spec': spec})
    assert not accepted({'answer': '\\boxed{4}', 'task_spec': {'kind': 'math', 'expected': 'v_{R}=4\\mathrm{~}/\\mathrm{}'}})
