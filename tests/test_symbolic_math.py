import json

import pytest

from symbolic_math import Unsupported, boxed, equivalent
from verified_rewards import grpo_verified_reward, math_reward, verify

EQUIVALENT = [(r'\boxed{\frac{1}{2}}', '0.5'), (r'\boxed{2/4}', r'\frac12'), (r'\boxed{2\sqrt{2}}', r'\sqrt{8}'),
              (r'\boxed{\dfrac{\sqrt3}{2}}', r'\frac{\sqrt{3}}{2}'), (r'\boxed{3+4i}', '4i+3'),
              (r'\boxed{\{2,1\}}', r'\{1,2\}'), (r'\boxed{x^2-1}', '(x-1)(x+1)'), (r'\boxed{10,\!000}', '10000'),
              (r'\boxed{2\pi}', r'\pi\cdot 2'), (r'\boxed{x=5}', '5'), (r'\boxed{\sqrt[3]{8}}', '2'),
              (r'\boxed{90^\circ}', '90'), ('Answer: 153', '153'), (r'\boxed{\left(1,\frac{1}{2}\right)}', '(1,0.5)'),
              (r'\boxed{3^{2025}}', '3^{2025}'), (r'\boxed{2^{25}\cdot 26!}', '26!*2^25'), ('Answer: 6', '3!')]
DIFFERENT = [(r'\boxed{(1,2)}', '(2,1)'), (r'\boxed{x^2+1}', '(x-1)(x+1)'), (r'\boxed{\sqrt{2}}', '1.41421356'),
             (r'\boxed{\{1,2\}}', r'\{1,2,3\}'), (r'\boxed{(0,1]}', '(0,1)'), ('2**100000', '1'),
             ('((2**64)**64)**64', '1'), ('2^{5000}', '1'), ('1001!', '1'), ('x^{100}', '1'), ('NaN', '1'), ('True', '1'), ('__import__("os").system("x")', '1')]


@pytest.mark.parametrize('completion,reference', EQUIVALENT)
def test_equivalent_answers_accepted(completion, reference):
    assert math_reward(completion, reference).correct


@pytest.mark.parametrize('completion,reference', DIFFERENT)
def test_different_or_unsafe_answers_rejected(completion, reference):
    assert not math_reward(completion, reference).correct


def test_tolerance_and_nested_boxed():
    assert math_reward(r'\boxed{\sqrt{2}}', '1.4142', tolerance='0.001').correct
    assert boxed(r'first \boxed{1} then \boxed{\frac{3}{4}} end') == r'\frac{3}{4}'
    assert boxed(r'\boxed{unclosed') is None


def test_unsupported_references_fail_loudly():
    with pytest.raises(ValueError):
        math_reward('1', r'\begin{pmatrix}1\end{pmatrix}')
    with pytest.raises(Unsupported):
        equivalent('x', r'\int x')


def test_grpo_reward_routes_spec_boxed_reference_and_exact():
    spec = json.dumps({'kind': 'math', 'expected': r'\frac{1}{3}'})
    padded = {'kind': 'math', 'expected': '2', 'tolerance': None, 'cases': None}  # Arrow-style padding
    rewards = grpo_verified_reward(
        [r'\boxed{2/6}', r'\boxed{2}', r'so \boxed{\frac{2}{4}}', 'same text', 'different'],
        answer=['ignored', 'ignored', r'<|begin_of_solution|>\boxed{0.5}<|end_of_solution|>', 'same text', 'x'],
        task_spec=[spec, padded, None, None, None])
    assert rewards == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert verify(r'\boxed{1/3}', spec).correct
    with pytest.raises(ValueError):
        grpo_verified_reward(['a'], answer=['a', 'b'])
