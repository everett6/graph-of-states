import subprocess

import pytest

import lean_verifier
from lean_verifier import build_file, extract_proof, statement_body

STATEMENT = 'theorem toy (a b : ℕ) (h₀ : a = 2) (h₁ : a + b = 5) :\n  b = 3 := sorry'


def lean_available():
    try:
        lean_verifier.lean_environment()
    except Exception:
        return False
    probe = subprocess.run(['bwrap', '--unshare-all', '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib',
                            '--symlink', 'usr/lib64', '/lib64', '--symlink', 'usr/bin', '/bin', '/usr/bin/true'],
                           capture_output=True)
    return probe.returncode == 0


def test_statement_and_proof_extraction():
    assert statement_body(STATEMENT) == ('toy', '(a b : ℕ) (h₀ : a = 2) (h₁ : a + b = 5) :\n  b = 3', '')
    name, body, defs = statement_body('def f (n : ℕ) : ℕ := n + 1\n\ntheorem g : f 1 = 2 := by\n')
    assert (name, body, defs) == ('g', ': f 1 = 2', 'def f (n : ℕ) : ℕ := n + 1')
    with pytest.raises(ValueError):
        statement_body('def x := 1')
    full = 'Plan.\n```lean4\ntheorem toy (a b : ℕ) : b = 3 := by\n  subst h₀\n  omega\n```'
    assert extract_proof(full) == 'subst h₀\nomega'
    assert extract_proof('```lean\n  omega\n```') == 'omega'
    source = build_file(STATEMENT, 'omega')
    assert source.index('theorem __gos_target') < source.index('omega') < source.index('#print axioms')


@pytest.mark.skipif(not lean_available(), reason='Lean/Mathlib or bubblewrap unavailable')
def test_lean_checks_real_proofs_and_rejects_cheats():
    from verified_rewards import verify
    spec = {'kind': 'lean', 'statement': STATEMENT}
    assert verify('```lean\nsubst h₀\nomega\n```', spec).correct
    wrong = verify('```lean\nnorm_num\n```', spec)
    assert wrong.verified and not wrong.correct and 'proof line' in wrong.reason
    for cheat in ('sorry', 'native_decide', 'exact sorryAx _ false', 'omega\n#exit'):
        assert not verify(f'```lean\n{cheat}\n```', spec).correct


@pytest.mark.skipif(not lean_available(), reason='Lean/Mathlib or bubblewrap unavailable')
def test_batched_checks_match_single_checks():
    from lean_verifier import check_proof, check_proofs
    other = 'theorem two : (2 : ℕ) + 2 = 4 := sorry'
    statements = [STATEMENT, STATEMENT, other, STATEMENT, STATEMENT, other, STATEMENT]
    proofs = ['subst h₀\nomega', 'norm_num', 'norm_num', 'sorry', 'omega\n/-', 'rfl', 'exact absurd h₁ (by omega)']
    batched = check_proofs(statements, proofs)
    single = [check_proof(s, p) for s, p in zip(statements, proofs)]
    assert [r['correct'] for r in batched] == [r['correct'] for r in single] == [
        True, False, True, False, False, True, False]
    assert batched[1]['reason'] == 'lean_error' and 'proof line' in lean_verifier.feedback(batched[1])
    from verified_rewards import grpo_verified_reward
    spec = {'kind': 'lean', 'statement': STATEMENT}
    assert grpo_verified_reward(['```lean\nomega\n```', 'x', '```lean\nsubst h₀\nomega\n```'],
                                task_spec=[spec, spec, spec]) == [1.0, 0.0, 1.0]


def test_workbook_proof_split():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
    from build_lean_train import split_proof
    full = ('import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real\n\n'
            '/- Show $b = 3$. -/\ntheorem wb_1 (a b : ℕ) (h₀ : a = 2)\n  (h₁ : a + b = 5) : b = 3 := by\n'
            '  /- subtract -/\n  subst h₀\n  omega\n')
    statement, opens, proof, informal = split_proof(full)
    assert statement == 'theorem wb_1 (a b : ℕ) (h₀ : a = 2)\n  (h₁ : a + b = 5) : b = 3 := sorry'
    assert (opens, proof, informal) == ('open BigOperators Real', '/- subtract -/\nsubst h₀\nomega', 'Show $b = 3$.')


def test_format_control_for_lean_ignores_correctness():
    from verified_rewards import format_only_reward
    spec = {'kind': 'lean', 'statement': STATEMENT}
    assert format_only_reward(['```lean\nnorm_num\n```', '```lean\nsorry\n```', 'no block'],
                              task_spec=[spec] * 3) == [1.0, 0.0, 0.0]
