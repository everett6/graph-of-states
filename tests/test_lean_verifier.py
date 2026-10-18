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
