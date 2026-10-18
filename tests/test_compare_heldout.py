import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from compare_heldout import compare, load, scaling, sign_test  # noqa: E402


def write(run, variant, outcomes):
    path = run / variant
    path.mkdir(parents=True)
    path.joinpath('heldout_records.jsonl').write_text(''.join(json.dumps(
        {'id': f'r{i}', 'cutoff_status': 'post_cutoff' if i % 2 else 'pre_cutoff', 'source': 's',
         'samples': [{'correct': c} for c in row]}) + '\n' for i, row in enumerate(outcomes)))


def test_paired_comparison(tmp_path):
    write(tmp_path, 'frozen', [[False, False]] * 30 + [[True, True]] * 10)
    write(tmp_path, 'better', [[True, True]] * 20 + [[False, False]] * 10 + [[True, True]] * 10)
    write(tmp_path, 'same', [[False, False]] * 30 + [[True, True]] * 10)
    write(tmp_path / 'recurrent', 'passes-4', [[True, False]] * 40)
    table = compare(load(tmp_path), 'frozen', resamples=2000)
    assert abs(table['recurrent/passes-4']['all']['diff'] - 0.25) < 1e-9
    better, same = table['better']['all'], table['same']['all']
    assert better['wins'] == 20 and better['losses'] == 0 and abs(better['diff'] - 0.5) < 1e-9
    assert better['ci95'][0] > 0.3 and better['sign_test_p'] < 1e-5
    assert same['diff'] == 0 and same['sign_test_p'] == 1.0 and same['ci95'] == [0.0, 0.0]
    assert set(table['better']) == {'all', 'cutoff=post_cutoff', 'cutoff=pre_cutoff'}
    assert abs(sign_test(3, 1) - 0.625) < 1e-12


def test_pass_scaling_within_each_variant(tmp_path):
    write(tmp_path / 'deep', 'passes-1', [[False, False]] * 20)
    write(tmp_path / 'deep', 'passes-4', [[True, False]] * 20)
    write(tmp_path / 'deep', 'passes-12', [[True, True]] * 20)
    write(tmp_path / 'deep', 'passes-12-step0.25', [[True, False]] * 20)
    write(tmp_path / 'shallow', 'passes-1', [[True, True]] * 20)
    write(tmp_path, 'frozen', [[False, False]] * 20)
    table = scaling(load(tmp_path), resamples=500)
    assert set(table) == {'deep'} and set(table['deep']) == {'passes-4', 'passes-12', 'passes-12-step0.25'}
    assert abs(table['deep']['passes-4']['all']['diff'] - 0.5) < 1e-9
    assert abs(table['deep']['passes-12']['all']['diff'] - 1.0) < 1e-9
