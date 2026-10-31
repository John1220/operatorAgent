"""The public CLI freezes explicit seeds without creating generation credit."""
import json

import pytest

from benchmarks.experiments.misc import operator_quality as cli
from operator_agent.decide import quality_evaluation as q
from operator_agent.core.run_contract import code_hash


CASE = 'layernorm-tail'
OTHER = 'matmul-strided-fp32'
SCHEMA = 'operator-seeded-paired-quality-plan-v1'
BAD = 'def kernel(X):\n    return X * 0\n'
GOOD = ('import numpy as np\ndef kernel(X):\n'
        '    z = X.astype(np.float32)\n'
        '    z = z - z.mean(axis=-1, keepdims=True)\n'
        '    return (z / np.sqrt((z*z).mean(axis=-1, keepdims=True)+1e-5)).astype(X.dtype)\n')


def source_file(tmp_path, source=BAD, name='initial.py'):
    path = tmp_path / name
    path.write_text(source, encoding='utf-8')
    return path


def plan_command(tmp_path, capsys, *, source=BAD, extra=()):
    path = source_file(tmp_path, source)
    assert cli.main(['--paired', '--initial-source', f'{CASE}={path}', *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_seeded_plan_is_pure_freezes_source_and_has_one_repair_per_arm(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('seeded planning must not construct a model, device, or concrete inputs')
    monkeypatch.setattr(cli, 'run_paired_plan', forbidden)
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    monkeypatch.setattr(q.quality_suite, 'evaluation_inputs_for', forbidden)
    path = source_file(tmp_path, BAD + '# UTF-8 跨距种子\n')
    output = tmp_path / 'plan.json'
    assert cli.main(['--paired', '--initial-source', f'{CASE}={path}',
                     '--output', str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan == json.loads(output.read_text())
    assert plan['schema'] == SCHEMA
    assert plan['initial_sources'] == {
        CASE: {'source': path.read_text(encoding='utf-8'),
               'sha256': code_hash(path.read_text(encoding='utf-8'))}}
    assert [row['case_id'] for row in plan['base_plan']['manifest']['cases']] == [CASE]
    assert plan['base_plan']['controls']['budget']['max_calls'] == 1
    assert plan['base_plan']['controls']['budget']['max_verifications'] == 2
    assert plan['pairing']['generation_calls'] == 0
    assert plan['pairing']['repair_calls_per_arm'] == 1
    with pytest.raises(FileExistsError):
        cli.main(['--paired', '--initial-source', f'{CASE}={path}', '--output', str(output)])


def test_repeated_seed_options_infer_only_the_declared_cases(tmp_path, capsys):
    first = source_file(tmp_path, name='layernorm.py')
    second = source_file(tmp_path, 'def kernel(A, B):\n    return A @ B\n', 'matmul.py')
    assert cli.main(['--paired', '--initial-source', f'{OTHER}={second}',
                     '--initial-source', f'{CASE}={first}']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert list(plan['initial_sources']) == [OTHER, CASE]
    assert [row['case_id'] for row in plan['base_plan']['manifest']['cases']] == [OTHER, CASE]


def test_explicit_case_selection_freezes_the_same_seed(tmp_path, capsys):
    plan = plan_command(tmp_path, capsys, extra=['--case', CASE])
    assert plan['schema'] == SCHEMA
    assert plan['initial_sources'][CASE]['source'] == BAD


def test_seed_filename_may_contain_equals(tmp_path, capsys):
    path = source_file(tmp_path, name='seed=one.py')
    assert cli.main(['--paired', '--initial-source', f'{CASE}={path}']) == 0
    assert json.loads(capsys.readouterr().out)['initial_sources'][CASE]['source'] == BAD


def test_seeded_cli_preserves_crlf_and_utf8_source_bytes(tmp_path, capsys):
    source = '# 原始跨距种子\r\ndef kernel(X):\r\n    return X * 0\r\n'
    path = tmp_path / 'crlf.py'
    path.write_bytes(source.encode('utf-8'))
    assert cli.main(['--paired', '--initial-source', f'{CASE}={path}']) == 0
    seed = json.loads(capsys.readouterr().out)['initial_sources'][CASE]
    assert seed['source'].encode('utf-8') == path.read_bytes()
    assert seed['sha256'] == code_hash(source)


@pytest.mark.parametrize('action', [
    [], ['--replay', 'missing-replies.json'],
    ['--execute', 'missing-plan.json'], ['--preflight', 'missing-plan.json'],
    ['--compare', 'missing-left.json', 'missing-right.json'],
])
def test_initial_source_requires_paired_before_work(tmp_path, capsys, action):
    path = source_file(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main([*action, '--initial-source', f'{CASE}={path}',
                  '--output', str(tmp_path / 'not-created')])
    assert error.value.code == 2
    assert capsys.readouterr().err
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('action', [
    ['--execute', 'missing-plan.json'], ['--preflight', 'missing-plan.json'],
    ['--compare', 'missing-left.json', 'missing-right.json'],
])
def test_initial_source_cannot_replace_a_frozen_action_plan(tmp_path, capsys, action):
    path = source_file(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(['--paired', *action, '--initial-source', f'{CASE}={path}',
                  '--output', str(tmp_path / 'not-created')])
    assert error.value.code == 2
    assert capsys.readouterr().err
    assert not (tmp_path / 'not-created').exists()


@pytest.mark.parametrize('kind', [
    'duplicate', 'unknown_case', 'missing_case', 'extra_case', 'malformed',
    'empty_case', 'empty_path', 'missing_file', 'directory', 'invalid_utf8',
    'oversize_utf8', 'empty_source',
])
def test_invalid_seed_declarations_fail_as_cli_usage_without_work(tmp_path, capsys, kind):
    path = source_file(tmp_path)
    entries = [f'{CASE}={path}']
    extra = []
    if kind == 'duplicate':
        entries.append(f'{CASE}={source_file(tmp_path, GOOD, "second.py")}')
    elif kind == 'unknown_case':
        entries = [f'unknown-quality-case={path}']
    elif kind == 'missing_case':
        extra = ['--case', CASE, '--case', OTHER]
    elif kind == 'extra_case':
        entries.append(f'{OTHER}={path}')
        extra = ['--case', CASE]
    elif kind == 'malformed':
        entries = [str(path)]
    elif kind == 'empty_case':
        entries = [f'={path}']
    elif kind == 'empty_path':
        entries = [f'{CASE}=']
    elif kind == 'missing_file':
        entries = [f'{CASE}={tmp_path / "missing.py"}']
    elif kind == 'directory':
        entries = [f'{CASE}={tmp_path}']
    elif kind == 'invalid_utf8':
        path.write_bytes(b'\xff\xfe')
    elif kind == 'oversize_utf8':
        # Character length is below the limit; the UTF-8 payload exceeds it.
        path.write_text('#' + '\u03bb' * 16384, encoding='utf-8')
    elif kind == 'empty_source':
        path.write_text(' \n\t', encoding='utf-8')
    argv = ['--paired', *extra, '--output', str(tmp_path / 'not-created.json')]
    for entry in entries:
        argv.extend(['--initial-source', entry])
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2
    assert capsys.readouterr().err
    assert not (tmp_path / 'not-created.json').exists()


def test_exact_utf8_source_byte_limit_is_accepted(tmp_path, capsys):
    prefix = BAD + '#'
    source = prefix + 'x' * (32768 - len(prefix.encode('utf-8')))
    assert len(source.encode('utf-8')) == 32768
    plan = plan_command(tmp_path, capsys, source=source)
    assert plan['initial_sources'][CASE]['source'] == source


@pytest.mark.parametrize('paired_flag', [[], ['--paired']])
def test_execute_dispatches_seeded_schema_without_reading_seed_files_again(
        tmp_path, capsys, monkeypatch, paired_flag):
    plan = plan_command(tmp_path, capsys)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    (tmp_path / 'initial.py').unlink()
    seen = []
    def run(declaration, output, **kwargs):
        seen.append((declaration, output, kwargs))
        return {'seeded_execution_dispatched': True}
    def forbidden(*args, **kwargs):
        pytest.fail('seeded schema must not dispatch to the ordinary runner')
    monkeypatch.setattr(cli, 'run_paired_plan', run)
    monkeypatch.setattr(cli, 'run_plan', forbidden)
    output = tmp_path / 'run'
    assert cli.main([*paired_flag, '--execute', str(path), '--host', 'synthetic-host',
                     '--output', str(output)]) == 0
    assert seen == [(plan, str(output), {'host': 'synthetic-host'})]
    assert json.loads(capsys.readouterr().out)['seeded_execution_dispatched'] is True


def test_preflight_validates_and_unwraps_seeded_schema(tmp_path, capsys, monkeypatch):
    plan = plan_command(tmp_path, capsys)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    seen = []
    def preflight(declaration, output, **kwargs):
        seen.append((declaration, kwargs))
        return {'model_calls': 0}
    monkeypatch.setattr(cli, 'run_preflight', preflight)
    assert cli.main(['--preflight', str(path), '--host', 'synthetic-host',
                     '--output', str(tmp_path / 'preflight')]) == 0
    assert seen == [(plan['base_plan'], {'host': 'synthetic-host'})]
    assert json.loads(capsys.readouterr().out)['model_calls'] == 0


def test_seeded_replay_repairs_both_arms_without_any_shared_generation(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('explicit seed replay must never generate a shared candidate')
    monkeypatch.setattr(q, '_generate', forbidden)
    monkeypatch.setattr(q, 'LLMClient', forbidden)
    path = source_file(tmp_path)
    replies = tmp_path / 'replies.json'
    replies.write_text(json.dumps({CASE: {
        'baseline': ['```python\n' + GOOD + '\n```'],
        'diagnosis': ['```python\n' + GOOD + '\n```'],
    }}))
    output = tmp_path / 'run'
    assert cli.main(['--paired', '--initial-source', f'{CASE}={path}',
                     '--replay', str(replies), '--output', str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['plan']['schema'] == SCHEMA
    assert report['shared']['costs']['model_calls']['known_total'] == 0
    assert report['total_actual_costs']['model_calls']['known_total'] == 2
    assert report['pairs'][0]['initial_source_hash'] == code_hash(BAD)
    assert report['pairs'][0]['source_hashes_match'] is True
    for arm in ('baseline', 'diagnosis'):
        assert report['arms'][arm]['metrics']['first_full_l2']['count'] == 0
        assert report['arms'][arm]['metrics']['final_full_l2']['count'] == 1
        assert report['arms'][arm]['costs']['model_calls']['known_total'] == 1
        assert (output / 'arms' / arm / CASE / 'candidate-1.py').read_text() == BAD

