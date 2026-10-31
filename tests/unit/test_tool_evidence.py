"""S1: independent source/result/report evidence, without model or GPU calls."""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from operator_agent.app import hitl_tools as ht
from operator_agent.core.schemas import VerificationLevel as Level, VerificationResult


SOURCE = 'def kernel(A, B):\r\n    # 原始源码\r\n    return A @ B\r\n'


def prepared(tmp_path):
    run = tmp_path / 'run'
    ht.prepare(dict(request='matmul fp32', shapes=[[8, 8, 8]]), run)
    return run


def outcomes(*, numeric=True, validity='valid'):
    return [
        VerificationResult(Level.L1_COMPILE, True, 'compiled'),
        VerificationResult(Level.L2_FUNCTION, numeric, 'unabridged ' * 120,
            metrics={'correctness': {'verdict': 'passed' if numeric else 'failed',
                'cases': [dict(case_id=f'case-{i}', verdict='passed',
                    metrics={'full_input_receipt': i}) for i in range(13)]},
                'input_fidelity_verified': True, 'reference_receipt': {'identity': 'native-reference'}}),
        VerificationResult(Level.L3_PERFORMANCE, False, 'timed',
            metrics={'measurement_validity': validity,
                'environment_fingerprint': {'device_uuid': 'synthetic-gpu'},
                'arm_windows': {'candidate': 'window-1'}, 'raw_samples': [1., 2., 3.]},
            error_log='complete backend log'),
    ]


def fake_verifier(monkeypatch, results):
    def verify(code, *args, **kwargs):
        print('Verifier observed source ' + hashlib.sha256(code.encode()).hexdigest())
        return results
    monkeypatch.setattr(ht, 'Verifier', lambda backend: SimpleNamespace(verify=verify))


def test_two_different_sources_keep_independent_reports_and_full_results(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    results = outcomes()
    fake_verifier(monkeypatch, results)
    first = ht.verify(run, SOURCE)
    files = {p: p.read_bytes() for p in (run / 'verify-1').iterdir() if p.is_file()}
    second = ht.verify(run, SOURCE.replace('A @ B', '(A @ B) * .9'))
    assert first['report'] != second['report']
    assert all(path.read_bytes() == data for path, data in files.items())
    source = Path(first['source'])
    assert source.read_bytes() == SOURCE.encode('utf-8')
    assert first['source_hash'] == hashlib.sha256(source.read_bytes()).hexdigest()
    full = json.loads(Path(first['evidence']).read_text())
    assert full['binding']['source_hash'] == first['source_hash']
    assert full['binding']['contract_hash'] == first['contract_hash']
    assert full['contract']['hash'] == first['contract_hash']
    assert full['verdicts'][1]['message'] == results[1].message
    assert len(full['verdicts'][1]['metrics']['correctness']['cases']) == 13
    assert full['verdicts'][1]['metrics']['correctness']['cases'][12]['metrics'] == {'full_input_receipt': 12}
    assert full['verdicts'][2]['error_log'] == 'complete backend log'
    assert full['verdicts'][2]['metrics']['raw_samples'] == [1., 2., 3.]
    assert len(first['verdicts'][1]['message']) == 600
    assert len(first['verdicts'][1]['metrics']['correctness']['cases']) == 10
    assert len(json.dumps(first, ensure_ascii=False)) < 12000
    assert (run / 'verify-1.json').is_file() and (run / 'verify-1.log').is_file()


def test_legacy_report_and_noncontiguous_round_numbers_are_never_overwritten(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    (run / 'report.md').write_text('old report referenced by a legacy receipt')
    (run / 'verify-1.json').write_text('{"report":"legacy"}')
    (run / 'verify-3.json').write_text('{"legacy":3}')
    before = {p: p.read_bytes() for p in run.iterdir() if p.is_file()}
    fake_verifier(monkeypatch, [VerificationResult(Level.L1_COMPILE, True)])
    result = ht.verify(run, SOURCE, stop_after='L1')
    assert result['round'] == 4
    assert all(path.read_bytes() == data for path, data in before.items())
    assert Path(result['report']).parent == run / 'verify-4'


@pytest.mark.parametrize('results,stop', [
    ([VerificationResult(Level.L1_COMPILE, False, 'compile error')], None),
    ([VerificationResult(Level.L1_COMPILE, True)], 'L1'),
    (outcomes(numeric=False), None),
    (outcomes(validity='uninformative'), None),
])
def test_recording_does_not_add_passed_stages_or_improve_validity(tmp_path, monkeypatch, results, stop):
    run = prepared(tmp_path)
    fake_verifier(monkeypatch, results)
    payload = ht.verify(run, SOURCE, stop_after=stop)
    full = json.loads(Path(payload['evidence']).read_text())
    expected = {r.level.value: r.passed for r in results}
    assert payload['levels'] == full['levels'] == expected
    assert full['binding']['stop_after'] == stop
    assert [r['passed'] for r in full['verdicts']] == [r.passed for r in results]
    assert payload['validity'] == full['validity']


def test_result_write_failure_keeps_previous_round_and_next_call_skips_partial_round(tmp_path, monkeypatch):
    from operator_agent.decide.tool_evidence import RoundEvidence
    run = prepared(tmp_path)
    fake_verifier(monkeypatch, [VerificationResult(Level.L1_COMPILE, True)])
    first = ht.verify(run, SOURCE, stop_after='L1')
    old = Path(first['evidence']).read_bytes()
    original = RoundEvidence._write_once
    def fail_one_file(path, data):
        if path == run / 'verify-2/result.json':
            raise OSError('synthetic evidence disk error')
        return original(path, data)
    with monkeypatch.context() as patch:
        patch.setattr(RoundEvidence, '_write_once', staticmethod(fail_one_file))
        with pytest.raises(OSError, match='synthetic evidence disk error'):
            ht.verify(run, SOURCE, stop_after='L1')
    assert Path(first['evidence']).read_bytes() == old
    assert not (run / 'verify-2/complete.json').exists()
    assert ht.verify(run, SOURCE, stop_after='L1')['round'] == 3


def test_backend_error_retains_source_and_never_completes_evidence(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    error = RuntimeError('synthetic backend unavailable')
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(ht, '_make_backend', fail)
    with pytest.raises(RuntimeError) as caught:
        ht.verify(run, SOURCE)
    assert caught.value is error
    assert (run / 'verify-1/candidate.py').read_bytes() == SOURCE.encode()
    assert not (run / 'verify-1/complete.json').exists()


def test_bench_retains_all_candidate_sources_and_full_per_arm_measurements(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    results = outcomes()
    monkeypatch.setattr(ht, 'Verifier', lambda backend: SimpleNamespace(verify_l3_only=lambda *a: results[-1:]))
    payload = ht.bench(run, [{'name': '../not-a-path', 'code': SOURCE}, {'name': 'other', 'code': SOURCE+'# v2\n'}])
    full = json.loads(Path(payload['evidence']).read_text())
    assert payload['grade'] == full['grade'] == 'exploration'
    assert 'passed' not in payload and 'eligibility' not in payload
    assert full['arm_evidence'][0]['verdicts'][0]['metrics']['raw_samples'] == [1., 2., 3.]
    assert Path(full['arm_evidence'][0]['source']).read_bytes() == SOURCE.encode()
    assert Path(full['arm_evidence'][1]['source']).read_bytes() == (SOURCE+'# v2\n').encode()
    assert not (tmp_path / 'not-a-path').exists()
    assert not list(run.glob('verify-*.json'))


@pytest.mark.parametrize('surface', ['cli', 'mcp'])
def test_source_file_crlf_and_unicode_survive_public_entry(tmp_path, monkeypatch, capsys, surface):
    from operator_agent.app import cli, mcp_server
    run = prepared(tmp_path)
    source = tmp_path / 'input.py'
    source.write_bytes(SOURCE.encode())
    fake_verifier(monkeypatch, [VerificationResult(Level.L1_COMPILE, True)])
    if surface == 'cli':
        assert cli.main(['verify', '--run-dir', str(run), '--code-file', str(source),
                         '--stop-after', 'L1']) == 0
        payload = json.loads(capsys.readouterr().out)
    else:
        reply = mcp_server._call_tool('operator_verify', dict(run_dir=str(run),
            code_file=str(source), stop_after='L1'))
        payload = json.loads(reply['content'][0]['text'])
    assert Path(payload['source']).read_bytes() == SOURCE.encode()


def test_parallel_reservations_are_unique_and_legacy_gap_is_preserved(tmp_path):
    from operator_agent.core.contract import freeze_from_task
    from operator_agent.decide.tool_evidence import RoundEvidence
    run = prepared(tmp_path)
    declaration = json.loads((run/'declaration.json').read_text())
    contract = freeze_from_task(ht.build_task(declaration))
    (run/'verify-7.log').write_text('old log without JSON')
    def reserve(_):
        return RoundEvidence(run, 'verify', declaration, contract, tool_version=ht.TOOL_VERSION)
    with ThreadPoolExecutor(max_workers=6) as pool:
        reservations = list(pool.map(reserve, range(12)))
    assert sorted(record.index for record in reservations) == list(range(8, 20))
    assert all(not (record.path/'complete.json').exists() for record in reservations)
    assert (run/'verify-7.log').read_text() == 'old log without JSON'


def test_complete_receipt_covers_full_bytes_and_nonfinite_values_stay_explicit(tmp_path, monkeypatch):
    results = outcomes()
    results[-1].metrics['raw_nonfinite'] = float('nan')
    fake_verifier(monkeypatch, results)
    payload = ht.verify(prepared(tmp_path), SOURCE)
    folder = Path(payload['evidence']).parent
    receipt = json.loads((folder/'complete.json').read_text())
    assert receipt['binding']['source_hash'] == payload['source_hash']
    assert 'result.json' in receipt['files_sha256']
    for name, expected in receipt['files_sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected
    full = json.loads(Path(payload['evidence']).read_text())
    assert full['verdicts'][-1]['metrics']['raw_nonfinite'] == {'$nonfinite': 'nan'}


def test_repair_keeps_initial_final_and_each_verified_source_with_complete_results(tmp_path, monkeypatch):
    from operator_agent.execute import llm_client
    from tests.unit.test_fixer_generation_records import FIXED, ORIGINAL, Replies, reply
    from tests.unit.test_repair_loop import FakeVerifier, bad
    results = [bad(), [VerificationResult(Level.L1_COMPILE, True),
        VerificationResult(Level.L2_FUNCTION, True, metrics={'correctness': {'verdict': 'passed'}})]]
    verifier = FakeVerifier(results)
    monkeypatch.setattr(ht, 'Verifier', lambda backend: verifier)
    monkeypatch.setattr(ht, '_llm_problem', lambda: None)
    monkeypatch.setattr(llm_client, 'LLMClient', lambda **kw: Replies([reply(FIXED)]))
    run = tmp_path/'layernorm'
    ht.prepare(dict(request='layernorm fp32', shapes=[[2, 3]]), run)
    payload = ht.repair(run, ORIGINAL, stop_after='L2')
    full = json.loads(Path(payload['evidence']).read_text())
    assert Path(payload['source']).read_text() == ORIGINAL
    assert (Path(payload['evidence']).parent/'final.py').read_text() == FIXED
    assert len(full['rounds']) == 2
    for index, code in enumerate((ORIGINAL, FIXED)):
        step = full['rounds'][index]['verification_evidence']
        assert Path(step['source']).read_text() == code
        saved = json.loads(Path(step['evidence']).read_text())
        assert saved['verdicts'][0]['message'] == results[index][0].message
    assert full['rounds'][0]['action'] == 'fixed'
    assert json.loads((run/'repair-1.json').read_text())['selected_source_hash'] == payload['selected_source_hash']


def test_conductor_history_retains_admitted_source_but_refusal_is_hash_only(tmp_path):
    from operator_agent.core.run_layout import run_layout
    from operator_agent.decide.tool_evidence import persist_conductor_round
    task = ht.build_task(dict(request='matmul fp32', shapes=[[8, 8, 8]]))
    layout = run_layout(str(tmp_path/'autonomous'))
    first = persist_conductor_round(layout, task, SOURCE, outcomes(), 1, admitted=True)
    files = {Path(first[key]): Path(first[key]).read_bytes() for key in ('source', 'evidence', 'report')}
    refused = 'REFUSED_EXECUTABLE_SOURCE_SHOULD_NOT_APPEAR'
    second = persist_conductor_round(layout, task, refused,
        [VerificationResult(Level.L1_COMPILE, False, 'action refused')], 1, admitted=False)
    assert second['source'] is None and second['source_retention'] == 'hash_only'
    assert first['evidence'] != second['evidence']
    assert all(path.read_bytes() == data for path, data in files.items())
    assert all(refused not in path.read_text() for path in (tmp_path/'autonomous').rglob('*') if path.is_file())


def test_independent_process_recovers_sources_after_both_producers_exit(tmp_path):
    run = tmp_path/'separate-process'
    subprocess.run([sys.executable, '-m', 'operator_agent.app.cli', 'prepare',
        '--request', 'matmul fp32', '--shapes', '8,8,8', '--run-dir', str(run)],
        check=True, capture_output=True, text=True)
    replies = []
    for index, source in enumerate((SOURCE, SOURCE+'# second\n')):
        path = tmp_path/f'input-{index}.py'
        path.write_bytes(source.encode())
        result = subprocess.run([sys.executable, '-m', 'operator_agent.app.cli', 'verify',
            '--run-dir', str(run), '--code-file', str(path), '--stop-after', 'L1'],
            check=True, capture_output=True, text=True)
        replies.append(json.loads(result.stdout))
    consumer = '''import sys,json,hashlib
from pathlib import Path
for name in sys.argv[1:]:
    value=json.loads(Path(name).read_text())
    assert hashlib.sha256(Path(value['source']).read_bytes()).hexdigest()==value['source_hash']
    assert Path(value['report']).is_file()
    assert set(value['levels'])=={'L1'}
print('recovered two independent rounds')
'''
    consumed = subprocess.run([sys.executable, '-c', consumer,
        *[row['evidence'] for row in replies]], check=True, capture_output=True, text=True)
    assert 'recovered two' in consumed.stdout
    assert Path(replies[0]['source']).read_bytes() == SOURCE.encode()
    assert Path(replies[1]['source']).read_bytes() == (SOURCE+'# second\n').encode()


def test_partial_bench_keeps_unmeasured_sources_without_inventing_results(tmp_path, monkeypatch):
    run = prepared(tmp_path)
    ticks = iter((0., 0., 121.))
    monkeypatch.setattr(ht.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(ht, 'Verifier', lambda backend: SimpleNamespace(
        verify_l3_only=lambda *args: outcomes()[-1:]))
    payload = ht.bench(run, [{'code': SOURCE}, {'code': SOURCE+'# unmeasured\n'}])
    full = json.loads(Path(payload['evidence']).read_text())
    assert payload['partial'] is True and payload['measured'] == 1 and payload['requested'] == 2
    assert len(full['candidate_sources']) == 2 and len(full['arm_evidence']) == 1
    assert Path(full['candidate_sources'][1]['source']).read_bytes() == (SOURCE+'# unmeasured\n').encode()
