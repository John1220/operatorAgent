"""Arm metering must reconcile counts without inventing cost or sampling equality."""
import json
from pathlib import Path

import pytest

from operator_agent.decide.arm_metering import (MeteringError, REQUIRED_SAMPLING_FIELDS,
                                               compare, normalize_arm_document,
                                               normalize_host_ledger)

SAMPLING = dict(provider='aliyun', model='deepseek-v3', temperature=0.1, n=1,
                max_output_tokens=64, enable_thinking=True)


def ledger(records, *, limits=None):
    return dict(schema='operator-host-model-budget-v1', task_id='a' * 32,
                contract_hash='b' * 64,
                limits=dict(max_model_calls=30, max_output_tokens_per_call=64,
                            total_output_tokens_reservations=1920) if limits is None else limits,
                records=records)


def record(index, *, status='answered', usage='default', provider='aliyun', model='deepseek-v3',
           sampling=True):
    row = dict(id=index, request_id=f'r{index}', provider=provider, model=model,
               max_output_tokens_reserved=64, status=status, reserved_at=1000 + index,
               finished_at=None if status == 'reserved' else 2000 + index,
               usage=None if usage is None else (dict(inputTokens=10, outputTokens=20,
                                                      totalTokens=30) if usage == 'default' else usage))
    if sampling:
        # 当前插件**还不写**这几个键; 这里按「将来会写」的形状造 fixture, 另有用例
        # 钉住「没写就必须 block」。
        row.update(temperature=0.1, n=1, enable_thinking=True)
    return row


def arm_document(rows, *, budget=None):
    default_usage = dict(inputTokens=10, outputTokens=20, totalTokens=30)
    merged = []
    for row in rows:
        base = dict(SAMPLING, usage=default_usage)
        base.update(row)
        merged.append(base)
    return dict(arm='A', source='provider proxy', metering='provider_proxy',
                budget=dict(max_calls=30, max_output_tokens_per_call=64, wall_clock_s=2400)
                if budget is None else budget, dispatches=merged)


def policy():
    return dict(schema='operator-host-task-budget-v1',
                limits=dict(wall_clock_s=2400, max_model_calls=30, max_output_tokens_per_call=64,
                            total_output_tokens_reservations=1920))


def test_two_arms_metered_the_same_way_establish_comparability():
    receipt = compare(normalize_arm_document(arm_document([dict(status='answered')])),
                      normalize_host_ledger(ledger([record(1)]), policy()))
    assert receipt['metering_comparable'] is True
    assert receipt['sampling_comparability'] == 'established'
    assert receipt['blockers'] == []
    assert receipt['cost'] == 'unknown'          # token 计数不是发票
    assert receipt['arms']['B']['output_tokens'] == 20
    assert receipt['arms']['B']['reserved_output_tokens'] == 64


def test_a_ledger_that_never_recorded_sampling_blocks_the_verdict_by_name():
    state = ledger([record(1, sampling=False)])
    for row in state['records']:
        row.pop('provider'), row.pop('model')
    receipt = compare(normalize_arm_document(arm_document([dict(status='answered')])),
                      normalize_host_ledger(state, policy()))
    assert receipt['sampling_comparability'] == 'not_established'
    assert any('sampling.recorded' in blocker for blocker in receipt['blockers'])
    assert any('temperature' in row for row in receipt['unrecorded_sampling'])
    assert any('enable_thinking' in row for row in receipt['unrecorded_sampling'])


def test_unequal_budgets_are_named_rather_than_averaged_away():
    receipt = compare(
        normalize_arm_document(arm_document([dict(status='answered')], budget={'max_calls': None})),
        normalize_host_ledger(ledger([record(1)]), policy()))
    assert receipt['sampling_comparability'] == 'not_established'
    detail = [row for row in receipt['checks'] if row['check'] == 'budget.max_model_calls'][0]
    assert detail['ok'] is False and 'A=None' in detail['detail'] and 'B=30' in detail['detail']


def test_unknown_usage_is_not_a_partial_sum():
    receipt = compare(normalize_arm_document(arm_document([dict(status='answered'),
                                                          dict(status='answered', usage=None)])),
                      normalize_host_ledger(ledger([record(1), record(2)]), policy()))
    arm = receipt['arms']['A']
    assert arm['usage_unknown'] == 1
    assert arm['output_tokens'] is None and arm['total_tokens'] is None
    assert receipt['metering_comparable'] is False
    assert any('usage_complete' in blocker for blocker in receipt['blockers'])


def test_open_reservation_and_cap_violation_are_both_blockers():
    state = ledger([record(1), record(2, status='reserved', usage=None)])
    state['records'][0]['usage'] = dict(inputTokens=1, outputTokens=99, totalTokens=100)
    receipt = compare(normalize_arm_document(arm_document([dict(status='answered')])),
                      normalize_host_ledger(state, policy()))
    names = {row['check']: row['ok'] for row in receipt['checks']}
    assert names['B.nothing_in_flight'] is False
    assert names['B.no_cap_violation'] is False
    assert receipt['arms']['B']['violations'] and '99' in receipt['arms']['B']['violations'][0]


def test_mixed_sampling_across_dispatches_is_not_uniform():
    rows = [dict(status='answered'), dict(status='answered', temperature=0.7)]
    receipt = compare(normalize_arm_document(arm_document(rows)),
                      normalize_host_ledger(ledger([record(1), record(2)]), policy()))
    assert receipt['arms']['A']['sampling']['temperature'] is None
    assert any('temperature: A=None' in row for row in receipt['unrecorded_sampling'])


def test_bad_records_are_refused_instead_of_counted():
    for broken in (dict(id=1, status='done', usage=None),
                   dict(id=2, status='answered', usage=dict(outputTokens=-1)),
                   dict(id=3, status='answered', usage=dict(outputTokens='20')),
                   dict(id=4, status='answered', usage=dict(inputTokens=1))):
        with pytest.raises(MeteringError):
            normalize_host_ledger(ledger([broken]), policy())
    with pytest.raises(MeteringError):
        normalize_arm_document({'arm': 'A', 'dispatches': [dict(status='nope')]})
    with pytest.raises(MeteringError):
        normalize_arm_document({'arm': 'A'})


def test_cli_reads_the_real_ledger_files_and_emits_one_json(tmp_path):
    import contextlib
    import io

    from operator_agent.app.cli import main as cli_main

    run = Path(tmp_path)/'run'
    run.mkdir()
    (run/'host-task-policy.json').write_text(json.dumps(policy()), encoding='utf-8')
    (run/'host-model-budget.json').write_text(json.dumps(ledger([record(1)])), encoding='utf-8')
    arm = Path(tmp_path)/'arm-a.json'
    arm.write_text(json.dumps(arm_document([dict(status='answered')])), encoding='utf-8')

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(['usage-reconcile', '--run-dir', str(run), '--a', str(arm)])
    assert code == 0
    receipt = json.loads(buf.getvalue())
    assert receipt['ok'] is True and receipt['sampling_comparability'] == 'established'
    assert set(receipt['inputs']) == {'run_dir', 'policy', 'ledger', 'arm_a'}
    assert set(receipt['arms']['B']['sampling']) == set(REQUIRED_SAMPLING_FIELDS)


def test_cli_reports_a_missing_input_instead_of_an_empty_receipt(tmp_path):
    import contextlib
    import io

    from operator_agent.app.cli import main as cli_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(['usage-reconcile', '--run-dir', str(tmp_path), '--a', str(tmp_path/'nope.json')])
    assert code == 2
    payload = json.loads(buf.getvalue())
    assert payload['ok'] is False and payload['error'] == 'usage_input'