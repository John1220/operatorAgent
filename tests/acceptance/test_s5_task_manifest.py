"""The S5 draft preserves quality inputs instead of silently changing tasks."""
from benchmarks.experiments.s5 import s5_task_manifest as s5
from operator_agent.decide.quality_suite import suite_manifest
from operator_agent.core.run_contract import digest


def test_existing_tasks_keep_the_original_contract_and_challenge_inputs():
    original = suite_manifest()['cases']
    proposed = s5.manifest()['tasks'][:10]
    for old, new in zip(original, proposed):
        assert {k: new[k] for k in old} == old
        assert new['exposure'] == 'known_historical'
        assert len(new['evaluation_inputs']) == 3


def test_new_candidates_keep_distinct_seeds_and_do_not_claim_unknown_evidence():
    value = s5.manifest()
    assert len(value['tasks']) == 14
    assert value['status'] == 'draft_not_sealed'
    assert value['holdouts_are_secret'] is False
    for row in value['tasks'][10:]:
        assert row['exposure'] == 'unknown'
        assert row['performance_feasibility'] == 'unknown'
        assert [x['inputs']['seed'] for x in row['evaluation_inputs']] == [row['seed'], 317, 619]
    assert value['hash'] == digest({k: v for k, v in value.items() if k != 'hash'})
