"""Safe gateway failures remain infrastructure unknowns, not model failures."""
from benchmarks.experiments.s5.s5_outcomes import classify


def base_record():
    return dict(execution_state='interrupted', status='unknown', error=None,
        development_audit=dict(complete=True), workspace_archive=dict(complete=True),
        development=dict(records=[]), terminal_kind='completed',
        budget_refusal='call_budget_exhausted')


def test_known_budget_stop_is_distinct_from_transport_failure():
    record=base_record()
    record['gateway_diagnostics']=dict(schema='s5-upstream-failures-v1',calls_created=24,
        failures=[],truncated=False,provider_billing_known=False)
    assert classify(record)=='call_budget_exhausted'


def test_any_upstream_transport_failure_blocks_inference_even_if_a_candidate_passed():
    record=base_record()
    record.update(status='passed',execution_state='finished',verification=dict(status='passed'))
    record['gateway_diagnostics']=dict(schema='s5-upstream-failures-v1',calls_created=2,
        failures=[dict(call_sequence=2,stage='response_headers',code='http_status',
                       http_status=502,cleanup='direct_child_reaped_descendants_unknown',
                       outcome='unknown')],truncated=False,provider_billing_known=False)
    assert classify(record)=='infrastructure_error'


def test_malformed_or_truncated_gateway_diagnostics_are_unknown():
    for detail in (dict(schema='wrong', failures=[]),
                   dict(schema='s5-upstream-failures-v1', failures='missing'),
                   dict(schema='s5-upstream-failures-v1', failures=[], truncated=True)):
        assert classify(dict(base_record(), gateway_diagnostics=detail))=='infrastructure_error'


def test_performance_record_can_omit_non_applicable_development_audit():
    record=base_record()
    record.update(status='passed',execution_state='finished',verification=dict(status='passed'),
                  development_audit=None)
    record['gateway_diagnostics']=dict(schema='s5-upstream-failures-v1',calls_created=2,
        failures=[],truncated=False,provider_billing_known=False)
    assert classify(record)=='infrastructure_error'
    assert classify(record,require_development_audit=False)=='completed'
