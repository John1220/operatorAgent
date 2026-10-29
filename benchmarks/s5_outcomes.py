"""Classify ordinary task noncompletion separately from infrastructure faults."""
def classify(record):
    if record.get('execution_state')=='not_executed':return 'not_executed'
    if record.get('error') and record['error']!='TimeoutExpired':return 'infrastructure_error'
    gateway=record.get('gateway_diagnostics')
    if gateway is not None:
        if (not isinstance(gateway,dict) or gateway.get('schema')!='s5-upstream-failures-v1'
                or not isinstance(gateway.get('failures'),list) or gateway.get('truncated') is True
                or gateway['failures']):
            return 'infrastructure_error'
    if record.get('development_audit',{}).get('complete') is not True:
        return 'infrastructure_error'
    if record.get('workspace_archive',{}).get('complete') is False:
        return 'infrastructure_error'
    if any(r.get('status')=='unknown' for r in record.get('development',{}).get('records',[])):
        return 'infrastructure_error'
    if record.get('error')=='TimeoutExpired':return 'wall_clock_exhausted'
    if record.get('status') in ('passed','failed'):
        return 'completed' if record['status']=='passed' else 'candidate_rejected'
    if record.get('verification') is not None:return 'infrastructure_error'
    if record.get('budget_refusal') in ('call_budget_exhausted','output_budget_exhausted',
                                      'per_call_output_exhausted','wall_clock_exhausted'):
        return record['budget_refusal']
    if record.get('execution_state')=='interrupted' and record.get('terminal_kind')=='completed':
        return 'no_submission'
    return 'infrastructure_error'


def fatal(record):
    return classify(record)=='infrastructure_error'
