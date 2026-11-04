"""Two-stage study registration: freeze local design, then bind measured instance."""
import copy
import importlib.metadata
from pathlib import Path
import sys

from . import connected_plan, plan as files, sampling, study_materials as materials

SCHEMA='phase1-study-preregistration-v1'
BUDGET=dict(wall_clock_s=2400.,max_llm_calls=30,oracle_total=20)
FINAL=dict(max_verifications=1,wall_clock_s=660.)
DESIGN=dict(A='DSH native shell/edit/read + public full-verification client',
    B='same DSH host + operator plugin delegation + production inner agents',
    starting_material='same declaration, builtin semantics and frozen contract; no starting candidate',
    initial_corpus='empty per invocation; inner additions remain run-local',
    deliberate_differences=['tool surface','prompts and built-in knowledge','context decomposition',
                            'native self-generation versus delegated inner strategy'],
    native_remote_access='verification via trusted SSH service; no arbitrary SSH session or credential exposure',
    visibility='outer host file boundary only; trusted inner and remote candidates are not hostile-code sandboxes',
    conclusion_scope='bounded workflows on four tasks; no generic Agent or individual plugin-component superiority claim',
    native_verification_limit='common oracle budget with protected confirmation; no extra round counter',
    inner_max_rounds=8)
OBSERVATION=dict(raw_host_transcript='retained',model_and_oracle_reservations='reconciled per cell',
    model_usage='validated API counters only; outer usage may be missing',cost='unknown without billing evidence',
    normalized_tools='partial; raw source retained',human_conversation='missing',external_edits='missing',
    human_count='unknown when coverage missing',human_minutes='unknown',
    runtime='measured and checked before each development cell and final evaluation')
STOP_RULES=['implementation/prompt/material drift stops new dispatch',
    'GPU UUID/name/index, runtime or runner drift refuses dispatch',
    'budget/deadline/cancel is terminal and never refunds a reservation',
    'keep failed, timed-out, cancelled and unstarted slots; retries require a new registration',
    'no candidate/feedback sharing across arms; no post-hoc task replacement']
OUTCOME=dict(correctness='full frozen L2',performance='at least one valid L3 and speedup>=1.0',
    selected_source='last native submission or production selected candidate, pinned before independent final',
    noise='report bands, contamination and ties; within-band changes do not establish stable gain',
    inference='four paired tasks; +1 is exploratory, not statistical significance')


def model(value):
    if value is None:return None
    # Reuse the connected adapter's closed endpoint/credential-reference schema.
    config=dict(backend='fake',entry_b='dsh',model=value,
        tasks=[dict(task_id='schema-check',declaration=dict(request='relu fp32',shapes=[[4,8]],dsl='python'))])
    result=connected_plan._normalize(config)['model']
    if result['temperature']!=sampling.POLICY['temperature']:
        raise ValueError('model temperature must equal the registered wire policy')
    return result


def _make(model_value):
    tasks=materials.tasks()
    versions={}
    for name in ('numpy','torch'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    value=dict(schema=SCHEMA,plan_id='phase1-agent-comparison-v2',
        status='preregistered_pending_instance',model=model_value,sampling=sampling.policy(),
        tasks=tasks,schedule=materials.schedule(),budget=dict(BUDGET),final_evaluation=dict(FINAL),
        design=copy.deepcopy(DESIGN),observation=copy.deepcopy(OBSERVATION),stop_rules=list(STOP_RULES),
        outcome=copy.deepcopy(OUTCOME),identity=materials.identity(),
        local_runtime=dict(python=sys.version,executable=sys.executable,dependencies=versions),
        host_runtime=connected_plan.host_identity('dsh'),
        bindings=dict(model_configured=model_value is not None,provider_behavior_verified=False,
            gpu=False,remote_runtime=False,runner_observed=False,card_gates=False),
        formal_execution_ready=False,formal_comparison_credit=False,gpu_acceptance=False)
    value['hash']=files.digest(value)
    return value


def prepare(output,model_config=None):
    value=_make(model(model_config))
    root=files.fresh_directory(output)
    files.write_once(root/'preregister.json',value)
    files.write_once(root/'instance-config.example.json',dict(
        connection=dict(host='REPLACE_SSH_ALIAS',work_dir='/absolute/prepared/runtime',timeout=300),
        gpu=dict(index=0,uuid='REPLACE_MEASURED_UUID',name='REPLACE_MEASURED_NAME'),
        probe='/absolute/measured-card-probe.json',instance_receipt='/absolute/instance-receipt.json'))
    return value


def validate(value,*,check_identity=True):
    if not isinstance(value,dict) or value.get('schema')!=SCHEMA:
        raise ValueError('not a study preregistration')
    if files.digest({k:v for k,v in value.items() if k!='hash'})!=value.get('hash'):
        raise ValueError('preregistration hash mismatch')
    raw=None if value.get('model') is None else {k:v for k,v in value['model'].items()
        if k in ('base_url','id','api_key_env','temperature','transport')}
    expected=_make(model(raw))
    # Historical inspection does not claim the current implementation is still
    # eligible. Execution and instance binding require exact current identity.
    if not check_identity:
        for key in ('identity','local_runtime','host_runtime'):expected[key]=value[key]
    expected['hash']=files.digest({k:v for k,v in expected.items() if k!='hash'})
    if expected!=value:raise ValueError('study design/material/model/implementation drift')
    return value


def load(path,**kwargs):return validate(files.read(path),**kwargs)


def instance_config(preregister,config):
    if not isinstance(config,dict) or set(config)!={'connection','gpu','probe','instance_receipt'}:
        raise ValueError('instance config requires connection/gpu/probe/instance_receipt')
    if preregister['model'] is None:raise ValueError('model is unbound; create a registration with model metadata')
    return dict(backend='ssh',entry_a='dsh_native',entry_b='dsh',sampling=preregister['sampling'],
        connection=config['connection'],gpu=config['gpu'],probe=config['probe'],
        model={k:v for k,v in preregister['model'].items() if k in
            ('base_url','id','api_key_env','temperature','transport')},
        tasks=[dict(task_id=t['task_id'],declaration=t['declaration']) for t in preregister['tasks']],
        budget=preregister['budget'],max_rounds=preregister['design']['inner_max_rounds'],
        final_wall_clock_s=preregister['final_evaluation']['wall_clock_s'])


def collect(preregister_path,config,output):
    preregister=load(preregister_path)
    normalized=connected_plan._normalize(instance_config(preregister,config))
    connected_plan._probe(normalized['probe'],normalized['gpu'])
    from . import study_instance
    receipt=study_instance.capture(normalized['connection'],normalized['gpu'],
        preregister['identity']['expected_remote_payload'])
    study_instance.validate(receipt,normalized['gpu'],preregister['identity']['expected_remote_payload'])
    result=dict(schema='phase1-study-collected-instance-v1',preregistration_hash=preregister['hash'],
                observation=receipt)
    files.write_once(output,result)
    return result


def bind(preregister_path,config,output):
    preregister=load(preregister_path)
    cfg=instance_config(preregister,config)
    collected=files.read(config['instance_receipt'])
    if (not isinstance(collected,dict) or set(collected)!={'schema','preregistration_hash','observation'}
            or collected.get('schema')!='phase1-study-collected-instance-v1'
            or collected.get('preregistration_hash')!=preregister['hash']):
        raise ValueError('instance receipt belongs to another preregistration')
    receipt=collected['observation']
    from . import study_instance
    study_instance.validate(receipt,cfg['gpu'],preregister['identity']['expected_remote_payload'])
    if study_instance._connection(cfg['connection'])!=receipt['connection']:
        raise ValueError('connection differs from the collected instance')
    cfg['connection']={k:v for k,v in receipt['connection'].items() if v is not None}
    plan=connected_plan.create(cfg)
    plan['study']=dict(schema='phase1-study-binding-v1',preregistration=preregister,
        instance_receipt=receipt,expected_runner_files=preregister['identity']['expected_remote_payload'],
        collection_preregistration_hash=collected['preregistration_hash'])
    plan['hash']=files.digest({k:v for k,v in plan.items() if k!='hash'})
    validate_binding(plan)
    root=files.fresh_directory(output)
    files.write_once(root/'plan.json',plan)
    return plan


def validate_binding(plan,*,check_identity=True):
    value=plan.get('study')
    if not isinstance(value,dict) or set(value)!={'schema','preregistration','instance_receipt',
                                                'expected_runner_files','collection_preregistration_hash'}:
        raise ValueError('invalid study binding')
    if value['schema']!='phase1-study-binding-v1':raise ValueError('unknown study binding')
    pre=validate(value['preregistration'],check_identity=check_identity)
    expected=pre['identity']['expected_remote_payload']
    from . import study_instance
    study_instance.validate(value['instance_receipt'],plan['gpu'],expected)
    tasks=[dict(task_id=t['task_id'],declaration=t['declaration'],contract_hash=t['contract_hash'],
                kernel_type=t['kernel_type']) for t in pre['tasks']]
    if (value['expected_runner_files']!=expected or value['collection_preregistration_hash']!=pre['hash']
            or study_instance._connection(plan['connection'])!=value['instance_receipt']['connection']
            or plan.get('entry_a')!='dsh_native' or plan.get('entry_b')!='dsh'
            or plan.get('model')!=pre['model'] or plan.get('sampling')!=pre['sampling']
            or plan.get('tasks')!=tasks or plan.get('schedule')!=pre['schedule']
            or plan.get('budget')!=pre['budget'] or plan.get('max_rounds')!=pre['design']['inner_max_rounds']
            or plan.get('final_wall_clock_s')!=pre['final_evaluation']['wall_clock_s']):
        raise ValueError('bound execution differs from the registered design')
    return plan
