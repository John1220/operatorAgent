"""Per-cell live bindings; no credentials are serialized into experiment files."""
from contextlib import contextmanager
import os
from pathlib import Path

from operator_agent.app import hitl_tools as ht
from operator_agent.execute.llm_client import LLMClient, _resolve_api_key
from operator_agent.execute.s5_inner_budget import SharedProductionBudget
from operator_agent.execute.task_budget import TaskBudgetConfig
from .sampling import request_timeout


def credential(model):
    value = os.environ.get(model['api_key_env'])
    if not value and model['api_key_env'] == 'LLM_API_KEY':
        value = _resolve_api_key(None)
    if not value or '\n' in value or '\r' in value:
        raise ValueError('configured model credential is missing or invalid')
    return value


@contextmanager
def environment(config):
    from operator_agent.execute.s5_verify_client import enabled
    if enabled():
        raise ValueError('connected worker refuses inherited S5 verification delegation')
    model = config['model']
    values = dict(LLM_BASE_URL=model['base_url'], LLM_MODEL=model['id'],
        LLM_API_KEY=credential(model), OPERATOR_AGENT_EXISTING_REMOTE_ONLY='1',
        OPERATOR_AGENT_S5_SHARED_LEDGER=config['shared_ledger'],
        OPERATOR_AGENT_S5_TASK_ID=config['shared_task_id'])
    if 'sampling' in config:values['LLM_THINKING_PARAMETER']='enable_thinking'
    if config['backend']=='ssh':
        gpu = config['gpu']
        values.update(OPERATOR_AGENT_REQUIRED_GPU=str(gpu['index']),
            OPERATOR_AGENT_REQUIRED_GPU_UUID=gpu['uuid'], OPERATOR_AGENT_REQUIRED_GPU_NAME=gpu['name'])
    else:
        values.update({key:None for key in ('OPERATOR_AGENT_REQUIRED_GPU',
            'OPERATOR_AGENT_REQUIRED_GPU_UUID','OPERATOR_AGENT_REQUIRED_GPU_NAME')})
    previous = {key:os.environ.get(key) for key in values}
    try:
        for key,value in values.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value
        yield
    finally:
        for key,value in previous.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value


def options(config):
    """把连接的执行选项交给后端。**常驻会话默认开**。

    为什么不是 False：一个 L2 完整套件是 15 个 case，非常驻时每个 case 都要重新
    SSH + 起远端 python + import torch（6.3 s 起）。2026-11-04 在同一张卡、同一个
    候选上实测：非常驻 L1+L2 **384.9 s**，常驻 **60.0 s**；全量（含 L3）从 405 s
    降到 **31.9 s**。A 臂单格因此会从 ~25 min 降到个位数分钟 —— 40 min 的单格墙钟
    才有机会装下「找到能过的候选」这件事。
    """
    if config['backend']=='fake':
        return {}
    value = dict(config['connection'])
    value['persistent'] = True
    return value


def client(config, directory):
    policy = TaskBudgetConfig.from_dict(config['budget'])
    timeout = request_timeout(config)
    budget = SharedProductionBudget(policy.llm_limits(timeout), Path(directory)/'model-journal',
        shared_path=config['shared_ledger'], task_id=config['shared_task_id'],
        deadline=config['deadline_monotonic'])
    model = config['model']
    return LLMClient(base_url=model['base_url'], model=model['id'], api_key=credential(model),
        temperature=model['temperature'], max_tokens=4096, enable_thinking=False,
        timeout=timeout, call_budget=budget)


def backend(config):
    return ht._backend(config['backend'], **options(config))
