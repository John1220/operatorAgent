"""Local preregistration materials; no model, GPU or archived-run observations.

Task declarations and the whole finite L2 input domain are owned by the existing
production contract. Large primary inputs are described without allocating them;
small built-in L2 fixtures are materialized on CPU to retain actual stride facts.
This is expected local identity, never proof of a remote runtime or execution.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from operator_agent import __version__
from operator_agent.core.run_contract import digest, json_value
from operator_agent.execute.remote_executor import SSHExecutor

from . import connected_plan, plan
from .support import check

REPO = Path(__file__).resolve().parents[3]
SCHEMA = 'phase1-study-materials-v1'
TASKS = (('matmul-4096', 'matmul', (4096,4096,4096)),
         ('relu-large', 'relu', (16384,6144)),
         ('softmax-4096', 'softmax', (4096,4096)),
         ('relu-small', 'relu', (1024,768)))
MEASUREMENT_FILES = (
    'benchmarks/probes/unknown_card_gates.py',
    'benchmarks/probes/collect_unknown_card_probe.py',
    'benchmarks/probes/collect_shape_inventory.py',
    'benchmarks/probes/probe_interleaved_ab.py',
    'benchmarks/measurement/measure_launch_floor.py',
    'benchmarks/calibration/calibrate_achievable_peak.py',
    'benchmarks/calibration/calibrate_dispersion_thresholds.py',
    'operator_agent/decide/measurement.py',
    'operator_agent/decide/calibration.py',
    'operator_agent/decide/cuda_calibration.py',
    'operator_agent/decide/cuda_ruler_profile.py',
)
EVALUATOR_FILES = (
    'benchmarks/experiments/phase1/connected_final.py',
    'benchmarks/experiments/phase1/connected_summary.py',
    'benchmarks/experiments/phase1/summary.py',
    'operator_agent/decide/verifier.py',
    'operator_agent/generate/repair_loop.py',
    'operator_agent/core/contract.py',
    'operator_agent/core/tolerance.py',
    'operator_agent/decide/correctness.py',
    'operator_agent/decide/correctness_profiles.py',
    'operator_agent/decide/correctness_boundaries.py',
)


def _file(path):
    path = Path(path)
    if path.is_symlink() or REPO.resolve() not in path.resolve().parents or not path.is_file():
        raise ValueError('study identity requires a regular repository file: '+str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(names):
    return {name:_file(REPO/name) for name in sorted(names)}


def _symbol(path, name):
    source=(REPO/path).read_text(encoding='utf-8')
    functions=[n for n in ast.walk(ast.parse(source)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if len(functions)!=1:
        raise ValueError('reference symbol missing or ambiguous: '+path+':'+name)
    fragment=ast.get_source_segment(source,functions[0])
    return dict(path=path,file_sha256=_file(REPO/path),symbol=name,
                symbol_sha256=hashlib.sha256(fragment.encode('utf-8')).hexdigest())


def _reference(kernel):
    return dict(kind='builtin_reference_expected_not_runtime_observed',framework='pytorch',
        semantics={'matmul':'A @ B','relu':'torch.relu(X)',
                   'softmax':'torch.nn.functional.softmax(X, dim=-1)'}[kernel],
        expected_remote_baseline='pytorch_eager',runtime_version=None,
        sources=[_symbol('remote_scripts/run_kernel.py','_reference_torch'),
                 _symbol('operator_agent/decide/verifier.py','_build_reference_fn')])


def _contiguous(tensor):
    # Primary BenchmarkCase materialization uses the shared seed protocol to
    # produce ordinary C-order arrays. Describing 100M elements must not allocate
    # them just to freeze the preregistration.
    size={'fp16':2,'fp32':4}[tensor.dtype]
    strides=[];stride=size
    for dimension in reversed(tensor.shape):
        strides.insert(0,stride);stride*=dimension
    return dict(name=tensor.name,shape=list(tensor.shape),dtype=tensor.dtype,
                itemsize=size,strides_bytes=strides,layout='contiguous')


def _inputs(contract):
    primary=[]
    for case in contract.cases:
        body=case.content_dict()
        primary.append(dict(case_id=case.key(),declaration=body,declaration_hash=digest(body),
            seed_spec=case.seed_spec(),tensors=[_contiguous(t) for t in sorted(case.tensors,key=lambda t:t.name)],
            stride_source='C-order shared seed protocol; large arrays not allocated'))
    auxiliary=[]
    for case in sorted(contract.correctness_suite.cases,key=lambda c:c.case_id):
        # Guard against a future recipe unexpectedly growing into the large
        # performance domain. Freeze must remain an offline, bounded operation.
        if sum(math.prod(t.shape) for t in case.inputs.tensors)>65536:
            raise ValueError('auxiliary L2 fixture exceeds preregistration CPU bound')
        arrays=case.materialize()
        body=case.content_dict()
        declared={t.name:t for t in case.inputs.tensors}
        auxiliary.append(dict(case_id=case.case_id,declaration=body,declaration_hash=digest(body),
            tensors=[dict(name=name,shape=list(value.shape),dtype=declared[name].dtype,
                itemsize=value.dtype.itemsize,strides_bytes=list(value.strides),layout=case.layout)
                for name,value in sorted(arrays.items())],
            stride_source='observed from trusted CPU fixture; no candidate execution'))
    result=dict(primary_cases=primary,auxiliary_l2_cases=auxiliary,
        l3_scope='primary cases only; auxiliary correctness fixtures never enter L3',
        primary_count=len(primary),auxiliary_l2_count=len(auxiliary),
        deterministic=contract.correctness_suite.deterministic,
        required_checks=list(contract.correctness_suite.required_checks))
    result['hash']=digest(result)
    return result


def tasks():
    """Fresh JSON values for all four fixed fp16/Triton task contracts."""
    result=[]
    for task_id,kernel,shape in TASKS:
        declaration=dict(request=kernel+' fp16',precision='fp16',dsl='triton',
                         shapes=[list(shape)],min_speedup=1.,hardware='nvidia_gpu')
        _,contract=check(declaration,'ssh')
        result.append(dict(task_id=task_id,declaration=declaration,kernel_type=kernel,
            contract=contract.as_dict(),contract_hash=contract.hash,
            input_manifest=_inputs(contract),reference=_reference(kernel)))
    return json_value(result)


def schedule():
    """All eight slots, fixed A/B, B/A, A/B, B/A order."""
    return plan.schedule([dict(task_id=task_id) for task_id,_,_ in TASKS])


def _git():
    result={}
    for name,args in (('head',['rev-parse','HEAD']),('branch',['rev-parse','--abbrev-ref','HEAD']),
                      ('status_porcelain',['status','--porcelain=v1','--untracked-files=normal'])):
        completed=subprocess.run(['git','-C',str(REPO),*args],capture_output=True,text=True,timeout=10)
        if completed.returncode:
            result[name]=None
        else:
            result[name]=completed.stdout.rstrip('\n')
    result['dirty']=(bool(result['status_porcelain']) if result['status_porcelain'] is not None else None)
    return result


def identity():
    """Expected local sources/payloads only; does not connect to any backend."""
    # Reuse the executor's payload inventory without constructing an executor:
    # its constructor would resolve SSH/provisioning state. Target basenames are
    # useful later for checking a prepared instance, not evidence of deployment.
    payload=SimpleNamespace(platform='nvidia',remote_runner_path='/unbound/run_kernel.py',
                            _runner_script='run_kernel.py')
    remote={Path(target).name:dict(source=str(Path(source).relative_to(REPO)),sha256=_file(source))
            for source,target in SSHExecutor._remote_files(payload)}
    runtime=[]
    for folder in ('operator_agent','remote_scripts'):
        for path in sorted((REPO/folder).rglob('*')):
            if '__pycache__' in path.parts:continue
            if path.is_symlink():raise ValueError('runtime source symlink is not an immutable inventory')
            if path.is_file() and path.suffix in ('.py','.md','.json','.yaml','.yml'):
                runtime.append(str(path.relative_to(REPO)))
    # S5 adapters import shared controllers from other experiment packages.
    # Freeze their Python source too, so a transitive helper cannot drift while
    # the primary phase1/S5 entry files keep the same hash.
    runtime.extend(str(path.relative_to(REPO)) for path in sorted((REPO/'benchmarks').rglob('*.py'))
                   if '__pycache__' not in path.parts)
    runtime.append('pyproject.toml')
    implementation=connected_plan.implementation_identity()
    return dict(schema=SCHEMA,scope='expected_local_identity_not_remote_observed',
        implementation=implementation,implementation_hash=digest(implementation),
        runtime_sources=_sources(runtime),expected_remote_payload=remote,
        measurement_sources=_sources(MEASUREMENT_FILES),evaluator_sources=_sources(EVALUATOR_FILES),
        package_versions=dict(python=__version__,plugin=json.loads(
            (REPO/'integrations/dsh-operator/package.json').read_text())['version']),
        git=_git(),remote_runner_observed=False,gpu_bound=False,model_bound=False,
        formal_execution_ready=False)
