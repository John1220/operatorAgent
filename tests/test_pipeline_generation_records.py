"""Pipeline receipts follow real adapter behavior; all model replies are doubles.

These tests never import or execute a candidate and grant no GPU qualification.
"""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from operator_agent.coder_pipeline import CoderPipeline
from operator_agent.generation_context import CONTEXT_SCHEMA, RECORD_SCHEMA
from operator_agent.interface_contract import broadcast_calling_convention
from operator_agent.prompt_manager import PromptManager
from operator_agent.run_contract import code_hash, digest
from operator_agent.schemas import (
    ConstraintSpec, Precision, SourceSpec, TargetSpec, TaskSchema, UnifiedSketch,
)


PLAN = dict(algorithm='layernorm', kernel_signature='def kernel(X)',
            jit_signature='', steps=['allocate', 'launch'], risks=[], notes='')
GOOD = 'import numpy as np\n\ndef kernel(X):\n    return X\n'
BAD = 'import numpy as np\n\ndef kernel(X:\n    return X\n'


def task(dsl='python'):
    return TaskSchema(source=SourceSpec(kernel_type='layernorm'),
        target=TargetSpec(hardware='mock_gpu', dsl=dsl),
        constraints=ConstraintSpec(precision=Precision.FP32))


def sketch(dsl='python'):
    return UnifiedSketch(algorithm='layernorm', target_dsl=dsl)


class FakeLLM:
    model = 'offline-pipeline-fixture'

    def __init__(self, handler=None):
        self.handler = handler or (lambda messages, **kw: GOOD)
        self.calls = []
        self.plans = []
        self.lock = threading.Lock()

    def chat_json(self, messages, **kwargs):
        self.plans.append((messages, kwargs))
        return dict(PLAN)

    def chat(self, messages, **kwargs):
        with self.lock:
            self.calls.append((messages, kwargs))
        return self.handler(messages, **kwargs)


def pipeline(llm=None, *, template='', compile_fn=None):
    return CoderPipeline(llm or FakeLLM(),
        kernel_db=SimpleNamespace(search=lambda **kw: []),
        rule_coder=SimpleNamespace(generate=lambda *args: template),
        compile_fn=compile_fn)


def assert_receipt(record):
    assert record['schema'] == RECORD_SCHEMA
    assert record['context']['schema'] == CONTEXT_SCHEMA
    assert record['source_hash'] == (code_hash(record['source']) if record['source'] else None)
    assert record['record_hash'] == digest({k: v for k, v in record.items() if k != 'record_hash'})
    assert record['context_hash'] == record['context']['hash']
    assert record['correctness_proven'] is False
    assert record['restored_eligibility'] is False
    assert record['gpu_measurements'] == 0
    assert record['candidate_execution'] is False
    json.dumps(record, allow_nan=False)


def model_samples(pipe):
    return [r for r in pipe.candidate_records
            if r['request'] and r['request']['caller'] == 'CoderPipeline.Coder'
            and r['status'] in ('proposed', 'model_failed', 'no_candidate')]


def test_direct_sample_keeps_the_original_wire_prompt_and_string_result():
    selected, designed = task(), sketch()
    llm = FakeLLM(lambda *args, **kw: 'Code:\n```python\n' + GOOD + '```')
    pipe = pipeline(llm)
    convention = broadcast_calling_convention({'X': None})
    result = pipe._sample_one('caller-owned plan', designed, selected, '', .4, convention)
    expected = PromptManager().render('coder_from_plan',
        _agent_role='coder', _skill_name='triton_skill', plan='caller-owned plan',
        kernel_type=selected.source.kernel_type, hardware=selected.target.hardware,
        dsl=designed.target_dsl, precision=selected.constraints.precision.value,
        block_m=designed.tiling.block_m, block_n=designed.tiling.block_n,
        block_k=designed.tiling.block_k, ref_code='无', optimization_hints='无',
        calling_convention=convention)
    assert result == GOOD.strip()
    assert llm.calls == [([dict(role='user', content=expected)],
                         dict(temperature=.4, max_tokens=pipe.GEN_MAX_TOKENS,
                              caller_tag='CoderPipeline.Coder'))]
    record = pipe.candidate_records[0]
    assert record['request']['messages_hash'] == digest(llm.calls[0][0])
    assert_receipt(record)


def test_parallel_receipts_follow_temperature_slots_despite_reverse_completion():
    high_done, middle_done = threading.Event(), threading.Event()
    completed = []

    def respond(messages, *, temperature, **kw):
        if temperature == .1:
            assert middle_done.wait(5)
        elif temperature == .4:
            assert high_done.wait(5)
        completed.append(temperature)
        if temperature == .7:
            high_done.set()
        elif temperature == .4:
            middle_done.set()
        return GOOD + '# temperature ' + str(temperature)

    llm, inputs = FakeLLM(respond), {'X': np.zeros((2, 3), dtype=np.float32)}
    pipe = pipeline(llm)
    result = pipe.run(sketch(), task(), test_inputs=inputs)
    assert completed == [.7, .4, .1]
    assert len(llm.plans) == 1 and len(llm.calls) == 3
    samples = model_samples(pipe)
    assert [r['attempt'] for r in samples] == [1, 2, 3]
    assert [r['request']['temperature'] for r in samples] == [.1, .4, .7]
    assert len({r['context_hash'] for r in samples}) == 1
    assert all(r['context']['input_metadata_status'] == 'known' for r in samples)
    assert samples[0]['context']['interface']['broadcast'] == broadcast_calling_convention(inputs)
    assert result.endswith('# temperature 0.1')
    assert pipe.last_finalists == [samples[0]['source'], samples[1]['source']]
    assert pipe.last_candidate_record['attempt'] == 1
    assert pipe.last_candidate_record['static_precheck'] == 'passed'
    assert all(r['static_precheck'] == 'not_checked' for r in samples)
    for record in pipe.candidate_records:
        assert_receipt(record)


def test_normalization_has_separate_non_model_lineage_and_retains_raw_responses():
    original = ('import triton\nimport triton.language as tl\nBLOCK_M = 256\n'
                '@triton.jit\ndef helper(X):\n    pass\n'
                'def kernel(X):\n    return X\n')
    llm = FakeLLM(lambda *args, **kw: original)
    pipe = pipeline(llm, compile_fn=lambda *args: (True, []))
    result = pipe.run(sketch('triton'), task('triton'))
    assert result != original.strip() and 'BLOCK_M = 64' in result
    samples = model_samples(pipe)
    assert len(samples) == len(llm.calls) == 3
    assert all(r['source'] == original.strip() for r in samples)
    assert all(r['source_author'] == 'model' for r in samples)
    transformed = [r for r in pipe.candidate_records
                   if r['source_author'] == 'mechanical_transform']
    assert transformed
    assert all(r['parent_source_hash'] == code_hash(original.strip()) for r in transformed)
    assert all(r['request'] is None and r['transformation'] for r in transformed)
    assert pipe.last_candidate_record['source'] == result
    assert pipe.last_candidate_record['source_author'] == 'mechanical_transform'
    assert pipe.last_candidate_record['static_precheck'] == 'passed'
    for record in pipe.candidate_records:
        assert_receipt(record)


def test_template_fallback_is_not_attributed_to_a_model_or_prechecked():
    checked = []

    def reject(code, dsl):
        checked.append(code)
        return False, ['fixture compile refusal']

    template = GOOD + '# rule template'
    llm = FakeLLM(lambda *args, **kw: BAD)
    pipe = pipeline(llm, template=template, compile_fn=reject)
    assert pipe.run(sketch(), task()) == template
    assert len(llm.plans) == 1 and len(llm.calls) == 5
    assert len([r for r in pipe.candidate_records if r['request']
                and r['request']['caller'] == 'CoderPipeline.Repair'
                and r['status'] == 'proposed']) == 2
    record = pipe.last_candidate_record
    assert record['source_author'] == 'rule_template'
    assert record['request'] is None
    assert record['static_precheck'] == 'not_checked'
    assert template not in checked
    assert pipe.last_finalists == [template]
    assert_receipt(record)


def test_repair_records_each_parent_without_adding_dispatches():
    middle = BAD + '# repair one'
    repairs = []

    def respond(messages, *, caller_tag, **kw):
        if caller_tag == 'CoderPipeline.Repair':
            repairs.append(messages)
            return middle if len(repairs) == 1 else GOOD
        return BAD

    llm, pipe = FakeLLM(respond), None
    pipe = pipeline(llm)
    assert pipe.run(sketch(), task()) == GOOD.strip()
    assert len(llm.plans) == 1 and len(llm.calls) == 5 and len(repairs) == 2
    receipts = [r for r in pipe.candidate_records if r['request']
                and r['request']['caller'] == 'CoderPipeline.Repair'
                and r['status'] == 'proposed']
    assert [r['attempt'] for r in receipts] == [1, 2]
    assert [r['parent_source_hash'] for r in receipts] == [code_hash(BAD.strip()), code_hash(middle.strip())]
    assert pipe.last_candidate_record['source_hash'] == code_hash(GOOD.strip())
    assert pipe.last_candidate_record['parent_source_hash'] == code_hash(middle.strip())
    assert pipe.last_candidate_record['static_precheck'] == 'passed'
    for record in pipe.candidate_records:
        assert_receipt(record)


def test_failed_parallel_call_preserves_all_slots_and_the_original_exception():
    failure = RuntimeError('offline fixture transport failure')

    def respond(messages, *, temperature, **kw):
        if temperature == .1:
            raise failure
        return '' if temperature == .4 else GOOD

    llm = FakeLLM(respond)
    pipe = pipeline(llm)
    with pytest.raises(RuntimeError) as raised:
        pipe.run(sketch(), task())
    assert raised.value is failure
    assert len(llm.plans) == 1 and len(llm.calls) == 3
    records = pipe.candidate_records
    assert [r['attempt'] for r in records] == [1, 2, 3]
    assert [r['status'] for r in records] == ['model_failed', 'no_candidate', 'proposed']
    assert records[0]['source'] is records[1]['source'] is None
    assert pipe.last_candidate_record is None
    for record in records:
        assert_receipt(record)


def test_empty_samples_do_not_invent_a_repair_or_a_candidate():
    llm = FakeLLM(lambda *args, **kw: '')
    pipe = pipeline(llm)
    assert pipe.run(sketch(), task()) == ''
    assert len(llm.calls) == 3 and len(llm.plans) == 1
    assert len(pipe.candidate_records) == 3
    assert all(r['status'] == 'no_candidate' for r in pipe.candidate_records)
    assert pipe.last_candidate_record is None
    assert pipe.last_finalists == []


def test_a_reused_pipeline_exposes_only_the_current_runs_receipts():
    llm = FakeLLM()
    pipe = pipeline(llm)
    first = pipe.run(sketch(), task())
    previous_records = list(pipe.candidate_records)
    llm.handler = lambda *args, **kw: GOOD + '# second run'
    second = pipe.run(sketch(), task())
    assert first != second and len(llm.calls) == 6
    assert len(model_samples(pipe)) == 3
    assert all(r['source'].endswith('# second run') for r in pipe.candidate_records)
    assert pipe.last_candidate_record['source'] == second
    assert previous_records[0]['source'] == first


def test_legacy_unknown_inputs_stay_unknown_and_do_not_block_dispatch():
    llm = FakeLLM()
    pipe = pipeline(llm)
    assert pipe.run(sketch(), task(), test_inputs={'X': None}) == GOOD.strip()
    assert len(llm.calls) == 3
    context = pipe.last_candidate_record['context']
    assert context['input_metadata'] is None
    assert context['input_metadata_status'] == 'unknown'
    assert context['interface']['broadcast'] == broadcast_calling_convention({'X': None})


def test_sample_render_preserves_the_sketch_dsl_when_task_dsl_differs():
    llm = FakeLLM()
    pipe = pipeline(llm)
    selected, designed = task('python'), sketch('triton')
    pipe._sample_one('plan', designed, selected, '', .1, 'caller-provided ABI')
    record = pipe.candidate_records[0]
    assert record['context']['task']['dsl'] == 'python'
    assert record['context']['policy']['render_dsl'] == 'triton'
    expected = PromptManager().render('coder_from_plan',
        _agent_role='coder', _skill_name='triton_skill', plan='plan',
        kernel_type='layernorm', hardware='mock_gpu', dsl='triton', precision='fp32',
        block_m=designed.tiling.block_m, block_n=designed.tiling.block_n,
        block_k=designed.tiling.block_k, ref_code='无', optimization_hints='无',
        calling_convention='caller-provided ABI')
    assert llm.calls[0][0] == [dict(role='user', content=expected)]


def test_a_failing_compile_callback_does_not_create_a_pass_receipt():
    def crash(*args):
        raise RuntimeError('offline gate unavailable')

    llm = FakeLLM()
    pipe = pipeline(llm, compile_fn=crash)
    with pytest.raises(RuntimeError, match='gate unavailable'):
        pipe.run(sketch(), task())
    assert len(llm.calls) == 3
    assert len(pipe.candidate_records) == 3
    assert all(r['status'] == 'proposed' for r in pipe.candidate_records)
    assert pipe.last_candidate_record is None
