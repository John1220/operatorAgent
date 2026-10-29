"""Portable retained-source scalar hypotheses; no GPU execution is simulated."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from operator_agent.compile_evidence import validate_compile_failure
from operator_agent.repair_scalar_diagnoser import RECOGNIZER, scalar_conversion_sites


FIXTURES = Path(__file__).parent / 'fixtures'
RETAINED = {
    'tile': '324b3ca8240efa8b4bdb12e6a94b98a09b0099f8e8c4172daec242955c987e3f',
    'guarded_tile': '9b9771f7a70a63f30cf10e281c2139d8e10fe7df7975cb78c5cef1938b02fbdb',
    'row': '2f0b0ef59819dfc1917e13a00ef33ba9576591b1ebd8cdce8fe22589806e5735',
}


def source(kind='tile'):
    return (FIXTURES / ('layernorm_single_column_scalar_' + kind + '.txt')).read_text()


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def seal(payload):
    body = {key: value for key, value in payload.items() if key != 'evidence_hash'}
    payload['evidence_hash'] = sha(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=False, allow_nan=False))
    return payload


def event(text, *, receiver='n_cols', node_kind='call'):
    """Construct source-bound typed data, without claiming compiler production.

    Native class/phase capture is separately tested in compile_evidence tests.
    This factory exercises the consumer's exact source and locus validation.
    """
    tree = ast.parse(text)
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != 'kernel')
    call = next(node for node in ast.walk(helper) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver)
    node = call
    if node_kind == 'attribute':
        node = call.func
    elif node_kind == 'assignment':
        node = next(node for node in helper.body if isinstance(node, ast.Assign) and node.value is call)
    elif node_kind == 'name':
        node = call.func.value
    payload = dict(schema='candidate-jit-compile-failure-v1',
                   producer='triton.compiler.errors.CompilationError', phase='candidate_launch',
                   source_hash=sha(text), helper_name=helper.name,
                   line=node.lineno, column=node.col_offset,
                   end_line=node.end_lineno, end_column=node.end_col_offset,
                   node_ast_hash=sha(ast.dump(node, include_attributes=False)))
    return seal(payload)


def diagnose(text, payload=None, rank=2):
    return scalar_conversion_sites(text, ast.parse(text), event(text) if payload is None else payload,
                                   input_rank=rank)


@pytest.mark.parametrize('kind', RETAINED)
@pytest.mark.parametrize('node_kind', ['call', 'attribute', 'assignment'])
def test_three_exact_retained_failures_bind_dimension_launch_and_native_locus(kind, node_kind):
    text = source(kind)
    assert sha(text) == RETAINED[kind]
    tree, payload = ast.parse(text), event(text, node_kind=node_kind)
    original_tree, original_payload = ast.dump(tree, include_attributes=True), copy.deepcopy(payload)
    sites = scalar_conversion_sites(text, tree, payload, input_rank=2)
    assert RECOGNIZER == 'bound-single-column-scalar-method-v1'
    assert [site['role'] for site in sites] == [
        'input_columns', 'launch_dimension', 'helper_dimension', 'scalar_conversion', 'compiler_failure_node']
    assert sites[0]['excerpt'] == 'n_cols = X.shape[1]'
    assert sites[1]['excerpt'] == sites[2]['excerpt'] == 'n_cols'
    assert sites[3]['excerpt'] == 'n_cols.to(tl.float32)'
    for site in sites:
        assert site['trust'] == 'untrusted_source_data'
        assert sha(site['excerpt']) == site['excerpt_sha256']
        assert site['line'] <= site['end_line']
    assert ast.dump(tree, include_attributes=True) == original_tree
    assert payload == original_payload
    assert text == source(kind)


@pytest.mark.parametrize('transform', [
    lambda s: s.replace('    n_rows = X.shape[0]\n    n_cols = X.shape[1]', '    n_rows, n_cols = X.shape'),
    lambda s: s.replace('        X, Y, n_rows, n_cols,', '        X, Y, n_rows, X.shape[1],'),
    lambda s: '"""纯源码，不执行候选。"""\n' + s,
    lambda s: s.replace('    n = n_cols.to(tl.float32)', '    n = n_cols.to(\n        tl.float32\n    )'),
])
def test_equivalent_bounded_shape_and_locus_forms(transform):
    assert diagnose(transform(source()))


@pytest.mark.parametrize('rank', [None, False, True, 0, 1, 3, 2.0, '2', [2]])
def test_rank_requires_explicit_typed_two(rank):
    assert diagnose(source(), rank=rank) is None


@pytest.mark.parametrize('field,value', [
    ('schema', 'CompilationError'), ('producer', 'other.CompilationError'), ('phase', 'reference'),
    ('source_hash', '0' * 64), ('helper_name', 'kernel'), ('line', 1), ('column', 0),
    ('end_line', 99), ('end_column', 99), ('node_ast_hash', '0' * 64),
    ('evidence_hash', '0' * 64), ('line', True),
])
def test_typed_payload_tampering_never_becomes_source_evidence(field, value):
    text = source()
    payload = event(text)
    payload[field] = value
    assert diagnose(text, payload) is None


@pytest.mark.parametrize('payload', [None, {}, [], 'CompilationError at n_cols.to',
                                    {'error': 'CompilationError: n_cols.to(tl.float32)'}])
def test_missing_or_prose_evidence_is_unknown(payload):
    text = source()
    assert scalar_conversion_sites(text, ast.parse(text), payload, input_rank=2) is None


def test_extra_prose_field_and_stale_source_are_unknown():
    text = source()
    payload = event(text)
    payload['error_message'] = 'int object has no attribute to'
    assert diagnose(text, seal(payload)) is None
    assert diagnose(text + '\n', event(text)) is None


@pytest.mark.parametrize('node_kind,receiver', [('name', 'n_cols'), ('call', 'y')])
def test_even_source_valid_event_must_bind_exact_scalar_conversion(node_kind, receiver):
    text = source('row')
    payload = event(text, node_kind=node_kind, receiver=receiver)
    assert validate_compile_failure(payload, text)
    assert diagnose(text, payload) is None


MUTATIONS = [
    ('host column alias', lambda s: s.replace('    n_cols = X.shape[1]', '    size = X.shape[1]\n    n_cols = size')),
    ('host dimension rebinding', lambda s: s.replace('    n_cols = X.shape[1]', '    n_cols = X.shape[1]\n    n_cols = 1')),
    ('host dimension row', lambda s: s.replace('    n_cols = X.shape[1]', '    n_cols = X.shape[0]')),
    ('launch dimension row', lambda s: s.replace('        X, Y, n_rows, n_cols,', '        X, Y, n_rows, n_rows,')),
    ('launch dimension stride', lambda s: s.replace('        X, Y, n_rows, n_cols,', '        X, Y, n_rows, stride_xc,')),
    ('launch dimension constant', lambda s: s.replace('        X, Y, n_rows, n_cols,', '        X, Y, n_rows, 1,')),
    ('helper parameter rebinding', lambda s: s.replace('    n = n_cols.to', '    n_cols = 1\n    n = n_cols.to')),
    ('helper augmented binding', lambda s: s.replace('    n = n_cols.to', '    n_cols += 0\n    n = n_cols.to')),
    ('helper unknown call', lambda s: s.replace('    n = n_cols.to', '    surprise = foo()\n    n = n_cols.to')),
    ('helper control flow', lambda s: s.replace('    n = n_cols.to', '    if n_cols > 1:\n        temp = 1\n    n = n_cols.to')),
    ('helper import', lambda s: s.replace('    n = n_cols.to', '    import os\n    n = n_cols.to')),
    ('helper module rebinding', lambda s: s.replace('    n = n_cols.to', '    tl = 1\n    n = n_cols.to')),
    ('helper pointer rebinding', lambda s: s.replace('    n = n_cols.to', '    X = Y\n    n = n_cols.to')),
    ('helper scalar constexpr', lambda s: s.replace('n_rows, n_cols,', 'n_rows, n_cols: tl.constexpr,', 1)),
    ('conversion wrong dtype', lambda s: s.replace('n_cols.to(tl.float32)', 'n_cols.to(tl.float16)')),
    ('conversion extra keyword', lambda s: s.replace('n_cols.to(tl.float32)', 'n_cols.to(tl.float32, bitcast=True)')),
    ('duplicate conversion', lambda s: s.replace('    n = n_cols.to', '    z = n_cols.to(tl.float32)\n    n = n_cols.to')),
    ('allocation wrong shape', lambda s: s.replace('X.new_empty((n_rows, n_cols))', 'X.new_empty((n_cols, n_rows))')),
    ('allocation clone', lambda s: s.replace('X.new_empty((n_rows, n_cols))', 'X.clone()')),
    ('host module mutation', lambda s: s.replace('    n_rows =', '    triton.jit = 1\n    n_rows =', 1)),
    ('host unknown call', lambda s: s.replace('    n_rows =', '    temp = foo()\n    n_rows =', 1)),
    ('host unknown method', lambda s: s.replace('    n_rows =', '    temp = X.storage()\n    n_rows =', 1)),
    ('host second launch', lambda s: s.replace('    return Y', '    _layernorm_kernel[grid](X, Y)\n    return Y')),
    ('host mutated input', lambda s: s.replace('    n_rows =', '    X = X.clone()\n    n_rows =', 1)),
    ('host control flow', lambda s: s.replace('    n_rows =', '    if True:\n        Z = 1\n    n_rows =', 1)),
    ('nonbare jit', lambda s: s.replace('@triton.jit', '@triton.jit()')),
    ('decorator extra', lambda s: s.replace('@triton.jit', '@other\n@triton.jit')),
    ('import alias', lambda s: s.replace('import triton\n', 'import triton as tt\n')),
    ('import duplicate', lambda s: s.replace('import triton\n', 'import triton\nimport triton\n')),
    ('import from', lambda s: s.replace('import triton\n', 'from triton import jit\n')),
    ('module statement', lambda s: s + '\nvalue = 1\n'),
    ('extra helper', lambda s: s + '\ndef other():\n    return 1\n'),
    ('helper default', lambda s: s.replace('eps,\n', 'eps=1e-5,\n')),
    ('entry default', lambda s: s.replace('def kernel(X):', 'def kernel(X=None):')),
    ('entry annotation', lambda s: s.replace('def kernel(X):', 'def kernel(X: torch.Tensor):')),
    ('unknown helper return', lambda s: s.replace('    tl.store(y_ptrs, y, mask=mask)', '    tl.store(y_ptrs, y, mask=mask)\n    return y')),
]


@pytest.mark.parametrize('name,mutation', MUTATIONS, ids=[item[0] for item in MUTATIONS])
def test_unsupported_programs_remain_unknown_even_with_source_bound_scalar_event(name, mutation):
    text = mutation(source())
    try:
        payload = event(text)
    except (SyntaxError, StopIteration):
        payload = event(source())
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    assert scalar_conversion_sites(text, tree, payload, input_rank=2) is None, name


@pytest.mark.parametrize('receiver', ['x', 'X', 'stride_xc', 'n_rows'])
def test_tensor_pointer_stride_or_other_scalar_to_is_not_column_finding(receiver):
    text = source().replace('n_cols.to(tl.float32)', receiver + '.to(tl.float32)')
    assert diagnose(text, event(text, receiver=receiver)) is None


def test_scalar_alias_is_not_inferred_into_direct_parameter_authority():
    text = source().replace('    n = n_cols.to', '    alias = n_cols\n    n = alias.to')
    assert diagnose(text, event(text, receiver='alias')) is None


@pytest.mark.parametrize('change', [
    lambda s: s.replace('X.dim() != 2', 'X.dim() == 2'),
    lambda s: s.replace('X.dim() != 2', 'X.dim() != 3'),
    lambda s: s.replace('X.contiguous()', 'X.clone()'),
    lambda s: s.replace('else X\n', 'else X.clone()\n'),
    lambda s: s.replace('if BLOCK_COLS < 1:', 'if BLOCK_COLS > 1:'),
    lambda s: s.replace('        BLOCK_COLS = 1', '        BLOCK_COLS = 2'),
    lambda s: s.replace('        BLOCK_COLS = 1', '        X = X.clone()'),
])
def test_rank_guard_and_block_floor_require_exact_identity_and_shape_semantics(change):
    assert diagnose(change(source('guarded_tile'))) is None


def test_tree_source_disagreement_and_cycles_depth_limits_are_unknown():
    text = source()
    tree = ast.parse(text)
    tree.body[-1].name = 'different'
    assert scalar_conversion_sites(text, tree, event(text), input_rank=2) is None
    cyclic = ast.Module(body=[], type_ignores=[])
    cyclic.body = [cyclic]
    assert scalar_conversion_sites(text, cyclic, event(text), input_rank=2) is None


@pytest.mark.parametrize('suffix', ['#' * 32769, '\n' + 'unused = 1\n' * 1100])
def test_source_byte_and_ast_count_limits_are_unknown(suffix):
    text = source() + suffix
    assert diagnose(text) is None


def test_bounded_dependency_steps_reject_long_straight_line_program():
    # Every individual expression is shallow, so this exercises the semantic
    # walk budget instead of relying on parser depth or source byte limits.
    text = source().replace('    n = n_cols.to', ''.join('    z%d = 1\n' % n for n in range(780)) + '    n = n_cols.to')
    assert len(list(ast.walk(ast.parse(text)))) < 4096
    assert diagnose(text) is None
