"""Bounded layout evidence and one straight-line Triton addressing hypothesis.

No candidate import, execution, rewriting, or numerical acceptance happens here.
Unsupported programs and incomplete readbacks deliberately produce no finding.
"""
from __future__ import annotations

import ast
import re

from .run_contract import code_hash, digest


RECOGNIZER = 'single-row-missing-input-column-stride-v1'
GUARDED_RECOGNIZER = 'rank-two-guarded-anchor-missing-input-column-stride-v1'


def project_input_layout(metrics, shape):
    """Recheck complete typed readback before projecting byte/element strides."""
    if metrics.get('input_fidelity_verified') is not True:
        return None
    expected, report = metrics.get('input_fidelity_expected'), metrics.get('input_fidelity')
    if (not isinstance(expected, dict) or set(expected) != {'X'}
            or not isinstance(report, dict) or report.get('representation') != 'torch-storage-v1'):
        return None
    devices = report.get('tensor_devices')
    if (metrics.get('device') != 'cuda' or not isinstance(devices, dict) or set(devices) != {'X'}
            or not isinstance(devices['X'], str) or not re.fullmatch(r'cuda:[0-9]+', devices['X'])):
        return None
    spec = expected['X']
    fields = {'shape', 'dtype', 'strides', 'offset', 'storage', 'writeable',
              'storage_bytes', 'storage_digest', 'value_digest'}
    if not isinstance(spec, dict) or set(spec) != fields:
        return None
    if (type(spec['shape']) is not list or spec['shape'] != shape
            or any(type(n) is not int or n < 1 for n in spec['shape'])
            or spec['dtype'] not in ('<f2', '<f4')
            or type(spec['strides']) is not list or len(spec['strides']) != 2
            or any(type(n) is not int or n < 0 for n in spec['strides'])
            or type(spec['offset']) is not int or spec['offset'] < 0
            or type(spec['storage']) is not int or spec['storage'] != 0
            or spec['writeable'] is not True
            or type(spec['storage_bytes']) is not int
            or not 0 < spec['storage_bytes'] <= 256 * 1024 * 1024
            or any(not isinstance(spec[k], str) or not re.fullmatch('[0-9a-f]{64}', spec[k])
                   for k in ('storage_digest', 'value_digest'))):
        return None
    itemsize = 2 if spec['dtype'] == '<f2' else 4
    if (any(n % itemsize for n in spec['strides']) or spec['offset'] % itemsize
            or spec['offset'] + sum((n - 1) * s for n, s in zip(shape, spec['strides']))
                + itemsize > spec['storage_bytes']):
        return None
    from .input_fidelity import check_readback
    if check_readback(expected, report)['input_fidelity_verified'] is not True:
        return None
    return dict(dtype=spec['dtype'], stride_bytes=spec['strides'][:],
                stride_elements=[n // itemsize for n in spec['strides']],
                offset_bytes=spec['offset'], storage_bytes=spec['storage_bytes'],
                expected_sha256=digest(expected), observed_sha256=digest(report['observed']),
                tensor_device=devices['X'])


def _name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _number(node, number):
    return isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value == number


def _call(node, owner, method):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and _name(node.func.value, owner) and node.func.attr == method)


def _doc(node):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _simple_signature(node):
    a = node.args
    return not (a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults or a.kw_defaults)


def _single_definitions(body, forbidden):
    definitions = {}
    for statement in body:
        if not isinstance(statement, ast.Assign):
            continue
        if len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        targets = target.elts if isinstance(target, ast.Tuple) else [target]
        for item in targets:
            if (not isinstance(item, ast.Name) or item.id in definitions or item.id in forbidden):
                return None
            definitions[item.id] = statement
    return definitions


def _ordered_locals(body, initial):
    """Every runtime local read must follow its unique definition."""
    available = set(initial)
    for statement in body:
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in available
               for n in ast.walk(statement)):
            return False
        if isinstance(statement, ast.Assign):
            available.update(n.id for target in statement.targets for n in ast.walk(target)
                             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    return True


def missing_column_stride_sites(source, tree):
    """Prove the narrow load-address chain, not absence of a stride spelling.

    Only canonical imports, one helper, the original X, empty_like output and a
    single whole-row launch are recognized. Copies, branches, rebinding and
    complex aliases stay outside this recognizer.
    """
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    wanted = {'torch': None, 'triton': None, 'triton.language': 'tl'}
    if (len(imports) != 3 or any(not isinstance(n, ast.Import) or len(n.names) != 1 for n in imports)
            or {n.names[0].name: n.names[0].asname for n in imports} != wanted):
        return None
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 2 or any(not isinstance(n, (ast.Import, ast.FunctionDef)) and not _doc(n) for n in tree.body):
        return None
    entries = [n for n in functions if n.name == 'kernel' and not n.decorator_list]
    helpers = [n for n in functions if len(n.decorator_list) == 1
               and isinstance(n.decorator_list[0], ast.Attribute)
               and _name(n.decorator_list[0].value, 'triton') and n.decorator_list[0].attr == 'jit']
    if len(entries) != 1 or len(helpers) != 1:
        return None
    entry, helper = entries[0], helpers[0]
    if any(n.lineno >= min(entry.lineno, helper.lineno) for n in imports):
        return None
    protected = {'X', 'tl', 'triton', 'torch', 'kernel', helper.name}
    if (helper.name in {'X', 'tl', 'triton', 'torch', 'kernel'} or not _simple_signature(entry)
            or not _simple_signature(helper) or [a.arg for a in entry.args.args] != ['X']
            or entry.args.args[0].annotation is not None or entry.returns is not None or helper.returns is not None):
        return None
    params = [a.arg for a in helper.args.args]
    if len(set(params)) != len(params) or set(params) & {'tl', 'triton', 'torch', 'kernel', helper.name}:
        return None
    if any(a.annotation is not None and not (
            isinstance(a.annotation, ast.Attribute) and _name(a.annotation.value, 'tl')
            and a.annotation.attr == 'constexpr') for a in helper.args.args):
        return None
    # Helper X is allowed as a formal, but every other protected name is not.
    host = _single_definitions(entry.body, protected)
    defs = _single_definitions(helper.body, set(params) | protected)
    if host is None or defs is None:
        return None
    if (not _ordered_locals(entry.body, {'X', 'torch', 'triton', helper.name})
            or not _ordered_locals(helper.body, set(params) | {'tl'})):
        return None
    if (any(not isinstance(n, (ast.Assign, ast.Expr, ast.Assert, ast.Return)) for n in entry.body)
            or any(not isinstance(n, (ast.Assign, ast.Expr)) for n in helper.body)
            or any(isinstance(n, (ast.NamedExpr, ast.Await, ast.Yield, ast.IfExp, ast.BoolOp,
                                   ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                                   ast.GeneratorExp, ast.Starred)) for n in ast.walk(tree))):
        return None
    shapes = [n for n in entry.body if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Tuple) and len(n.targets[0].elts) == 2
              and isinstance(n.value, ast.Attribute) and _name(n.value.value, 'X') and n.value.attr == 'shape']
    allocations = [n for n in entry.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and _call(n.value, 'torch', 'empty_like')
                   and len(n.value.args) == 1 and _name(n.value.args[0], 'X') and not n.value.keywords]
    if len(shapes) != 1 or len(allocations) != 1:
        return None
    shape, allocation = shapes[0], allocations[0]
    rows, columns = [n.id for n in shape.targets[0].elts]
    output = allocation.targets[0].id
    returns = [n for n in entry.body if isinstance(n, ast.Return)]
    if len(returns) != 1 or entry.body[-1] is not returns[0] or not _name(returns[0].value, output):
        return None
    launches = [n for n in ast.walk(entry) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)]
    if len(launches) != 1:
        return None
    launch = launches[0]
    if (not _name(launch.func.value, helper.name) or not isinstance(launch.func.slice, ast.Tuple)
            or len(launch.func.slice.elts) != 1 or not _name(launch.func.slice.elts[0], rows)
            or len(launch.args) > len(params) or any(isinstance(n, ast.Starred) for n in launch.args)):
        return None
    bindings = dict(zip(params, launch.args))
    for kw in launch.keywords:
        if kw.arg not in params or kw.arg in bindings:
            return None
        bindings[kw.arg] = kw.value
    if set(bindings) != set(params) or not shape.lineno < launch.lineno or not allocation.lineno < launch.lineno:
        return None
    if not any(isinstance(n, ast.Expr) and n.value is launch for n in entry.body):
        return None
    allowed_host = {id(launch)}
    for n in ast.walk(entry):
        if not isinstance(n, ast.Call) or id(n) in allowed_host:
            continue
        if (_call(n, 'torch', 'empty_like') and n is allocation.value):
            continue
        if (_call(n, 'triton', 'next_power_of_2') and len(n.args) == 1
                and _name(n.args[0], columns) and not n.keywords):
            continue
        if (isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id in {'X', output} and n.func.attr in {'stride', 'dim'}
                and not n.keywords and ((n.func.attr == 'stride' and len(n.args) == 1 and _number(n.args[0], 0))
                                       or (n.func.attr == 'dim' and not n.args))):
            continue
        return None
    for statement in entry.body:
        if isinstance(statement, ast.Expr) and statement.value is not launch and not _doc(statement):
            return None
        if isinstance(statement, ast.Assert):
            test = statement.test
            if (statement.msg is not None or not isinstance(test, ast.Compare) or len(test.ops) != 1
                    or not isinstance(test.ops[0], ast.Eq) or not _call(test.left, 'X', 'dim')
                    or test.left.args or test.left.keywords or not _number(test.comparators[0], 2)):
                return None
    loads = [n for n in ast.walk(helper) if _call(n, 'tl', 'load')]
    stores = [n for n in ast.walk(helper) if _call(n, 'tl', 'store')]
    if len(loads) != 1 or len(stores) != 1:
        return None
    load, store = loads[0], stores[0]
    load_defs = [n for n in helper.body if isinstance(n, ast.Assign) and load in list(ast.walk(n.value))]
    if len(load_defs) != 1 or not isinstance(load_defs[0].targets[0], ast.Name):
        return None
    load_def = load_defs[0]
    value = load_def.value
    if value is not load and not (
            isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == 'to' and value.func.value is load and len(value.args) == 1
            and isinstance(value.args[0], ast.Attribute) and _name(value.args[0].value, 'tl')
            and value.args[0].attr == 'float32' and not value.keywords):
        return None
    if any(not isinstance(n.targets[0], ast.Name) for n in helper.body if isinstance(n, ast.Assign)):
        return None
    for n in ast.walk(helper):
        if isinstance(n, ast.Call):
            if (isinstance(n.func, ast.Attribute) and _name(n.func.value, 'tl')
                    and n.func.attr in {'program_id', 'arange', 'load', 'sum', 'where', 'sqrt', 'store'}):
                continue
            if isinstance(n.func, ast.Attribute) and n.func.attr == 'to' and len(n.args) == 1 and not n.keywords:
                continue
            return None
    for statement in helper.body:
        if isinstance(statement, ast.Expr) and statement.value is not store and not _doc(statement):
            return None

    # Resolve a small positive address grammar with single definitions and
    # before-use ordering. Unknown arithmetic or runtime rebinding is rejected.
    sites = dict(input_shape=shape, output_allocation=allocation, helper_launch=launch)
    address_steps = 0
    def resolve(node, before, depth=0):
        nonlocal address_steps
        address_steps += 1
        if depth > 32 or address_steps > 1024:
            raise ValueError('address_depth')
        if isinstance(node, ast.Name):
            if node.id in bindings:
                bound = bindings[node.id]
                if _name(bound, 'X'):
                    return ('input',)
                if _name(bound, output):
                    return ('output',)
                if _name(bound, columns):
                    return ('length',)
                if isinstance(bound, ast.Name) and bound.id in host:
                    definition = host[bound.id]
                    if definition.lineno >= launch.lineno:
                        raise ValueError('host_definition_order')
                    bound = definition.value
                if (_call(bound, 'X', 'stride') and len(bound.args) == 1 and _number(bound.args[0], 0)
                        and not bound.keywords):
                    sites['input_row_stride'] = bound
                    return ('input_stride',)
                if (_call(bound, output, 'stride') and len(bound.args) == 1 and _number(bound.args[0], 0)
                        and not bound.keywords):
                    return ('output_stride',)
                if (_call(bound, 'triton', 'next_power_of_2') and len(bound.args) == 1
                        and _name(bound.args[0], columns) and not bound.keywords):
                    if bound.lineno <= shape.lineno:
                        raise ValueError('shape_definition_order')
                    return ('block',)
                raise ValueError('unsupported_launch_binding')
            definition = defs.get(node.id)
            if definition is None or definition.lineno >= before:
                raise ValueError('helper_definition_order')
            return resolve(definition.value, definition.lineno, depth + 1)
        if _call(node, 'tl', 'program_id') and len(node.args) == 1 and _number(node.args[0], 0) and not node.keywords:
            sites['row_index'] = node
            return ('row',)
        if (_call(node, 'tl', 'arange') and len(node.args) == 2 and _number(node.args[0], 0)
                and not node.keywords and resolve(node.args[1], before, depth + 1) == ('block',)):
            sites['column_index'] = node
            return ('column',)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            left, right = resolve(node.left, before, depth + 1), resolve(node.right, before, depth + 1)
            if isinstance(node.op, ast.Mult):
                return ('multiply', *sorted((left, right)))
            terms = []
            for value in (left, right):
                terms.extend(value[1:] if value[0] == 'add' else [value])
            return ('add', *sorted(terms))
        raise ValueError('unsupported_address')
    try:
        if len(load.args) != 1 or len(store.args) != 2:
            return None
        actual = resolve(load.args[0], load.lineno)
        expected = ('add', *sorted([('input',), ('column',), ('multiply', *sorted([('row',), ('input_stride',)]))]))
        if actual != expected:
            return None
        output_pointer = resolve(store.args[0], store.lineno)
        expected_output = ('add', *sorted([('output',), ('column',), ('multiply', *sorted([('row',), ('output_stride',)]))]))
        if output_pointer != expected_output:
            return None
        keywords = {k.arg: k.value for k in load.keywords}
        if len(keywords) != len(load.keywords) or set(keywords) != {'mask', 'other'} or not _number(keywords['other'], 0):
            return None
        mask = keywords['mask']
        if not isinstance(mask, ast.Name) or mask.id not in defs:
            return None
        comparison = defs[mask.id]
        test = comparison.value
        if (comparison.lineno >= load.lineno or not isinstance(test, ast.Compare) or len(test.ops) != 1
                or not isinstance(test.ops[0], ast.Lt) or resolve(test.left, comparison.lineno) != ('column',)
                or resolve(test.comparators[0], comparison.lineno) != ('length',)):
            return None
        if len(store.keywords) != 1 or store.keywords[0].arg != 'mask' or not _name(store.keywords[0].value, mask.id):
            return None
    except (ValueError, TypeError, RecursionError):
        return None
    load_name = load_def.targets[0].id
    dependency_steps = 0
    def depends(node, before, depth=0):
        nonlocal dependency_steps
        dependency_steps += 1
        if depth > 32 or dependency_steps > 1024:
            return False
        if isinstance(node, ast.Name):
            if node.id == load_name:
                return load_def.lineno < before
            definition = defs.get(node.id)
            return bool(definition and definition.lineno < before and depends(definition.value, definition.lineno, depth + 1))
        return any(depends(child, before, depth + 1) for child in ast.iter_child_nodes(node))
    if not depends(store.args[1], store.lineno):
        return None
    sites.update(input_load=load_def, output_store=store)
    containers = {id(child): statement for statement in entry.body + helper.body
                  for child in ast.walk(statement)}
    result = []
    for role, node in sites.items():
        node = containers.get(id(node), node)
        excerpt = ast.get_source_segment(source, node)
        if excerpt is None or len(excerpt.encode()) > 512:
            return None
        result.append(dict(role=role, line=node.lineno, end_line=node.end_lineno,
                           column=node.col_offset, end_column=node.end_col_offset,
                           excerpt=excerpt, excerpt_sha256=code_hash(excerpt), trust='untrusted_source_data'))
    return result


def guarded_column_stride_sites(source, tree, *, input_rank):
    """Recognize one rank-proven identity guard and two input read addresses.

    The supplied rank is trusted input evidence, not a fact inferred from source.
    For rank two the canonical guard's copying branch is unreachable. All other
    branches, copies, mutations and dynamic calls remain unsupported. This is
    only a source-address hypothesis; arithmetic and GPU semantics are not proved.
    """
    if type(input_rank) is not int or input_rank != 2:
        return None
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if (len(imports) != 3
            or any(not isinstance(n, ast.Import) or len(n.names) != 1 for n in imports)
            or {n.names[0].name: n.names[0].asname for n in imports}
                != {'torch': None, 'triton': None, 'triton.language': 'tl'}):
        return None
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if (len(functions) != 2
            or any(not isinstance(n, (ast.Import, ast.FunctionDef)) and not _doc(n) for n in tree.body)):
        return None
    entries = [n for n in functions if n.name == 'kernel' and not n.decorator_list]
    helpers = [n for n in functions if len(n.decorator_list) == 1
               and isinstance(n.decorator_list[0], ast.Attribute)
               and _name(n.decorator_list[0].value, 'triton')
               and n.decorator_list[0].attr == 'jit']
    if len(entries) != 1 or len(helpers) != 1:
        return None
    entry, helper = entries[0], helpers[0]
    protected = {'X', 'tl', 'triton', 'torch', 'kernel', helper.name}
    if (helper.name in {'X', 'tl', 'triton', 'torch', 'kernel'}
            or any(n.lineno >= min(entry.lineno, helper.lineno) for n in imports)
            or not _simple_signature(entry) or not _simple_signature(helper)
            or [a.arg for a in entry.args.args] != ['X']
            or entry.args.args[0].annotation is not None
            or entry.returns is not None or helper.returns is not None):
        return None
    params = [a.arg for a in helper.args.args]
    if (len(params) != 7 or len(set(params)) != len(params)
            or set(params) & {'tl', 'triton', 'torch', 'kernel', helper.name}
            or any(a.annotation is not None and not (
                isinstance(a.annotation, ast.Attribute) and _name(a.annotation.value, 'tl')
                and a.annotation.attr == 'constexpr') for a in helper.args.args)):
        return None
    body = [n for n in entry.body if not _doc(n)]
    if len(body) not in (7, 8):
        return None

    def assignment(statement):
        return (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name))

    # Check the complete wrapper before allowing its single X rebinding.
    guard, row_shape, col_shape, allocation, block = body[:5]
    if not assignment(guard) or not _name(guard.targets[0], 'X') or not isinstance(guard.value, ast.IfExp):
        return None
    condition = guard.value.test
    if (not isinstance(condition, ast.Compare) or len(condition.ops) != 1
            or not isinstance(condition.ops[0], ast.NotEq)
            or not _call(condition.left, 'X', 'dim') or condition.left.args or condition.left.keywords
            or not _number(condition.comparators[0], 2)
            or not _call(guard.value.body, 'X', 'contiguous')
            or guard.value.body.args or guard.value.body.keywords
            or not _name(guard.value.orelse, 'X')):
        return None
    def shape_index(statement, index):
        return (assignment(statement) and isinstance(statement.value, ast.Subscript)
                and isinstance(statement.value.value, ast.Attribute)
                and _name(statement.value.value.value, 'X') and statement.value.value.attr == 'shape'
                and _number(statement.value.slice, index))
    if not shape_index(row_shape, 0) or not shape_index(col_shape, 1) or not assignment(allocation):
        return None
    rows, columns, output = (row_shape.targets[0].id, col_shape.targets[0].id, allocation.targets[0].id)
    if (not assignment(block) or len({rows, columns, output, block.targets[0].id}) != 4
            or {rows, columns, output, block.targets[0].id} & protected):
        return None
    block_name = block.targets[0].id
    value = allocation.value
    if (not _call(value, 'torch', 'empty') or len(value.args) != 1
            or not isinstance(value.args[0], ast.Tuple) or len(value.args[0].elts) != 2
            or not _name(value.args[0].elts[0], rows) or not _name(value.args[0].elts[1], columns)
            or len(value.keywords) != 2
            or {kw.arg for kw in value.keywords} != {'dtype', 'device'}
            or any(not isinstance(kw.value, ast.Attribute) or not _name(kw.value.value, 'X')
                   or kw.value.attr != kw.arg for kw in value.keywords)):
        return None
    if (not _call(block.value, 'triton', 'next_power_of_2') or len(block.value.args) != 1
            or not _name(block.value.args[0], columns) or block.value.keywords):
        return None
    if len(body) == 8:
        lower_bound = body[5]
        if (not isinstance(lower_bound, ast.If) or lower_bound.orelse
                or len(lower_bound.body) != 1 or not assignment(lower_bound.body[0])
                or not _name(lower_bound.body[0].targets[0], block_name)
                or not _number(lower_bound.body[0].value, 1)):
            return None
        test = lower_bound.test
        if (not isinstance(test, ast.Compare) or len(test.ops) != 1
                or not isinstance(test.ops[0], ast.Lt) or not _name(test.left, block_name)
                or not _number(test.comparators[0], 1)):
            return None
    launch_statement, returned = body[-2:]
    if (not isinstance(launch_statement, ast.Expr)
            or not isinstance(launch_statement.value, ast.Call)
            or not isinstance(launch_statement.value.func, ast.Subscript)
            or not _name(launch_statement.value.func.value, helper.name)
            or not isinstance(returned, ast.Return) or not _name(returned.value, output)):
        return None
    launch = launch_statement.value
    if (not isinstance(launch.func.slice, ast.Tuple) or len(launch.func.slice.elts) != 1
            or not _name(launch.func.slice.elts[0], rows) or len(launch.args) > len(params)
            or any(isinstance(n, ast.Starred) for n in launch.args)):
        return None
    bindings = dict(zip(params, launch.args))
    for kw in launch.keywords:
        if kw.arg not in params or kw.arg in bindings:
            return None
        bindings[kw.arg] = kw.value
    if set(bindings) != set(params):
        return None

    sites = dict(input_rank_guard=guard, input_rows=row_shape, input_columns=col_shape,
                 output_allocation=allocation, helper_launch=launch)
    def host_role(node):
        if isinstance(node, ast.Name):
            return {'X': ('input',), output: ('output',), columns: ('length',), block_name: ('block',)}.get(node.id)
        for tensor, role in [('X', 'input_stride'), (output, 'output_stride')]:
            if (_call(node, tensor, 'stride') and len(node.args) == 1
                    and _number(node.args[0], 0) and not node.keywords):
                if tensor == 'X':
                    sites['input_row_stride'] = node
                return (role,)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value == 1e-5:
            return ('epsilon',)
        return None
    roles = {name: host_role(node) for name, node in bindings.items()}
    if (any(value is None for value in roles.values())
            or set(roles.values()) != {(n,) for n in ('input', 'output', 'length', 'block',
                                                     'input_stride', 'output_stride', 'epsilon')}):
        return None
    if not _ordered_locals(helper.body, set(params) | {'tl'}):
        return None
    if (any(not isinstance(n, (ast.Assign, ast.Expr)) for n in helper.body)
            or any(isinstance(n, (ast.NamedExpr, ast.Await, ast.Yield, ast.IfExp, ast.BoolOp,
                                  ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                                  ast.GeneratorExp, ast.Starred)) for n in ast.walk(helper))):
        return None
    definitions = {}
    self_casts = []
    for statement in helper.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not assignment(statement) or statement.targets[0].id in set(params) | protected:
            return None
        name = statement.targets[0].id
        if name in definitions:
            cast = statement.value
            if (len(definitions[name]) != 1 or not _call(cast, 'tl', 'cast')
                    or len(cast.args) != 2 or not _name(cast.args[0], name) or cast.keywords
                    or not isinstance(cast.args[1], ast.Attribute)
                    or not _name(cast.args[1].value, 'tl') or cast.args[1].attr != 'float32'):
                return None
            self_casts.append(name)
        definitions.setdefault(name, []).append(statement)
    loads = [n for n in ast.walk(helper) if _call(n, 'tl', 'load')]
    stores = [n for n in ast.walk(helper) if _call(n, 'tl', 'store')]
    if len(loads) != 2 or len(stores) != 1:
        return None
    store = stores[0]
    allowed_functions = {'program_id', 'arange', 'load', 'sum', 'where', 'sqrt', 'cast', 'store'}
    for n in ast.walk(helper):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Attribute) and _name(n.func.value, 'tl') and n.func.attr in allowed_functions:
            if n.func.attr == 'cast' and (len(n.args) != 2 or n.keywords
                    or not isinstance(n.args[1], ast.Attribute) or not _name(n.args[1].value, 'tl')
                    or n.args[1].attr != 'float32'):
                return None
            continue
        if (isinstance(n.func, ast.Attribute) and n.func.attr == 'to' and len(n.args) == 1
                and not n.keywords and isinstance(n.args[0], ast.Attribute)
                and _name(n.args[0].value, 'tl') and n.args[0].attr == 'float32'):
            continue
        return None
    if any(isinstance(n, ast.Expr) and n.value is not store and not _doc(n) for n in helper.body):
        return None
    steps = 0
    def definition(name, before):
        previous = [n for n in definitions.get(name, []) if n.lineno < before]
        if not previous:
            raise ValueError('definition_order')
        return previous[-1]
    def resolve(node, before, depth=0):
        nonlocal steps
        steps += 1
        if depth > 32 or steps > 1024:
            raise ValueError('address_depth')
        if isinstance(node, ast.Name):
            if node.id in roles:
                return roles[node.id]
            previous = definition(node.id, before)
            return resolve(previous.value, previous.lineno, depth + 1)
        if _call(node, 'tl', 'program_id') and len(node.args) == 1 and _number(node.args[0], 0) and not node.keywords:
            sites['row_index'] = node
            return ('row',)
        if (_call(node, 'tl', 'arange') and len(node.args) == 2 and _number(node.args[0], 0)
                and not node.keywords and resolve(node.args[1], before, depth + 1) == ('block',)):
            sites['column_index'] = node
            return ('column',)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            left, right = resolve(node.left, before, depth + 1), resolve(node.right, before, depth + 1)
            if isinstance(node.op, ast.Mult):
                return ('multiply', *sorted((left, right)))
            terms = []
            for value in (left, right):
                terms.extend(value[1:] if value[0] == 'add' else [value])
            return ('add', *sorted(terms))
        raise ValueError('unsupported_address')
    def load_assignment(load):
        found = [n for values in definitions.values() for n in values if load in list(ast.walk(n.value))]
        if len(found) != 1:
            raise ValueError('load_definition')
        statement = found[0]
        value = statement.value
        if value is not load and not (
                isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == 'to' and value.func.value is load
                and len(value.args) == 1 and not value.keywords
                and isinstance(value.args[0], ast.Attribute) and _name(value.args[0].value, 'tl')
                and value.args[0].attr == 'float32') and not (
                _call(value, 'tl', 'cast') and len(value.args) == 2 and value.args[0] is load
                and not value.keywords and isinstance(value.args[1], ast.Attribute)
                and _name(value.args[1].value, 'tl') and value.args[1].attr == 'float32'):
            raise ValueError('load_wrapper')
        return statement
    row_offset = ('multiply', *sorted([('row',), ('input_stride',)]))
    anchor_address = ('add', *sorted([('input',), row_offset]))
    main_address = ('add', *sorted([('input',), row_offset, ('column',)]))
    output_address = ('add', *sorted([('output',), ('column',),
                         ('multiply', *sorted([('row',), ('output_stride',)]))]))
    try:
        addressed = []
        for load in loads:
            if len(load.args) != 1:
                return None
            addressed.append((resolve(load.args[0], load.lineno), load, load_assignment(load)))
        mains = [n for n in addressed if n[0] == main_address]
        anchors = [n for n in addressed if n[0] == anchor_address]
        if len(mains) != 1 or len(anchors) != 1:
            return None
        _, main_load, main_statement = mains[0]
        _, anchor_load, anchor_statement = anchors[0]
        if anchor_load.keywords or any(name != anchor_statement.targets[0].id for name in self_casts):
            return None
        keywords = {k.arg: k.value for k in main_load.keywords}
        if (len(main_load.keywords) != 2 or set(keywords) != {'mask', 'other'}
                or not _number(keywords['other'], 0) or not isinstance(keywords['mask'], ast.Name)):
            return None
        mask_name = keywords['mask'].id
        mask_definition = definition(mask_name, main_load.lineno)
        comparison = mask_definition.value
        if (not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1
                or not isinstance(comparison.ops[0], ast.Lt)
                or resolve(comparison.left, mask_definition.lineno) != ('column',)
                or resolve(comparison.comparators[0], mask_definition.lineno) != ('length',)):
            return None
        if (len(store.args) != 2 or resolve(store.args[0], store.lineno) != output_address
                or len(store.keywords) != 1 or store.keywords[0].arg != 'mask'
                or not _name(store.keywords[0].value, mask_name)):
            return None
    except (ValueError, TypeError, RecursionError):
        return None
    dependency_steps = 0
    def depends(node, before, target, depth=0):
        nonlocal dependency_steps
        dependency_steps += 1
        if depth > 32 or dependency_steps > 1024:
            raise ValueError('dependency_depth')
        if isinstance(node, ast.Name):
            if node.id not in definitions:
                return False
            previous = definition(node.id, before)
            return previous is target or depends(previous.value, previous.lineno, target, depth + 1)
        return any(depends(child, before, target, depth + 1) for child in ast.iter_child_nodes(node))
    try:
        if not all(depends(store.args[1], store.lineno, target) for target in (main_statement, anchor_statement)):
            return None
    except (ValueError, RecursionError):
        return None
    sites.update(input_load=main_statement, input_row_anchor=anchor_statement, output_store=store)
    containers = {id(child): statement for statement in entry.body + helper.body
                  for child in ast.walk(statement)}
    result = []
    for role, node in sites.items():
        node = containers.get(id(node), node)
        excerpt = ast.get_source_segment(source, node)
        if excerpt is None or len(excerpt.encode()) > 512:
            return None
        result.append(dict(role=role, line=node.lineno, end_line=node.end_lineno,
                           column=node.col_offset, end_column=node.end_col_offset,
                           excerpt=excerpt, excerpt_sha256=code_hash(excerpt), trust='untrusted_source_data'))
    return result


def stride_finding(sites, observations):
    first = observations[0]['input_layout']
    itemsize = 2 if first['dtype'] == '<f2' else 4
    return dict(kind='missing_input_column_stride', status='supported_hypothesis',
        source_sites=sites, affected_observations=observations[:4],
        relation='source column offset = column; declared column offset = column * input_stride_elements[1]',
        evidence_scope='single_launch_straight_line_AST_address_chain_not_full_program_or_GPU_semantics',
        observation_scope='verified_nonunit_column_stride_and_failed_numerics_do_not_prove_error_causality',
        witness=dict(scope='trusted_scalar_address_arithmetic_not_candidate_execution',
            logical_column=1, source_relative_elements=1,
            declared_relative_elements=first['stride_elements'][1], source_relative_bytes=itemsize,
            declared_relative_bytes=first['stride_bytes'][1],
            added_to_correctness_suite=False, used_as_acceptance_credential=False),
        repair_obligations=[
            'Derive the input column stride from the original runtime tensor and carry it into the JIT helper.',
            'Use column * input_column_stride in the load pointer; derive output strides independently.',
            'Preserve wrapper inputs, reference, tolerances, masks and the complete correctness suite; reverify the new source.'])
