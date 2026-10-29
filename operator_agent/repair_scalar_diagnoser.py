"""A bounded AST hypothesis for a dimension scalar's tensor-only conversion.

The compiler event is typed and bound to the original source. This module does
not import, execute, modify or accept a candidate; callers separately establish
the trusted single-column test observation.
"""
from __future__ import annotations

import ast
import math

from .compile_evidence import validate_compile_failure
from .run_contract import code_hash


RECOGNIZER = 'bound-single-column-scalar-method-v1'


def _name(node, value):
    return isinstance(node, ast.Name) and node.id == value


def _number(node, value=None):
    return (isinstance(node, ast.Constant) and type(node.value) in (int, float)
            and (type(node.value) is int or math.isfinite(node.value))
            and (value is None or node.value == value))


def _call(node, owner, method):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and _name(node.func.value, owner) and node.func.attr == method)


def _doc(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _signature(node):
    args = node.args
    return not (args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
                or args.defaults or args.kw_defaults or node.returns)


def _same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _parse(source, supplied):
    if type(source) is not str or len(source.encode('utf-8')) > 32768:
        return None
    try:
        tree = ast.parse(source)
        if not isinstance(supplied, ast.Module):
            return None
        for item in (tree, supplied):
            pending, count = [(item, 0)], 0
            while pending:
                node, depth = pending.pop()
                count += 1
                if count > 4096 or depth > 32:
                    return None
                pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        return tree if isinstance(supplied, ast.Module) and _same(tree, supplied) else None
    except (SyntaxError, ValueError, TypeError, RecursionError, UnicodeError):
        return None


def _guard(node):
    """This conditional is the identity only with the caller's rank-2 proof."""
    if not isinstance(node, ast.IfExp):
        return False
    test = node.test
    return (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.NotEq) and len(test.comparators) == 1
            and _call(test.left, 'X', 'dim') and not test.left.args and not test.left.keywords
            and _number(test.comparators[0], 2) and _call(node.body, 'X', 'contiguous')
            and not node.body.args and not node.body.keywords and _name(node.orelse, 'X'))


class _PatternError(ValueError):
    pass


class _Matcher:
    def __init__(self, source, entry, helper):
        self.source, self.entry, self.helper = source, entry, helper
        self.steps = 0
        self.env = {'X': ('input',)}
        self.shape_sites = {}
        self.output = None
        self.launch = None
        self.guard_seen = False

    def step(self, depth):
        self.steps += 1
        if depth > 32 or self.steps > 1024:
            raise _PatternError('bound')

    def host(self, node, depth=0):
        self.step(depth)
        if isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id]
        if _number(node):
            return ('number', node.value)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                and _name(node.value.value, 'X') and node.value.attr == 'shape'
                and _number(node.slice) and type(node.slice.value) is int and node.slice.value in (0, 1)):
            return ('rows',) if node.slice.value == 0 else ('columns',)
        if (isinstance(node, ast.Tuple) and len(node.elts) in (1, 2)):
            return ('tuple', tuple(self.host(n, depth + 1) for n in node.elts))
        if isinstance(node, ast.Call) and not node.keywords:
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ('X', self.output) and node.func.attr == 'stride'):
                owner = 'input' if node.func.value.id == 'X' else 'output'
                if not node.args:
                    return ('tuple', (('stride', owner, 0), ('stride', owner, 1)))
                if (len(node.args) == 1 and _number(node.args[0])
                        and type(node.args[0].value) is int and node.args[0].value in (0, 1)):
                    return ('stride', owner, node.args[0].value)
            if _call(node, 'triton', 'next_power_of_2') and len(node.args) == 1:
                if self.host(node.args[0], depth + 1) == ('columns',):
                    return ('block_columns',)
            if _call(node, 'triton', 'cdiv') and len(node.args) == 2:
                left, right = [self.host(n, depth + 1) for n in node.args]
                if left == ('rows',) and right[0] == 'number' and type(right[1]) is int and right[1] > 0:
                    return ('grid_rows', right[1])
        raise _PatternError('host expression')

    def allocation(self, node):
        if _call(node, 'torch', 'empty_like') and len(node.args) == 1 and _name(node.args[0], 'X') and not node.keywords:
            return True
        if not isinstance(node, ast.Call) or len(node.args) != 1:
            return False
        if _call(node, 'X', 'new_empty'):
            if node.keywords:
                return False
        elif _call(node, 'torch', 'empty'):
            kws = {kw.arg: kw.value for kw in node.keywords}
            if len(kws) != len(node.keywords) or set(kws) != {'dtype', 'device'}:
                return False
            if any(not isinstance(kws[key], ast.Attribute) or not _name(kws[key].value, 'X')
                   or kws[key].attr != key for key in kws):
                return False
        else:
            return False
        return self.host(node.args[0]) == ('tuple', (('rows',), ('columns',)))

    def wrapper(self):
        reserved = {'X', 'torch', 'triton', 'tl', 'kernel', self.helper.name}
        body = [n for n in self.entry.body if not _doc(n)]
        if not body or not isinstance(body[-1], ast.Return):
            raise _PatternError('return')
        for index, statement in enumerate(body):
            self.step(0)
            if isinstance(statement, ast.Assign):
                if self.launch is not None or len(statement.targets) != 1:
                    raise _PatternError('assignment')
                target = statement.targets[0]
                if _name(target, 'X'):
                    if index != 0 or self.guard_seen or not _guard(statement.value):
                        raise _PatternError('input reassignment')
                    self.guard_seen = True
                    continue
                targets = target.elts if isinstance(target, ast.Tuple) else [target]
                if any(not isinstance(n, ast.Name) or n.id in self.env or n.id in reserved for n in targets):
                    raise _PatternError('host rebinding')
                if len({n.id for n in targets}) != len(targets):
                    raise _PatternError('tuple rebinding')
                value = statement.value
                if isinstance(value, ast.Attribute) and _name(value.value, 'X') and value.attr == 'shape':
                    semantic = ('tuple', (('rows',), ('columns',)))
                elif isinstance(target, ast.Name) and isinstance(value, ast.Call) and (
                        _call(value, 'torch', 'empty_like') or _call(value, 'torch', 'empty')
                        or _call(value, 'X', 'new_empty')):
                    if self.output is not None or not self.allocation(value):
                        raise _PatternError('allocation')
                    self.output = target.id
                    semantic = ('output',)
                else:
                    semantic = self.host(value)
                if isinstance(target, ast.Tuple):
                    if semantic[0] != 'tuple' or len(semantic[1]) != len(targets):
                        raise _PatternError('tuple assignment')
                    values = semantic[1]
                else:
                    values = (semantic,)
                for name, item in zip(targets, values):
                    # Dimension aliases are deliberately unsupported. Their
                    # original definition must be an input shape read.
                    if item[0] in ('rows', 'columns'):
                        direct = (isinstance(value, ast.Attribute) and _name(value.value, 'X')
                                  and value.attr == 'shape') or isinstance(value, ast.Subscript)
                        if not direct or item[0] in self.shape_sites:
                            raise _PatternError('shape alias')
                        self.shape_sites[item[0]] = statement
                    self.env[name.id] = item
            elif isinstance(statement, ast.If):
                test = statement.test
                if (self.launch is not None or statement.orelse or len(statement.body) != 1
                        or not isinstance(test, ast.Compare) or len(test.ops) != 1
                        or not isinstance(test.ops[0], ast.Lt) or len(test.comparators) != 1
                        or not isinstance(test.left, ast.Name) or self.env.get(test.left.id) != ('block_columns',)
                        or not _number(test.comparators[0]) or type(test.comparators[0].value) is not int
                        or test.comparators[0].value < 1):
                    raise _PatternError('host control flow')
                update = statement.body[0]
                if (not isinstance(update, ast.Assign) or len(update.targets) != 1
                        or not _name(update.targets[0], test.left.id)
                        or not _same(update.value, test.comparators[0])):
                    raise _PatternError('block floor')
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                launch = statement.value
                if (self.launch is not None or not isinstance(launch.func, ast.Subscript)
                        or not _name(launch.func.value, self.helper.name)):
                    raise _PatternError('launch')
                grid = self.host(launch.func.slice)
                if (grid[0] != 'tuple' or len(grid[1]) != 1
                        or grid[1][0][0] not in ('rows', 'grid_rows')):
                    raise _PatternError('grid')
                self.launch = launch
            elif isinstance(statement, ast.Return):
                if index != len(body) - 1 or self.launch is None or not _name(statement.value, self.output):
                    raise _PatternError('return')
            else:
                raise _PatternError('wrapper statement')
        if self.output is None or set(self.shape_sites) != {'rows', 'columns'}:
            raise _PatternError('shape and output')
        params = [arg.arg for arg in self.helper.args.args]
        if len(self.launch.args) > len(params):
            raise _PatternError('argument count')
        bindings = dict(zip(params, self.launch.args))
        for kw in self.launch.keywords:
            if kw.arg not in params or kw.arg in bindings:
                raise _PatternError('keyword binding')
            bindings[kw.arg] = kw.value
        if set(bindings) != set(params):
            raise _PatternError('missing binding')
        values = {name: self.host(node) for name, node in bindings.items()}
        if (sum(value == ('input',) for value in values.values()) != 1
                or sum(value == ('output',) for value in values.values()) != 1
                or sum(value == ('columns',) for value in values.values()) != 1):
            raise _PatternError('parameter authority')
        return bindings, values

    def helper_body(self, bindings, values):
        params = {arg.arg: arg for arg in self.helper.args.args}
        scalar = next(name for name, value in values.items() if value == ('columns',))
        if params[scalar].annotation is not None:
            raise _PatternError('annotated dimension')
        env = {name: ('pointer' if value[0] in ('input', 'output') else 'scalar')
               for name, value in values.items()}
        pointer_names = {name for name, kind in env.items() if kind == 'pointer'}
        conversions, loads, stores = [], [], []

        def dtype(node):
            return (isinstance(node, ast.Attribute) and _name(node.value, 'tl')
                    and node.attr in ('float16', 'float32')) or (
                        isinstance(node, ast.Attribute) and node.attr == 'element_ty'
                        and isinstance(node.value, ast.Attribute) and node.value.attr == 'dtype'
                        and isinstance(node.value.value, ast.Name) and node.value.value.id in pointer_names)

        def expression(node, depth=0):
            self.step(depth)
            if isinstance(node, ast.Name) and node.id in env:
                return env[node.id]
            if _number(node):
                return 'scalar'
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                kind = expression(node.operand, depth + 1)
                if kind in ('scalar', 'tensor'):
                    return kind
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.BitAnd, ast.BitOr)):
                left, right = expression(node.left, depth + 1), expression(node.right, depth + 1)
                if 'pointer' in (left, right):
                    if (isinstance(node.op, ast.Add) and (left, right).count('pointer') == 1):
                        return 'pointer'
                else:
                    return 'tensor' if 'tensor' in (left, right) else 'scalar'
            if (isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1
                    and isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq))):
                kinds = [expression(n, depth + 1) for n in (node.left, node.comparators[0])]
                if 'pointer' not in kinds:
                    return 'tensor' if 'tensor' in kinds else 'scalar'
            if isinstance(node, ast.Subscript) and expression(node.value, depth + 1) == 'tensor':
                slices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                if (1 <= len(slices) <= 2 and all((isinstance(n, ast.Constant) and n.value is None)
                        or (isinstance(n, ast.Slice) and n.lower is None and n.upper is None and n.step is None)
                        for n in slices)):
                    return 'tensor'
            if not isinstance(node, ast.Call) or any(kw.arg is None for kw in node.keywords):
                raise _PatternError('helper expression')
            kws = {kw.arg: kw.value for kw in node.keywords}
            if len(kws) != len(node.keywords):
                raise _PatternError('duplicate keyword')
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'to':
                if len(node.args) != 1 or kws or not dtype(node.args[0]):
                    raise _PatternError('conversion')
                receiver = expression(node.func.value, depth + 1)
                if receiver == 'tensor':
                    return 'tensor'
                if (_name(node.func.value, scalar) and isinstance(node.args[0], ast.Attribute)
                        and _name(node.args[0].value, 'tl') and node.args[0].attr == 'float32'):
                    conversions.append(node)
                    return 'scalar'
                raise _PatternError('other scalar conversion')
            if not isinstance(node.func, ast.Attribute) or not _name(node.func.value, 'tl'):
                raise _PatternError('helper call authority')
            method = node.func.attr
            if method == 'program_id' and len(node.args) == 1 and not kws and _number(node.args[0], 0):
                return 'tensor'
            if method == 'arange' and len(node.args) == 2 and not kws and _number(node.args[0], 0):
                if (isinstance(node.args[1], ast.Name) and node.args[1].id in params
                        and isinstance(params[node.args[1].id].annotation, ast.Attribute)
                        and _name(params[node.args[1].id].annotation.value, 'tl')
                        and params[node.args[1].id].annotation.attr == 'constexpr'):
                    return 'tensor'
            if method == 'load' and len(node.args) == 1 and set(kws) <= {'mask', 'other'}:
                if expression(node.args[0], depth + 1) == 'pointer' and (
                        'mask' not in kws or expression(kws['mask'], depth + 1) == 'tensor') and (
                        'other' not in kws or _number(kws['other'])):
                    loads.append(node)
                    return 'tensor'
            if method == 'store' and len(node.args) == 2 and set(kws) <= {'mask'}:
                if (expression(node.args[0], depth + 1) == 'pointer'
                        and expression(node.args[1], depth + 1) in ('scalar', 'tensor')
                        and ('mask' not in kws or expression(kws['mask'], depth + 1) == 'tensor')):
                    stores.append(node)
                    return 'store'
            if method == 'sum' and len(node.args) == 1 and set(kws) == {'axis'}:
                if expression(node.args[0], depth + 1) == 'tensor' and (
                        _number(kws['axis'], 0) or _number(kws['axis'], 1)):
                    return 'tensor'
            if method == 'sqrt' and len(node.args) == 1 and not kws:
                if expression(node.args[0], depth + 1) in ('scalar', 'tensor'):
                    return 'tensor'
            if method == 'where' and len(node.args) == 3 and not kws:
                if (expression(node.args[0], depth + 1) == 'tensor'
                        and all(expression(n, depth + 1) in ('scalar', 'tensor') for n in node.args[1:])):
                    return 'tensor'
            if method == 'cast' and len(node.args) == 2 and not kws and dtype(node.args[1]):
                kind = expression(node.args[0], depth + 1)
                if kind in ('scalar', 'tensor'):
                    return kind
            raise _PatternError('unsupported helper call')

        reserved = {'X', 'torch', 'triton', 'tl', 'kernel', self.helper.name}
        for statement in self.helper.body:
            self.step(0)
            if _doc(statement):
                continue
            if isinstance(statement, ast.Assign):
                if (len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name)
                        or statement.targets[0].id in set(params) | reserved):
                    raise _PatternError('helper parameter rebinding')
                kind = expression(statement.value)
                if kind == 'store':
                    raise _PatternError('store assignment')
                env[statement.targets[0].id] = kind
            elif isinstance(statement, ast.Expr):
                if expression(statement.value) != 'store':
                    raise _PatternError('helper expression statement')
            else:
                raise _PatternError('helper control flow')
        if len(conversions) != 1 or not loads or len(stores) != 1:
            raise _PatternError('conversion or memory operations')
        return scalar, params[scalar], conversions[0]


def scalar_conversion_sites(source, tree, compile_failure, *, input_rank):
    """Bind one tensor-only scalar method to a native compiler AST event.

    A caller must additionally establish a failed, verified single-column suite
    input. Even then the returned sites support a repair hypothesis, not cause
    proof or a numerical acceptance credential.
    """
    if type(input_rank) is not int or input_rank != 2:
        return None
    try:
        tree = _parse(source, tree)
        event = validate_compile_failure(compile_failure, source)
        if tree is None or event is None:
            return None
        imports = [node for node in tree.body if isinstance(node, ast.Import)]
        imported = {node.names[0].name: node.names[0].asname for node in imports if len(node.names) == 1}
        wanted = {'triton': None, 'triton.language': 'tl'}
        if imported not in (wanted, dict(wanted, torch=None)) or len(imports) != len(imported):
            return None
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if (len(functions) != 2 or any(not isinstance(node, (ast.Import, ast.FunctionDef))
                and not _doc(node) for node in tree.body)):
            return None
        entries = [node for node in functions if node.name == 'kernel' and not node.decorator_list]
        helpers = [node for node in functions if len(node.decorator_list) == 1
                   and isinstance(node.decorator_list[0], ast.Attribute)
                   and _name(node.decorator_list[0].value, 'triton') and node.decorator_list[0].attr == 'jit']
        if len(entries) != 1 or len(helpers) != 1:
            return None
        entry, helper = entries[0], helpers[0]
        if (helper.name in {'kernel', 'X', 'torch', 'triton', 'tl'} or helper.name != event['helper_name']
                or not _signature(entry) or not _signature(helper)
                or [arg.arg for arg in entry.args.args] != ['X'] or entry.args.args[0].annotation is not None
                or any(node.lineno >= min(entry.lineno, helper.lineno) for node in imports)):
            return None
        params = [arg.arg for arg in helper.args.args]
        if (len(set(params)) != len(params) or set(params) & {'kernel', 'torch', 'triton', 'tl', helper.name}
                or any(arg.annotation is not None and not (isinstance(arg.annotation, ast.Attribute)
                    and _name(arg.annotation.value, 'tl') and arg.annotation.attr == 'constexpr')
                    for arg in helper.args.args)):
            return None
        # Import authority must exist for every host module read.
        if 'torch' not in imported and any(_name(node, 'torch') for node in ast.walk(entry)):
            return None
        matcher = _Matcher(source, entry, helper)
        bindings, values = matcher.wrapper()
        scalar, formal, conversion = matcher.helper_body(bindings, values)
        locus = (event['line'], event['column'], event['end_line'], event['end_column'])
        nodes = [node for node in ast.walk(helper)
                 if (getattr(node, 'lineno', None), getattr(node, 'col_offset', None),
                     getattr(node, 'end_lineno', None), getattr(node, 'end_col_offset', None)) == locus
                 and code_hash(ast.dump(node, include_attributes=False)) == event['node_ast_hash']]
        if len(nodes) != 1:
            return None
        failed = nodes[0]
        assignments = [node for node in helper.body if isinstance(node, ast.Assign) and node.value is conversion]
        if failed is not conversion and failed is not conversion.func and not any(failed is n for n in assignments):
            return None
        sites = [('input_columns', matcher.shape_sites['columns']),
                 ('launch_dimension', bindings[scalar]), ('helper_dimension', formal),
                 ('scalar_conversion', conversion), ('compiler_failure_node', failed)]
        result = []
        for role, node in sites:
            excerpt = ast.get_source_segment(source, node)
            if excerpt is None or len(excerpt.encode('utf-8')) > 512:
                return None
            result.append(dict(role=role, line=node.lineno, end_line=node.end_lineno,
                               column=node.col_offset, end_column=node.end_col_offset, excerpt=excerpt,
                               excerpt_sha256=code_hash(excerpt), trust='untrusted_source_data'))
        return result
    except (_PatternError, AttributeError, KeyError, TypeError, ValueError, RecursionError, UnicodeError):
        return None
