"""Structural Allo checks only: no imports, execution, scheduling, or codegen."""
from __future__ import annotations

import ast
import math
import re
from dataclasses import fields, is_dataclass

from . import cadl_ast as c
from .parser import CADLParseError, parse_proc


def walk_nodes(value):
    """Visit CADL nodes, including instructions nested in control flow."""
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from walk_nodes(child)
    elif is_dataclass(value):
        yield value
        for field in fields(value):
            if field.name != "span":
                yield from walk_nodes(getattr(value, field.name))


def reject_allo_backend(proc: c.Proc, backend: str) -> None:
    if proc.allo_kernels or any(isinstance(node, c.InvokeStmt)
                               for flow in proc.flows.values()
                               for node in walk_nodes(flow.body)):
        raise NotImplementedError(
            f"{backend} does not support CADL Allo kernels/invoke. "
            "The independent Allo + Vitis wrapper backend is not implemented; "
            "use aps-frontend check for structural validation. "
            "Allo designs must not enter the Aquas HLS pipeline.")


def constant_integer(expr):
    """Fold literal integer expressions without evaluating user code or state."""
    if isinstance(expr, c.LitExpr) and isinstance(expr.literal.lit, c.LiteralInner_Fixed):
        return expr.literal.lit.value
    if isinstance(expr, c.UnaryExpr):
        value = constant_integer(expr.operand)
        if value is not None:
            if expr.op == c.UnaryOp.NEG:
                return -value
            if expr.op == c.UnaryOp.BIT_NOT:
                return ~value
    if isinstance(expr, c.BinaryExpr):
        left, right = constant_integer(expr.left), constant_integer(expr.right)
        if left is None or right is None:
            return None
        operations = {
            c.BinaryOp.ADD: lambda: left + right,
            c.BinaryOp.SUB: lambda: left - right,
            c.BinaryOp.MUL: lambda: left * right,
            c.BinaryOp.BIT_AND: lambda: left & right,
            c.BinaryOp.BIT_OR: lambda: left | right,
            c.BinaryOp.BIT_XOR: lambda: left ^ right,
        }
        if expr.op in operations:
            return operations[expr.op]()
        if expr.op in (c.BinaryOp.LSHIFT, c.BinaryOp.RSHIFT) and 0 <= right <= 4096:
            return left << right if expr.op == c.BinaryOp.LSHIFT else left >> right
    return None


class _ModuleBindings(ast.NodeVisitor):
    """Names that can be exported by a Python module, without entering scopes."""
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Lambda(self, node):
        pass

    def visit_ListComp(self, node):
        pass

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_Import(self, node):
        self.names.update(alias.asname or alias.name.split('.')[0] for alias in node.names)

    def visit_ImportFrom(self, node):
        self.names.update(alias.asname or alias.name for alias in node.names if alias.name != '*')


def _return_count(function):
    annotation = function.returns
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return 0
    if isinstance(annotation, ast.Tuple):
        return len(annotation.elts)
    if isinstance(annotation, ast.Subscript) and (
        isinstance(annotation.value, ast.Name) and annotation.value.id in {'tuple', 'Tuple'}
        or isinstance(annotation.value, ast.Attribute) and annotation.value.attr == 'Tuple'
    ):
        elements = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        if any(isinstance(item, ast.Constant) and item.value is Ellipsis for item in elements):
            return None
        return len(elements)
    if annotation is not None:
        element = annotation.value if isinstance(annotation, ast.Subscript) else annotation
        name = element.id if isinstance(element, ast.Name) else (
            element.attr if isinstance(element, ast.Attribute) else "")
        if re.fullmatch(r"(?:u?int|float)\d+|bfloat16|index|bool", name):
            return 1
        # A string, qualified alias or type expression may describe a tuple;
        # only Allo can resolve it. Avoid rejecting a generic kernel here.
        return None
    # An unannotated function with no value-return is unambiguously void.
    pending = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Return) and node.value is not None:
            return None
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))
    return 0


class _Checker:
    def __init__(self, proc, source, filename):
        self.proc, self.source, self.filename = proc, source, filename
        self.signatures = {}

    def error(self, message, span):
        raise CADLParseError(message, span.line, span.column,
                             self.filename, self.source.splitlines())

    def kernel(self, kernel):
        try:
            module = ast.parse(kernel.source.text)
        except SyntaxError as exc:
            line, column = kernel.source.source_position(exc.lineno or 1, exc.offset or 1)
            raise CADLParseError(f"Invalid embedded Python: {exc.msg}", line, column,
                                 self.filename, self.source.splitlines()) from None
        functions = [node for node in module.body
                     if isinstance(node, ast.FunctionDef) and node.name == kernel.name]
        if len(functions) != 1:
            self.error(f"Allo kernel {kernel.name} requires exactly one top-level Python def of that name",
                       kernel.span)
        exports = _ModuleBindings()
        exports.visit(module)
        if kernel.schedule not in exports.names:
            self.error(f"Schedule name {kernel.schedule} is not declared in the Python block", kernel.span)
        function = functions[0]
        args = function.args
        if args.vararg or args.kwarg:
            self.error("Allo invocation requires a fixed, named parameter list (no *args/**kwargs)", kernel.span)
        names = [arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs]
        self.signatures[kernel.name] = names, _return_count(function)

    def expression(self, expr, locals_, span):
        known = locals_ | self.proc.statics.keys() | self.proc.registers.keys()
        known |= {'_irf', '_mem', '_csr', '_burst_read', '_burst_write'}
        for node in walk_nodes(expr):
            if isinstance(node, c.IdentExpr) and node.name not in known:
                self.error(f"Unknown CADL value in Allo binding: {node.name}", span)

    def view(self, view, locals_, span):
        if not isinstance(view.expr, c.IdentExpr):
            self.error("Allo array bindings require a global SPM range", span)
        name = view.expr.name
        memory = self.proc.statics.get(name)
        if name in locals_ or memory is None or not isinstance(memory.ty, c.DataType_Array):
            self.error(f"Allo array binding {name} must name an unshadowed global static SPM array", span)
        self.expression(view.start, locals_, span)
        length = constant_integer(view.length)
        if length is None or length <= 0:
            self.error("Allo SPM ranges require an explicit positive constant element count", span)
        capacity = math.prod(memory.ty.dimensions)
        start = constant_integer(view.start)
        if length > capacity or start is not None and (start < 0 or start + length > capacity):
            self.error(f"Allo SPM range exceeds {name}'s {capacity} elements", span)

    def invoke(self, stmt, locals_):
        if stmt.kernel not in self.signatures:
            self.error(f"Unknown Allo kernel: {stmt.kernel}", stmt.span)
        parameters, returns = self.signatures[stmt.kernel]
        seen = set()
        for binding in stmt.bindings:
            if binding.name in seen:
                self.error(f"Duplicate Allo parameter binding: {binding.name}", binding.span)
            seen.add(binding.name)
            if binding.name not in parameters:
                self.error(f"Unknown parameter {binding.name} for Allo kernel {stmt.kernel}", binding.span)
            value = binding.value
            if isinstance(value, c.RangeSliceExpr):
                self.view(value, locals_, binding.span)
            else:
                self.expression(value, locals_, binding.span)
                for node in walk_nodes(value):
                    if isinstance(node, c.RangeSliceExpr):
                        self.error("An Allo SPM range must be the whole parameter binding", binding.span)
                if isinstance(value, c.IdentExpr):
                    memory = self.proc.statics.get(value.name)
                    if value.name not in locals_ and memory and isinstance(memory.ty, c.DataType_Array):
                        self.error("Bind an explicit SPM range, not a bare array name", binding.span)
        missing = set(parameters) - seen
        if missing:
            self.error(f"Missing Allo parameter bindings: {', '.join(sorted(missing))}", stmt.span)
        if returns is not None and returns != len(stmt.results):
            self.error(f"Allo kernel {stmt.kernel} has {returns} return values, got {len(stmt.results)} targets", stmt.span)
        new_names = set()
        for result in stmt.results:
            if isinstance(result, c.AlloArrayResult):
                self.view(result.view, locals_, result.span)
            else:
                if result.name in (locals_ | new_names | self.proc.statics.keys() | self.proc.registers.keys()
                                   | {'_irf', '_mem', '_csr', '_burst_read', '_burst_write'}):
                    self.error(f"Allo scalar result must declare a new local value: {result.name}", result.span)
                new_names.add(result.name)
        # No result name is in scope while other arguments/targets are evaluated.
        locals_.update(new_names)

    def statements(self, statements, locals_):
        for stmt in statements or []:
            if isinstance(stmt, c.InvokeStmt):
                self.invoke(stmt, locals_)
            elif isinstance(stmt, c.AssignStmt) and isinstance(stmt.lhs, c.IdentExpr):
                locals_.add(stmt.lhs.name)
            elif isinstance(stmt, c.StaticStmt):
                locals_.add(stmt.static.id)
            elif isinstance(stmt, c.IfStmt):
                self.statements(stmt.then_body, locals_.copy())
                self.statements(stmt.else_body, locals_.copy())
            elif isinstance(stmt, c.DoWhileStmt):
                locals_.update(binding.id for binding in stmt.bindings)
                self.statements(stmt.body, locals_.copy())
            elif isinstance(stmt, c.GuardStmt):
                self.statements([stmt.stmt], locals_.copy())
            elif isinstance(stmt, c.SpawnStmt):
                invocation = next((node for node in walk_nodes(stmt) if isinstance(node, c.InvokeStmt)), None)
                if invocation:
                    self.error("invoke is synchronous and is not supported inside spawn", invocation.span)

    def run(self):
        for kernel in self.proc.allo_kernels.values():
            self.kernel(kernel)
        for flow in self.proc.flows.values():
            self.statements(flow.body, {name for name, _ in flow.inputs})


def check_proc(proc: c.Proc, source: str, filename: str | None = None) -> None:
    """Check structural bindings, not Allo types, effects, banking, or RTL legality."""
    _Checker(proc, source, filename).run()


def check_source(source: str, filename: str | None = None) -> c.Proc:
    proc = parse_proc(source, filename)
    check_proc(proc, source, filename)
    return proc
