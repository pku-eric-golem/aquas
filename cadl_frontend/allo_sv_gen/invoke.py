"""Subtask C: `invoke` -> ap_ctrl_hs handshake and SPM <-> ap_memory bridges.

An invoke latches each array binding's element offset into `hls_<param>_base`
and each scalar binding into `hls_<param>`, pulses ap_start for one cycle
(the core samples it while idle), then waits for ap_done. Memory ports read
`spm[base + address]` with one cycle of latency and write only while the
core runs, as in the manual adapter.

Results bind in Python return order: an array result is an SPM range, like
an array argument; a scalar result is captured from its ap_vld strobe and
becomes a local value of the flow once the core is done. Array views are
checked against the kernel manifest: element count, width and float-ness.
"""
from __future__ import annotations

import math

from .. import cadl_ast as ast
from ..allo import constant_integer
from ..allo_kernel.interface import HlsInstance, KernelParam
from .fsm import FlowBuilder, _basic
from .spm import SpmLayout


class InvokeBridge:
    def __init__(self, instance: HlsInstance, layout: SpmLayout):
        self.instance, self.layout = instance, layout
        self.arrays = {m.param for m in instance.memories}
        self.spm_of: dict[str, str] = {}  # array param -> bound SPM
        self.sites = 0
        largest = max(s.elements for s in layout.spms)
        self.index_bits = max(1, math.ceil(math.log2(largest)))

    def base(self, param: str) -> str:
        return f"hls_{param}_base"

    def captured(self, result: str) -> str:
        return f"hls_{result}_value"

    def seen(self, result: str) -> str:
        return f"hls_{result}_seen"

    def handle(self, b: FlowBuilder, stmt: ast.InvokeStmt, directives: dict) -> None:
        if directives:
            raise NotImplementedError(f"{b.flow.name}: directives on invoke")
        if stmt.kernel != self.instance.top:
            raise ValueError(f"{b.flow.name}: invoke {stmt.kernel}, but the HLS core is "
                             f"{self.instance.top}")
        bound = [binding.name for binding in stmt.bindings]
        expected = sorted(p.name for p in self.instance.params)
        if sorted(bound) != expected:
            raise ValueError(f"{b.flow.name}: invoke binds {sorted(bound)}, kernel takes {expected}")
        returns = self.instance.returns
        if len(stmt.results) != len(returns):
            raise ValueError(f"{b.flow.name}: invoke binds {len(stmt.results)} results, "
                             f"{stmt.kernel} returns {len(returns)}")
        params = {p.name: p for p in self.instance.params}
        for binding in stmt.bindings:
            value = binding.value
            if binding.name in self.arrays:
                self._view(b, params[binding.name], value, binding.name)
            else:
                if isinstance(value, ast.RangeSliceExpr):
                    raise ValueError(f"{b.flow.name}: scalar {binding.name} bound to a range")
                bits = self.instance.scalars[binding.name].bits
                b.action(f"{self.instance.signal(binding.name)} <= {bits}'({b.expr(value)});")
        scalars = []
        for index, (result, ret) in enumerate(zip(stmt.results, returns)):
            what = f"result {index} of {stmt.kernel}"
            if ret.kind == "ap_memory":
                if not isinstance(result, ast.AlloArrayResult):
                    raise ValueError(f"{b.flow.name}: {what} is an array; bind an SPM range")
                self._view(b, ret, result.view, what)
            else:
                if not isinstance(result, ast.AlloScalarResult):
                    raise ValueError(f"{b.flow.name}: {what} is a scalar; bind a new local value")
                var = b.declare(result.name, *_basic(result.ty.basic_type))
                if ret.bits is not None and var.bits != ret.bits:
                    raise ValueError(f"{b.flow.name}: {result.name} is {var.bits} bits, "
                                     f"{what} is {ret.bits}")
                scalars.append((result.name, var, ret.name))
        start = b.wait("invoke_start", "1'b1")
        b.drive(self.instance.signal("ap_start"), start)
        run = b.wait("invoke_run", self.instance.signal("ap_done"))
        b.drive("hls_running", run)
        # The strobe may fire on the ap_done cycle, before the capture register has it.
        for name, var, port in scalars:
            sig = self.instance.signal
            value = var.cast(f"{sig(HlsInstance.valid(port))} ? {sig(port)} : {self.captured(port)}")
            b.action(f"{var.reg} <= {value};")
            b.scope[name] = var
            b.env[name] = value
        self.sites += 1

    def _view(self, b: FlowBuilder, param: KernelParam, value: ast.Expr, what: str) -> None:
        """Latch the base of an SPM range bound to array `param`, after checking it fits."""
        if not (isinstance(value, ast.RangeSliceExpr) and isinstance(value.expr, ast.IdentExpr)
                and value.expr.name in {s.name for s in self.layout.spms}):
            raise ValueError(f"{b.flow.name}: {what} must bind an SPM range")
        spm = self.layout[value.expr.name]
        count = constant_integer(value.length)
        if param.shape and count != math.prod(param.shape):
            raise ValueError(f"{b.flow.name}: {what} binds {count} elements of {spm.name}, "
                             f"the kernel's {param.name} has shape {list(param.shape)}")
        if param.bits is not None and (param.bits, param.dtype.startswith(("f", "bf"))) \
                != (spm.elem_bits, spm.is_float):
            raise ValueError(f"{b.flow.name}: {what} is {param.dtype}, {spm.name} holds "
                             f"{spm.elem_bits}-bit {'float' if spm.is_float else 'integer'}s")
        if self.spm_of.setdefault(param.name, spm.name) != spm.name:
            raise NotImplementedError(
                f"{param.name} is bound to both {self.spm_of[param.name]} and {spm.name}")
        b.action(f"{self.base(param.name)} <= {self.index_bits}'({b.expr(value.start)});")

    def declarations(self) -> str:
        lines = [f"reg [{self.index_bits - 1}:0] {self.base(p)};" for p in sorted(self.arrays)]
        for name, port in self.instance.results.items():
            lines += [f"reg [{port.bits - 1}:0] {self.captured(name)};", f"reg {self.seen(name)};"]
        return "\n".join(lines + ["reg hls_running;"]) + "\n"

    def bridge(self) -> str:
        if not self.sites:
            raise ValueError(f"{self.instance.top} is never invoked")
        sig = self.instance.signal
        reads, writes, checks = [], [], []
        for mem in self.instance.memories:
            spm = self.layout[self.spm_of[mem.param]]
            port = lambda role: sig(f"{mem.param}_{role}{mem.index}")  # noqa: E731
            index = f"{self.base(mem.param)} + {port('address')}"
            wide = f"{{1'b0, {self.base(mem.param)}}} + {port('address')}"
            checks.append(f"  if ({port('ce')}) assert({wide} < {spm.elements});")
            if "q" in mem.signals:
                reads.append(f"  if (!rst && {port('ce')})\n"
                             f"    {port('q')} <= {spm.signal}[{index}];")
            if "d" in mem.signals:
                writes.append(f"  if (!rst && hls_running && {port('ce')} && {port('we')})\n"
                              f"    {spm.signal}[{index}] <= {port('d')};")
        captures = []
        for name in self.instance.results:
            valid = sig(HlsInstance.valid(name))
            captures.append(f"  if (rst || {sig('ap_start')}) {self.seen(name)} <= 1'b0;\n"
                            f"  else if ({valid}) {self.seen(name)} <= 1'b1;\n"
                            f"  if (!rst && {valid}) {self.captured(name)} <= {sig(name)};")
            checks.append(f"  if (hls_running && {sig('ap_done')})\n"
                          f"    assert({self.seen(name)} || {valid});")
        return ("always @(posedge clk) begin\n" + "\n".join(reads + writes + captures) + "\nend\n"
                "`ifndef SYNTHESIS\nalways @(posedge clk) if (!rst) begin\n"
                + "\n".join(checks) + "\nend\n`endif\n")
