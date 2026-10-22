"""Subtask C: `invoke` -> ap_ctrl_hs handshake and SPM <-> ap_memory bridges.

An invoke latches each array binding's element offset into `hls_<param>_base`
and each scalar binding into `hls_<param>`, pulses ap_start for one cycle
(the core samples it while idle), then waits for ap_done. Memory ports read
`spm[base + address]` with one cycle of latency and write only while the
core runs, as in the manual adapter.
"""
from __future__ import annotations

import math

from .. import cadl_ast as ast
from .fsm import FlowBuilder
from ..allo_kernel.interface import HlsInstance
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

    def handle(self, b: FlowBuilder, stmt: ast.InvokeStmt, directives: dict) -> None:
        if directives:
            raise NotImplementedError(f"{b.flow.name}: directives on invoke")
        if stmt.kernel != self.instance.top:
            raise ValueError(f"{b.flow.name}: invoke {stmt.kernel}, but the HLS core is "
                             f"{self.instance.top}")
        if stmt.results:
            raise NotImplementedError(f"{b.flow.name}: invoke results")
        bound = [binding.name for binding in stmt.bindings]
        expected = sorted(self.arrays | set(self.instance.scalars))
        if sorted(bound) != expected:
            raise ValueError(f"{b.flow.name}: invoke binds {sorted(bound)}, kernel takes {expected}")
        for binding in stmt.bindings:
            value = binding.value
            if binding.name in self.arrays:
                if not (isinstance(value, ast.RangeSliceExpr) and isinstance(value.expr, ast.IdentExpr)
                        and value.expr.name in {s.name for s in self.layout.spms}):
                    raise ValueError(f"{b.flow.name}: {binding.name} must bind an SPM range")
                spm = value.expr.name
                if self.spm_of.setdefault(binding.name, spm) != spm:
                    raise NotImplementedError(
                        f"{binding.name} is bound to both {self.spm_of[binding.name]} and {spm}")
                b.action(f"{self.base(binding.name)} <= {self.index_bits}'({b.expr(value.start)});")
            else:
                if isinstance(value, ast.RangeSliceExpr):
                    raise ValueError(f"{b.flow.name}: scalar {binding.name} bound to a range")
                bits = self.instance.scalars[binding.name].bits
                b.action(f"{self.instance.signal(binding.name)} <= {bits}'({b.expr(value)});")
        start = b.wait("invoke_start", "1'b1")
        b.drive(self.instance.signal("ap_start"), start)
        run = b.wait("invoke_run", self.instance.signal("ap_done"))
        b.drive("hls_running", run)
        self.sites += 1

    def declarations(self) -> str:
        lines = [f"reg [{self.index_bits - 1}:0] {self.base(p)};" for p in sorted(self.arrays)]
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
        return ("always @(posedge clk) begin\n" + "\n".join(reads + writes) + "\nend\n"
                "`ifndef SYNTHESIS\nalways @(posedge clk) if (!rst) begin\n"
                + "\n".join(checks) + "\nend\n`endif\n")
