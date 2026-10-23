"""Subtask C: `invoke` -> ap_ctrl_hs handshake and SPM <-> ap_memory bridges.

An invoke latches each array binding's element offset into `hls_<param>_base`
and each scalar binding into `hls_<param>`, then holds ap_start until the core
raises ap_ready (ap_ctrl_hs; a loop pipelined in the top function only issues
iterations while ap_start is high), then waits for ap_done unless it came with
ap_ready, as from a sequential core.
Memory ports read `spm[base + element]` with one cycle of latency and write
only while the invoke runs (both states); `element` is the port address, or
for a bank of a partitioned kernel array the element that bank address holds
(interface.bank_element). Writes go to the SPM's write block (spm.py).

Each SPM bank serves at most BANK_PORTS kernel ports at once (a true
dual-port memory). Per invoke, the kernel ports that may reach each bank are
counted from the partitions of both sides: exactly for a constant base, and
for a runtime base the worst case over all bases (for a cyclic SPM, every
base rotates one binding's per-bank counts, so its largest count holds; for
block banks or one bank, all its ports may meet). More is rejected rather
than arbitrated, since ap_memory ports cannot be stalled.

Results bind in Python return order: an array result is an SPM range, like
an array argument; a scalar result is captured from its ap_vld strobe and
becomes a local value of the flow once the core is done. Array views are
checked against the kernel manifest: element count, width and float-ness.
"""
from __future__ import annotations

import math

from .. import cadl_ast as ast
from ..allo import constant_integer
from ..allo_kernel.interface import HlsInstance, KernelParam, MemoryPort, Partition, bank_element
from .fsm import FlowBuilder, _basic
from .spm import Spm, SpmLayout, SpmWrite

BANK_PORTS = 2


def bank_element_sv(shape: tuple[int, ...], partition: Partition | None, bank: int | None,
                    addr: str) -> str:
    """SystemVerilog for interface.bank_element at a runtime address."""
    if partition is None:
        return addr
    k = partition.dim - 1
    inner, local, factor = math.prod(shape[k + 1:]), shape[k] // partition.factor, partition.factor
    first = bank if partition.kind == "cyclic" else bank * local
    step = factor if partition.kind == "cyclic" else 1
    j = f"({addr})" if inner == 1 else f"(({addr}) / {inner})"
    if k > 0:
        j = f"({j} % {local})"
    i = f"({first} + {j} * {step})"
    outer = f"(({addr}) / {inner * local}) * {shape[k]} + " if k > 0 else ""
    rest = f" + ({addr}) % {inner}" if inner > 1 else ""
    return f"(({outer}{i}){f' * {inner}' if inner > 1 else ''}{rest})"


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
        self.site_banks: list[tuple[KernelParam, Spm, int | None]] = []
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
        self._check_banks(b.flow.name)
        sig = self.instance.signal
        start = b.wait("invoke_start", sig("ap_ready"))
        b.drive(sig("ap_start"), start)
        b.drive("hls_running", start)
        run = b.wait_unless("invoke_run", sig("ap_done"))
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
        index = b.check_range(value.start, count, spm, "invoke_range", what)
        if self.spm_of.setdefault(param.name, spm.name) != spm.name:
            raise NotImplementedError(
                f"{param.name} is bound to both {self.spm_of[param.name]} and {spm.name}")
        b.action(f"{self.base(param.name)} <= {self.index_bits}'({index});")
        self.site_banks.append((param, spm, constant_integer(value.start)))

    def _ports(self, param: KernelParam) -> list[MemoryPort]:
        return [m for m in self.instance.memories if m.param == param.name]

    def _bank_counts(self, param: KernelParam, spm: Spm, base: int) -> list[int]:
        """Kernel ports of `param` that may reach each bank of `spm` from element `base`."""
        counts = [0] * spm.banks
        for mem in self._ports(param):
            size = math.prod(param.partition.bank_shape(param.shape)) if param.partition \
                else math.prod(param.shape)
            for bank in {spm.bank_of(base + bank_element(param.shape, param.partition, mem.bank, a))
                         for a in range(size)}:
                counts[bank] += 1
        return counts

    def _check_banks(self, flow: str) -> None:
        """At most BANK_PORTS kernel ports per SPM bank during this invoke."""
        for spm in {spm.name: spm for _, spm, _ in self.site_banks}.values():
            exact, worst = [0] * spm.banks, 0
            for param, bound, base in self.site_banks:
                if bound is not spm:
                    continue
                if base is not None:
                    exact = [x + y for x, y in zip(exact, self._bank_counts(param, spm, base))]
                elif spm.banks > 1 and spm.cyclic:
                    worst += max(self._bank_counts(param, spm, 0))
                else:
                    worst += len(self._ports(param))
            if max(exact) + worst > BANK_PORTS:
                names = sorted(p.name for p, bound, _ in self.site_banks if bound is spm)
                raise ValueError(
                    f"{flow}: up to {max(exact) + worst} kernel ports of {names} may reach one "
                    f"bank of {spm.name} ({spm.banks} {'cyclic' if spm.cyclic else 'block'} "
                    f"bank{'s' if spm.banks > 1 else ''}) at once; a bank has {BANK_PORTS} ports")

    def declarations(self) -> str:
        lines = [f"reg [{self.index_bits - 1}:0] {self.base(p)};" for p in sorted(self.arrays)]
        for name, port in self.instance.results.items():
            lines += [f"reg [{port.bits - 1}:0] {self.captured(name)};", f"reg {self.seen(name)};"]
        return "\n".join(lines + ["reg hls_running;", "reg hls_start_prev;",
                                   "wire hls_start_rise = hls_ap_start && !hls_start_prev;"]) + "\n"

    def bridge(self) -> str:
        if not self.sites:
            raise ValueError(f"{self.instance.top} is never invoked")
        sig = self.instance.signal
        reads, checks = [], []
        self.writes: list[SpmWrite] = []
        params = {p.name: p for p in [*self.instance.params, *self.instance.returns]}
        for mem in self.instance.memories:
            spm, param = self.layout[self.spm_of[mem.param]], params[mem.param]
            port = lambda role: sig(mem.port(role))  # noqa: E731
            element = bank_element_sv(param.shape, param.partition, mem.bank, port("address"))
            index = f"{self.base(mem.param)} + {element}"
            wide = f"{{1'b0, {self.base(mem.param)}}} + {element}"
            checks.append(f"  if ({port('ce')}) assert({wide} < {spm.elements});")
            if "q" in mem.signals:
                reads.append(f"  if (!rst && {port('ce')})\n"
                             f"    {port('q')} <= {spm.ref(index)};")
            if "d" in mem.signals:
                self.writes.append(SpmWrite(spm.name, "invoke",
                                            f"!rst && hls_running && {port('ce')} && {port('we')}",
                                            index, port("d")))
        captures = ["  hls_start_prev <= !rst && hls_ap_start;"]
        for name in self.instance.results:
            valid = sig(HlsInstance.valid(name))
            captures.append(f"  if (!rst && {valid}) {self.seen(name)} <= 1'b1;\n"
                            f"  else if (rst || hls_start_rise) {self.seen(name)} <= 1'b0;\n"
                            f"  if (!rst && {valid}) {self.captured(name)} <= {sig(name)};")
            checks.append(f"  if (hls_running && {sig('ap_done')})\n"
                          f"    assert({self.seen(name)} || {valid});")
        return ("always @(posedge clk) begin\n" + "\n".join(reads + captures) + "\nend\n"
                "`ifndef SYNTHESIS\nalways @(posedge clk) if (!rst) begin\n"
                + "\n".join(checks) + "\nend\n`endif\n")
