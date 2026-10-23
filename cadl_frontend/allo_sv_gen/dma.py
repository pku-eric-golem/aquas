"""Subtask D: `_mem` <-> SPM transfers through DMA channel 0.

TlDmaMultichannel semantics (generators/aps/.../TlDmaMultichannel.scala):
a transfer moves `1 << length` bytes in 8-byte beats; with stride_x != 0 the
SPM address jumps by stride_x * stride_y bytes after every stride_x bytes.
The wrapper ties ready and poll_for_idle to the channel's idle flag, so a
transfer is: assert enable until ready, wait for idle to fall (the engine
took the request), then wait for idle to rise.

CADL counts `_mem[addr +: n]` in SPM elements and gives addr and stride_x
in bytes; SPM offsets are elements.
"""
from __future__ import annotations

import math

from .. import cadl_ast as ast
from .fsm import FlowBuilder
from .spm import SpmLayout

LOAD, STORE = "dma_cpu_to_isax_ch0", "dma_isax_to_cpu_ch0"
FIELDS = {"cpu_addr": 32, "isax_addr": 32, "length": 4, "stride_x": 8, "stride_y": 8}


def _literal(e: ast.Expr | None, what: str) -> int:
    if not (isinstance(e, ast.LitExpr) and isinstance(e.literal.lit, ast.LiteralInner_Fixed)):
        raise ValueError(f"{what} must be an integer literal")
    return e.literal.lit.value


def _range_of(e: ast.Expr, names) -> ast.RangeSliceExpr | None:
    if isinstance(e, ast.RangeSliceExpr) and isinstance(e.expr, ast.IdentExpr) and e.expr.name in names:
        return e
    return None


class DmaBridge:
    def __init__(self, layout: SpmLayout, max_burst_bytes: int = 128, beat_bytes: int = 8):
        self.layout, self.max_burst, self.beat = layout, max_burst_bytes, beat_bytes
        self.spm_names = {s.name for s in layout.spms}
        self.sites = 0

    def handle(self, b: FlowBuilder, stmt: ast.AssignStmt, directives: dict) -> None:
        where = b.flow.name
        if (mem := _range_of(stmt.rhs, {"_mem"})) and (spm := _range_of(stmt.lhs, self.spm_names)):
            channel = LOAD
        elif (mem := _range_of(stmt.lhs, {"_mem"})) and (spm := _range_of(stmt.rhs, self.spm_names)):
            channel = STORE
        else:
            raise NotImplementedError(f"{where}: DMA statement {stmt} (only _mem <-> SPM ranges)")

        count = _literal(mem.length, f"{where}: _mem transfer length")
        if spm.length is not None and _literal(spm.length, f"{where}: SPM range length") != count:
            raise ValueError(f"{where}: SPM and _mem ranges differ in length")
        nbytes = count * self.layout.elem_bytes
        if nbytes & (nbytes - 1) or not self.beat <= nbytes <= self.max_burst:
            raise ValueError(f"{where}: a DMA burst moves a power of two between {self.beat} "
                             f"and {self.max_burst} bytes, not {nbytes}")

        extra = set(directives) - {"stride_x", "stride_y"}
        if extra:
            raise NotImplementedError(f"{where}: DMA directives {sorted(extra)}")
        stride_x = _literal(directives["stride_x"], "stride_x") if "stride_x" in directives else 0
        stride_y = _literal(directives["stride_y"], "stride_y") if "stride_y" in directives else 0
        if stride_y and not stride_x:
            raise ValueError(f"{where}: stride_y needs stride_x")
        if not (0 <= stride_x < 256 and 0 <= stride_y < 256) or stride_x % self.beat:
            raise ValueError(f"{where}: stride_x must be a multiple of {self.beat} below 256 "
                             f"and stride_y below 256")

        # SPM elements the transfer touches: rows of stride_x bytes, stride_x * stride_y apart.
        if stride_x:
            span = (nbytes - 1) // stride_x * stride_x * stride_y + (nbytes - 1) % stride_x + 1
        else:
            span = nbytes
        b.check_range(spm.start, -(-span // self.layout.elem_bytes), self.layout[spm.expr.name],
                      "dma_range", "DMA range")
        base = self.layout.base(spm.expr.name)
        b.action(f"dma_cpu_addr <= 32'({b.expr(mem.start)});")
        b.action(f"dma_isax_addr <= 32'(32'h{base:x} + ({b.expr(spm.start)}) * "
                 f"32'd{self.layout.elem_bytes});")
        b.action(f"dma_length <= 4'd{int(math.log2(nbytes))};")
        b.action(f"dma_stride_x <= 8'd{stride_x};")
        b.action(f"dma_stride_y <= 8'd{stride_y};")
        kind = "load" if channel == LOAD else "store"
        issue = b.wait(f"dma_{kind}_issue", f"{channel}_ready")
        b.drive(f"{channel}_enable", issue)
        b.wait("dma_busy", "!dma_idle")
        b.wait("dma_done", "dma_idle")
        self.sites += 1

    def declarations(self) -> str:
        if not self.sites:
            return ""
        regs = [f"reg [{bits - 1}:0] dma_{name};" for name, bits in FIELDS.items()]
        return "\n".join(regs + ["wire dma_idle = dma_poll_for_idle_ch0_ready && "
                                 "dma_poll_for_idle_ch0_res0;"]) + "\n"

    def wiring(self) -> str:
        if not self.sites:
            return ""
        return "\n".join(f"assign {ch}_{name} = dma_{name};"
                         for ch in (LOAD, STORE) for name in FIELDS) + "\n"
