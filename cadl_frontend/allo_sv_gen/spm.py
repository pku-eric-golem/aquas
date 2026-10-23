"""Subtask A: scratchpad (SPM) layout and storage for CADL statics.

Each static array gets its own byte-address region of equal power-of-two size,
so the DMA side selects a scratchpad with a bit slice of the address (GEMM:
three 512 KiB regions, select = addr[20:19], element = addr[18:2]).

A static with CADL partition attributes on its one dimension
(#[partition_dim_array([0])], #[partition_factor_array([f])],
#[partition_cyclic_array([1])] for cyclic or [0] for block) is stored as f
physical banks, `spm_<name>[bank][offset]`; every access maps its element
index to a bank and offset with `Spm.ref`. Others stay one flat array.

All writes to one SPM sit in one always block (`generate_spm_writes`): DMA
burst lanes, kernel ports during an invoke and FSM element writes. Only one
of the three is active at a time (one command in flight; DMA and invoke never
overlap), which simulation asserts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import cadl_ast as ast


@dataclass
class Spm:
    name: str
    index: int
    elements: int
    elem_bits: int
    is_float: bool = False
    signed: bool = False
    banks: int = 1
    cyclic: bool = True  # cyclic or block banks; irrelevant for one bank

    @property
    def signal(self) -> str:
        return f"spm_{self.name}"

    @property
    def bank_size(self) -> int:
        return self.elements // self.banks

    def bank_of(self, element: int) -> int:
        return element % self.banks if self.cyclic else element // self.bank_size

    def ref(self, index: str) -> str:
        """The storage word of element `index` (a SystemVerilog expression)."""
        if self.banks == 1:
            return f"{self.signal}[{index}]"
        if self.cyclic:
            return f"{self.signal}[({index}) % {self.banks}][({index}) / {self.banks}]"
        return f"{self.signal}[({index}) / {self.bank_size}][({index}) % {self.bank_size}]"


@dataclass
class SpmWrite:
    """A write source of an SPM: when `enable` holds, element `index` gets `data`."""
    spm: str
    origin: str  # "dma", "invoke" or "fsm"; sources of different origins never overlap
    enable: str
    index: str
    data: str


@dataclass
class SpmLayout:
    spms: list[Spm]
    offset_bits: int  # log2 of the bytes per region
    select_bits: int

    def __getitem__(self, name: str) -> Spm:
        return next(spm for spm in self.spms if spm.name == name)

    def base(self, name: str) -> int:
        """Byte address of a scratchpad as seen by the DMA engine."""
        return self[name].index << self.offset_bits

    @property
    def elem_bytes(self) -> int:
        return self.spms[0].elem_bits // 8

    def select(self, addr: str) -> str:
        hi = self.offset_bits + self.select_bits - 1
        return f"{addr}[{hi}:{self.offset_bits}]"

    def element(self, addr: str) -> str:
        return f"{addr}[{self.offset_bits - 1}:{int(math.log2(self.elem_bytes))}]"


_FLOATS = (ast.BasicType_Float32, ast.BasicType_Float64, ast.BasicType_BFloat16)


def _bits(ty: ast.BasicType) -> int:
    if isinstance(ty, ast.BasicType_Float32):
        return 32
    if isinstance(ty, ast.BasicType_Float64):
        return 64
    if isinstance(ty, (ast.BasicType_ApFixed, ast.BasicType_ApUFixed, ast.BasicType_BFloat16)):
        return ty.width
    raise NotImplementedError(f"no SPM element width for {ty}")


def _attr_list(name: str, attrs: dict, key: str) -> list[int]:
    value = attrs.get(key)
    items = value.elements if isinstance(value, ast.ArrayLiteralExpr) else None
    if not items or not all(isinstance(e, ast.LitExpr)
                            and isinstance(e.literal.lit, ast.LiteralInner_Fixed) for e in items):
        raise ValueError(f"static {name}: #[{key}([...])] must list integer literals")
    return [e.literal.lit.value for e in items]


def _banks(name: str, ty: ast.DataType_Array, attrs: dict) -> tuple[int, bool]:
    """Bank count and cyclic flag from the partition attributes, (1, True) if none."""
    keys = ("partition_dim_array", "partition_factor_array", "partition_cyclic_array")
    present = [key for key in attrs if key.startswith("partition")]
    if not present:
        return 1, True
    if sorted(present) != sorted(keys):
        raise ValueError(f"static {name}: give all of {', '.join(keys)}, not only {present}")
    dims, factors, cyclic = (_attr_list(name, attrs, key) for key in keys)
    if len(ty.dimensions) != 1 or dims != [0] or len(factors) != 1 or len(cyclic) != 1:
        raise NotImplementedError(f"static {name}: only a partition of the single dimension "
                                  f"of a 1-D static (dim [0])")
    elements, factor = ty.dimensions[0], factors[0]
    if factor < 1 or elements % factor or cyclic[0] not in (0, 1):
        raise ValueError(f"static {name}: partition factor {factor} must divide {elements} "
                         "and partition_cyclic_array be [0] or [1]")
    return factor, bool(cyclic[0])


def build_layout(proc: ast.Proc) -> SpmLayout:
    arrays = [(name, s.ty, s.attrs) for name, s in proc.statics.items()
              if isinstance(s.ty, ast.DataType_Array)]
    if not arrays:
        raise ValueError("proc declares no static arrays")
    spms = [Spm(name, index, math.prod(ty.dimensions), _bits(ty.element_type),
                isinstance(ty.element_type, _FLOATS),
                isinstance(ty.element_type, ast.BasicType_ApFixed),
                *_banks(name, ty, attrs))
            for index, (name, ty, attrs) in enumerate(arrays)]
    if {spm.elem_bits for spm in spms} != {32}:
        # Burst lanes are carved from the beat in element-sized slices.
        raise NotImplementedError("only 32-bit SPM elements are supported")
    region = max(spm.elements * spm.elem_bits // 8 for spm in spms)
    return SpmLayout(spms, math.ceil(math.log2(region)), max(1, math.ceil(math.log2(len(spms)))))


def generate_spm(layout: SpmLayout) -> str:
    lines = [f"// SPM {spm.name}: DMA byte base 0x{layout.base(spm.name):x}" for spm in layout.spms]
    lines += [f"reg [{spm.elem_bits - 1}:0] {spm.signal} [0:{spm.elements - 1}];"
              if spm.banks == 1 else
              f"reg [{spm.elem_bits - 1}:0] {spm.signal} [0:{spm.banks - 1}][0:{spm.bank_size - 1}];"
              for spm in layout.spms]
    return "\n".join(lines) + "\n"


def generate_spm_writes(layout: SpmLayout, writes: list[SpmWrite]) -> str:
    out = []
    for spm in layout.spms:
        mine = [w for w in writes if w.spm == spm.name]
        if not mine:
            continue
        body = "\n".join(f"  if ({w.enable})\n    {spm.ref(w.index)} <= {w.data};" for w in mine)
        out.append(f"// SPM {spm.name}: every write\nalways @(posedge clk) begin\n{body}\nend")
        origins = sorted({w.origin for w in mine})
        if len(origins) > 1:
            active = ", ".join("(" + " || ".join(f"({w.enable})" for w in mine if w.origin == o) + ")"
                               for o in origins)
            out.append(f"`ifndef SYNTHESIS\nalways @(posedge clk) if (!rst)\n"
                       f"  assert($countones({{{active}}}) <= 1);  // {', '.join(origins)}\n`endif")
    return "\n".join(out) + "\n" if out else ""
