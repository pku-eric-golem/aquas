"""Subtask A: scratchpad (SPM) layout and storage for CADL statics.

Each static array gets its own byte-address region of equal power-of-two size,
so the DMA side selects a scratchpad with a bit slice of the address (GEMM:
three 512 KiB regions, select = addr[20:19], element = addr[18:2]).
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

    @property
    def signal(self) -> str:
        return f"spm_{self.name}"


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


def build_layout(proc: ast.Proc) -> SpmLayout:
    arrays = [(name, s.ty) for name, s in proc.statics.items()
              if isinstance(s.ty, ast.DataType_Array)]
    if not arrays:
        raise ValueError("proc declares no static arrays")
    spms = [Spm(name, index, math.prod(ty.dimensions), _bits(ty.element_type),
                isinstance(ty.element_type, _FLOATS))
            for index, (name, ty) in enumerate(arrays)]
    if {spm.elem_bits for spm in spms} != {32}:
        # Burst lanes are carved from the beat in element-sized slices.
        raise NotImplementedError("only 32-bit SPM elements are supported")
    region = max(spm.elements * spm.elem_bits // 8 for spm in spms)
    return SpmLayout(spms, math.ceil(math.log2(region)), max(1, math.ceil(math.log2(len(spms)))))


def generate_spm(layout: SpmLayout) -> str:
    lines = [f"// SPM {spm.name}: DMA byte base 0x{layout.base(spm.name):x}" for spm in layout.spms]
    lines += [f"reg [{spm.elem_bits - 1}:0] {spm.signal} [0:{spm.elements - 1}];"
              for spm in layout.spms]
    return "\n".join(lines) + "\n"
