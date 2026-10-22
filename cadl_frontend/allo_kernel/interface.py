"""The RTL interface of a Vitis HLS top module, read back from its port list.

`classify` groups the synthesized ports by kernel parameter: an ap_memory
array becomes one MemoryPort per port index (A_address0/A_ce0/A_q0, ...), an
ap_none scalar maps to its single port.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CONTROL = ("ap_start", "ap_done", "ap_idle", "ap_ready")


@dataclass
class KernelParam:
    name: str
    kind: str  # "ap_memory" or "ap_none"
    dtype: str | None = None  # Allo element type, e.g. "f32", "ui8"
    bits: int | None = None
    shape: tuple[int, ...] = ()  # () for scalars


@dataclass
class HlsPort:
    name: str
    direction: str
    bits: int


@dataclass
class MemoryPort:
    """One ap_memory port group, e.g. A_address0/A_ce0/A_q0."""
    param: str
    index: int
    signals: dict[str, HlsPort] = field(default_factory=dict)  # role -> port


@dataclass
class HlsInstance:
    top: str
    ports: list[HlsPort]
    memories: list[MemoryPort]
    scalars: dict[str, HlsPort]

    @staticmethod
    def signal(port: str) -> str:
        return f"hls_{port}"


def load_rtl_ports(verilog: Path) -> list[HlsPort]:
    found = re.findall(r"^(input|output)\s+(?:wire\s+)?(?:\[(\d+):0\]\s+)?(\w+);",
                       verilog.read_text(), re.M)
    if not found:
        raise ValueError(f"no ports in {verilog}")
    return [HlsPort(name, direction, int(high) + 1 if high else 1)
            for direction, high, name in found]


def classify(top: str, rtl_ports: list[HlsPort], params: list[KernelParam]) -> HlsInstance:
    memories: dict[tuple[str, int], MemoryPort] = {}
    scalars = {}
    kinds = {p.name: p.kind for p in params}
    for port in rtl_ports:
        if port.name in ("ap_clk", "ap_rst") or port.name in CONTROL:
            continue
        if kinds.get(port.name) == "ap_none":
            scalars[port.name] = port
            continue
        match = re.fullmatch(r"(\w+)_(address|ce|we|q|d)(\d+)", port.name)
        if not match or kinds.get(match[1]) != "ap_memory":
            raise ValueError(f"unexpected HLS port {port.name} on {top}")
        key = (match[1], int(match[3]))
        memories.setdefault(key, MemoryPort(*key)).signals[match[2]] = port
    missing = [p.name for p in params
               if p.name not in scalars and all(m.param != p.name for m in memories.values())]
    if missing:
        raise ValueError(f"parameters without RTL ports on {top}: {missing}")
    return HlsInstance(top, rtl_ports, list(memories.values()), scalars)
