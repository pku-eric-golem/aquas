"""The RTL interface of a Vitis HLS top module, read back from its port list.

`classify` groups the synthesized ports by kernel parameter: an ap_memory
array becomes one MemoryPort per port index (A_address0/A_ce0/A_q0, ...), an
ap_none scalar maps to its single port, and an ap_vld scalar return to its
data port, with the strobe at `<name>_ap_vld`.

A top array partitioned on one dimension becomes one memory per bank, with the
bank number after the name (A_0_address0, A_1_address0, ...). Each bank holds
the elements whose index on that dimension falls in it, row-major over the
bank's own shape; `bank_element` maps a bank address back to the element.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

CONTROL = ("ap_start", "ap_done", "ap_idle", "ap_ready")
ROLES = ("address", "ce", "we", "q", "d")


@dataclass(frozen=True)
class Partition:
    """An Allo/Vitis array partition of one dimension (1-based) into `factor` banks."""
    kind: str  # "cyclic", "block" or "complete"
    dim: int
    factor: int

    def bank_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        dims = list(shape)
        dims[self.dim - 1] //= self.factor
        return tuple(dims)


def bank_element(shape: tuple[int, ...], partition: Partition | None, bank: int, addr: int) -> int:
    """Row-major element of `shape` that address `addr` of bank `bank` holds."""
    if partition is None:
        return addr
    k = partition.dim - 1
    inner = math.prod(shape[k + 1:])
    local = shape[k] // partition.factor
    outer, rest = divmod(addr, inner * local)
    j, r = divmod(rest, inner)
    i = bank + partition.factor * j if partition.kind == "cyclic" else bank * local + j
    return (outer * shape[k] + i) * inner + r


@dataclass
class KernelParam:
    name: str
    kind: str  # "ap_memory", "ap_none" (scalar argument) or "ap_vld" (scalar return)
    dtype: str | None = None  # Allo element type, e.g. "f32", "ui8"
    bits: int | None = None
    shape: tuple[int, ...] = ()  # () for scalars
    partition: Partition | None = None

    @property
    def banks(self) -> int:
        return self.partition.factor if self.partition else 1


@dataclass
class HlsPort:
    name: str
    direction: str
    bits: int


@dataclass
class MemoryPort:
    """One ap_memory port group, e.g. A_address0/A_ce0/A_q0, or A_1_address0 of bank 1."""
    param: str
    index: int
    bank: int | None = None  # None unless the array is partitioned
    signals: dict[str, HlsPort] = field(default_factory=dict)  # role -> port

    def port(self, role: str) -> str:
        bank = "" if self.bank is None else f"{self.bank}_"
        return f"{self.param}_{bank}{role}{self.index}"


@dataclass
class HlsInstance:
    top: str
    ports: list[HlsPort]
    memories: list[MemoryPort]
    scalars: dict[str, HlsPort]
    results: dict[str, HlsPort] = field(default_factory=dict)  # ap_vld returns
    params: list[KernelParam] = field(default_factory=list)
    returns: list[KernelParam] = field(default_factory=list)

    @staticmethod
    def signal(port: str) -> str:
        return f"hls_{port}"

    @staticmethod
    def valid(result: str) -> str:
        return f"{result}_ap_vld"


def load_rtl_ports(verilog: Path) -> list[HlsPort]:
    found = re.findall(r"^(input|output)\s+(?:wire\s+)?(?:\[(\d+):0\]\s+)?(\w+);",
                       verilog.read_text(), re.M)
    if not found:
        raise ValueError(f"no ports in {verilog}")
    return [HlsPort(name, direction, int(high) + 1 if high else 1)
            for direction, high, name in found]


def classify(top: str, rtl_ports: list[HlsPort], params: list[KernelParam],
             returns: list[KernelParam] = ()) -> HlsInstance:
    memories: dict[tuple[str, int | None, int], MemoryPort] = {}
    scalars, results, strobes = {}, {}, set()
    kinds = {p.name: p.kind for p in [*params, *returns]}
    banked = {p.name: p.banks for p in [*params, *returns] if p.partition}
    for port in rtl_ports:
        if port.name in ("ap_clk", "ap_rst") or port.name in CONTROL:
            continue
        if kinds.get(port.name) == "ap_none":
            scalars[port.name] = port
            continue
        if kinds.get(port.name) == "ap_vld":
            results[port.name] = port
            continue
        if port.name.endswith("_ap_vld") and kinds.get(port.name[:-7]) == "ap_vld":
            strobes.add(port.name[:-7])
            continue
        match = re.fullmatch(rf"(\w+?)(?:_(\d+))?_({'|'.join(ROLES)})(\d+)", port.name)
        if match and match[2] is not None and match[1] not in banked:
            match = re.fullmatch(rf"(\w+)()_({'|'.join(ROLES)})(\d+)", port.name)
        if not match or kinds.get(match[1]) != "ap_memory" \
                or (match[1] in banked) != bool(match[2]) \
                or (match[2] and int(match[2]) >= banked[match[1]]):
            raise ValueError(f"unexpected HLS port {port.name} on {top}")
        key = (match[1], int(match[2]) if match[2] else None, int(match[4]))
        memories.setdefault(key, MemoryPort(match[1], key[2], key[1])).signals[match[3]] = port
    missing = [p.name for p in [*params, *returns]
               if p.name not in scalars and p.name not in results
               and all(m.param != p.name for m in memories.values())]
    missing += [f"{name} bank {b}" for name, banks in banked.items() for b in range(banks)
                if all((m.param, m.bank) != (name, b) for m in memories.values())]
    missing += [HlsInstance.valid(name) for name in results if name not in strobes]
    if missing:
        raise ValueError(f"parameters without RTL ports on {top}: {missing}")
    return HlsInstance(top, rtl_ports, list(memories.values()), scalars, results,
                       list(params), list(returns))
