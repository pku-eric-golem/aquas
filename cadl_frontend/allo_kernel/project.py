"""The HLS project of a kernel build as plain data, readable without Allo.

hls.py creates it; build.py stores it as project.json next to kernel.json, so
that a process that never imports Allo (isolated.py) can take over the
results of a build that ran in another one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .interface import KernelParam, Partition


@dataclass
class HLSProject:
    top: str
    directory: Path
    params: list[KernelParam]
    float_calls: dict[str, int]
    memory_latency: int  # read latency of every ap_memory port, in cycles
    returns: list[KernelParam] = field(default_factory=list)  # in Python return order

    def to_json(self) -> dict:
        return {"top": self.top, "directory": str(self.directory),
                "params": [param_to_json(p) for p in self.params],
                "returns": [param_to_json(p) for p in self.returns],
                "float_calls": self.float_calls, "memory_latency": self.memory_latency}

    @classmethod
    def from_json(cls, data: dict) -> "HLSProject":
        return cls(data["top"], Path(data["directory"]),
                   [param_from_json(p) for p in data["params"]], data["float_calls"],
                   data["memory_latency"], [param_from_json(p) for p in data["returns"]])


def param_to_json(param: KernelParam) -> dict:
    part = param.partition
    return {"name": param.name, "kind": param.kind, "dtype": param.dtype, "bits": param.bits,
            "shape": list(param.shape),
            "partition": part and {"kind": part.kind, "dim": part.dim, "factor": part.factor}}


def param_from_json(data: dict) -> KernelParam:
    part = data["partition"]
    return KernelParam(data["name"], data["kind"], data["dtype"], data["bits"], tuple(data["shape"]),
                       part and Partition(part["kind"], part["dim"], part["factor"]))
