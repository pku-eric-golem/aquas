"""Instantiate the Vitis HLS top module from its synthesized port list.

Every non-clock/reset port connects to a signal `hls_<port>`; inputs of the
core are declared `reg` so the invoke bridge and FSM can drive them.
"""
from __future__ import annotations

from ..allo_kernel.interface import HlsInstance


def generate_hls_instance(inst: HlsInstance, name: str = "u_hls") -> str:
    decls, conns = [], []
    for port in inst.ports:
        if port.name == "ap_clk":
            conns.append("  .ap_clk(clk)")
            continue
        if port.name == "ap_rst":
            conns.append("  .ap_rst(rst)")
            continue
        signal = inst.signal(port.name)
        kind = "reg" if port.direction == "input" else "wire"
        width = f"[{port.bits - 1}:0] " if port.bits > 1 else ""
        decls.append(f"{kind} {width}{signal};")
        conns.append(f"  .{port.name}({signal})")
    return "\n".join(decls) + f"\n\n{inst.top} {name}(\n" + ",\n".join(conns) + "\n);\n"
