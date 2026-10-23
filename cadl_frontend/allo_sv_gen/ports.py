"""The `main` module ports expected by Chipyard's APS RoCC wrapper.

The port list is read from ApsWrapper.scala so it cannot drift from the
wrapper that instantiates `main`, including the group of ports it adds for
each custom CSR (`csr_<name>_value_res0`, `csr_<name>_set_enable`, ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APS_WRAPPER = ROOT / "thirdparty/chipyard/generators/aps/src/main/scala/ApsWrapper.scala"


@dataclass
class WrapperPort:
    name: str
    direction: str  # "input" or "output"
    bits: int | None  # None for a 1-bit Bool/Clock


CSR_WIDTH = 32  # xLen of the rv32 Rocket the wrapper runs on


def load_ports(scala: Path = APS_WRAPPER, csrs: list[str] = ()) -> list[WrapperPort]:
    """The fixed ports, then one CSR port group per name in `csrs`, in order."""
    text = scala.read_text()
    fields = text[text.index("private def fields:"):text.index(") ++ csrNames")]
    found = re.findall(r'"(\w+)" -> (Input|Output)\((Clock|Bool|UInt)\((?:(\d+)\.W)?\)\)', fields)
    if not found:
        raise ValueError(f"no ports parsed from {scala}")
    ports = [WrapperPort(name, direction.lower(), int(bits) if bits else None)
             for name, direction, _, bits in found]
    start = text.index(") ++ csrNames")
    group = text[start:text.index("fields.foreach", start)]
    template = re.findall(r's"([\w${}]+)" -> (Input|Output)\((Bool|UInt)\((csrWidth\.W)?\)\)', group)
    if csrs and not template:
        raise ValueError(f"no CSR ports parsed from {scala}")
    for csr in csrs:
        ports += [WrapperPort(name.replace("${prefix}", f"csr_{csr}").replace("${name}", csr),
                              direction.lower(), CSR_WIDTH if width else None)
                  for name, direction, _, width in template]
    return ports


def width_of(ports: list[WrapperPort], name: str) -> int:
    port = next(p for p in ports if p.name == name)
    return port.bits or 1


def generate_ports(ports: list[WrapperPort], module: str = "main") -> str:
    lines = [f"  {p.direction} wire {f'[{p.bits - 1}:0] ' if p.bits else ''}{p.name}"
             for p in ports]
    return f"module {module}(\n" + ",\n".join(lines) + "\n);\n"


def tie_off_unused(body: str, ports: list[WrapperPort]) -> str:
    """Drive every output the body leaves unassigned with zero."""
    ties = [f"assign {p.name} = '0;" for p in ports
            if p.direction == "output" and not re.search(rf"assign\s+{p.name}\b", body)]
    return "\n".join(ties) + ("\n" if ties else "")


def assemble_module(ports: list[WrapperPort], body: str, module: str = "main") -> str:
    return generate_ports(ports, module) + body + tie_off_unused(body, ports) + "endmodule\n"
