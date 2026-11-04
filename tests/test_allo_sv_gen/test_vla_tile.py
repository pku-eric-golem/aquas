"""vla_testbench tile engines from examples/allo/vla_tile as ISAXs, end to end.

Each example is a port of accelerate/<paper>/test/src/compact_accelerator.cadl
in vla_testbench: the original statics and write, run, read and reset
instructions, with the core block of run replaced by an invoke of the engine
written as an Allo kernel. The expected rd of every command comes from the
original CADL itself, executed by vla_testbench's own reference lowering
(test/cadl_model.py: CADL to C, gcc without fp contraction), so the port has
to match the original instruction by instruction, bit for bit.

Command streams are random but cover the run checks (bad bank counts,
contexts, continue flags and upper bits), signed zeros and subnormals, every
packed lane value, a reset in the middle and reads of every address.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from test_accept import Scenario, _simulate

ROOT = Path(__file__).resolve().parents[2]
TILE = ROOT / "examples/allo/vla_tile"
TESTBENCH = Path(os.environ.get("VLA_TESTBENCH", ROOT.parent / "vla_testbench"))

# example name -> (vla_testbench paper directory prefix, custom opcode)
PAPERS = {
    "ditpa": ("DiTPA", 0x7b),
    "specvla": ("Algorithm-Architecture", 0x5b),
    "deltoris": ("Deltoris", 0x2b),
    "vqvla": ("A Motion-Aware", 0x0b),
}
WRITE, RUN, READ, RESET = range(4)


def _paper(name):
    prefix, opcode = PAPERS[name]
    found = sorted((TESTBENCH / "accelerate").glob(prefix + "*"))
    if not found:
        pytest.skip(f"vla_testbench not found at {TESTBENCH} (set VLA_TESTBENCH)")
    return found[0] / "test", opcode


class Reference:
    """The original compact_accelerator.cadl, run by vla_testbench's lowering."""

    def __init__(self, test_dir: Path, work: Path):
        import ctypes
        import subprocess
        spec = importlib.util.spec_from_file_location("vla_cadl_model", test_dir / "cadl_model.py")
        lowering = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lowering)
        from cadl_frontend import cadl_ast, parse_proc
        source = test_dir / "src/compact_accelerator.cadl"
        proc = parse_proc(source.read_text(encoding="utf8"), str(source))
        work.mkdir(parents=True)
        (work / "functional.c").write_text(lowering.Lowering(proc, cadl_ast).emit())
        subprocess.run(["gcc", "-std=c11", "-O2", "-fPIC", "-shared", "-fno-fast-math",
                        "-ffp-contract=off", "-fwrapv", "functional.c", "-lm", "-o",
                        "functional.so"], cwd=work, check=True, capture_output=True)
        self.lib = ctypes.CDLL(str(work / "functional.so"))
        self.lib.command.argtypes = [ctypes.c_uint32] * 3
        self.lib.command.restype = ctypes.c_uint32

    def __call__(self, funct, a, b):
        return self.lib.command(funct, a, b)


def _f32(x) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _word(rng, port) -> int:
    """A value for one word of `port`, as the write instruction takes it."""
    if port["type"] == "f32":
        kind = rng.integers(8)
        if kind == 0:
            return _f32(rng.choice([0.0, -0.0]))
        if kind == 1:  # subnormal
            return int(rng.integers(1, 1 << 23)) | (int(rng.integers(2)) << 31)
        return _f32(np.float32(rng.standard_normal() * 4))
    if "packing" in port or port["type"].startswith("i"):
        return int(rng.integers(0, 1 << 32))
    return int(rng.choice([0, 1, 2, 4, 8, 5, 0xFFFFFFFF]))  # mode, precision


def _stream(name, ports, rng, runs=40):
    """(funct, rs1, rs2) commands: writes, runs, reads of every address, a reset."""
    writable = [p for p in ports if "w" in p["access"]]
    outputs = [p["base"] + i for p in ports if p["access"] == "r" for i in range(p["count"])]
    last = max(p["base"] + p["count"] for p in ports)
    sweep = [(READ, address, 0) for address in range(last + 1)]
    commands = [(RESET, 0, 0)] + sweep
    for p in writable:
        commands += [(WRITE, p["base"] + i, _word(rng, p)) for i in range(p["count"])]
    commands += sweep
    for k in range(runs):
        if k == runs // 2:
            commands.append((RESET, 0, 0))
            commands += sweep
            commands += [(WRITE, p["base"] + i, _word(rng, p))
                         for p in writable for i in range(p["count"])]
        for _ in range(int(rng.integers(0, 6))):
            p = writable[int(rng.integers(len(writable)))]
            commands.append((WRITE, p["base"] + int(rng.integers(p["count"])), _word(rng, p)))
        slot = int(rng.integers(0, 6))  # few contexts, so continues find them
        carry = int(rng.random() < 0.6)
        count = int(rng.choice([1, 2, 3, 4, 4, 0, 5]))
        rs1 = slot | carry << 4
        if rng.random() < 0.05:
            rs1 |= 1 << int(rng.integers(5, 32))  # upper bits must be zero
        commands.append((RUN, rs1, count))
        commands += [(READ, address, 0) for address in outputs]
    commands += sweep
    commands.append((WRITE, last + 7, 1))  # bad address
    return commands


class TileScenario(Scenario):
    """Scenario whose compute commands carry the engine's own opcode."""

    def __init__(self, opcode):
        super().__init__()
        self.opcode = opcode

    def commands(self):
        for name, funct, rs1, rs2, rd in self.invokes:
            yield name, self.opcode, funct, rs1, rs2, rd


def _scenario(name, tmp_path):
    test_dir, opcode = _paper(name)
    original = (test_dir / "src/compact_accelerator.cadl").read_bytes()
    port = (TILE / f"{name}.cadl").read_text()
    assert hashlib.sha256(original).hexdigest() in port, \
        f"{name}.cadl was ported from another version of the vla_testbench engine"
    ports = json.loads((test_dir / "src/system_map.json").read_text(encoding="utf8"))["ports"]
    reference = Reference(test_dir, tmp_path / "reference")
    s = TileScenario(opcode)
    labels = ("write", "run", "read", "reset")
    for funct, a, b in _stream(name, ports, np.random.default_rng(5)):
        s.invoke(labels[funct], a, b, reference(funct, a, b), funct=funct)
    return s


@pytest.mark.skipif(shutil.which("vitis_hls") is None, reason="vitis_hls not on PATH")
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not on PATH (reference model)")
@pytest.mark.parametrize("name", list(PAPERS))
def test_vla_tile_end_to_end(name, tmp_path):
    pytest.importorskip("allo")
    from cadl_frontend.allo_sv_gen.build import build
    s = _scenario(name, tmp_path)
    result = build(TILE / f"{name}.cadl", tmp_path / "build")
    log, _ = _simulate(tmp_path / "sim", result.output, s)
    assert "DMA BENCH DONE" in log, log
    responses = re.findall(r"^resp (\w+) ([0-9a-f]{8})$", log, re.M)
    want = [(c[0], c[5]) for c in s.commands()]
    got = [(n, int(v, 16)) for n, v in responses]
    assert len(got) == len(want), (len(got), len(want))
    bad = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
    commands = list(s.commands())
    assert not bad, (f"{len(bad)} of {len(want)} responses differ, first at command {bad[0]} "
                     f"{commands[bad[0]][:5]}: got {got[bad[0]][1]:08x} want {want[bad[0]][1]:08x}")
