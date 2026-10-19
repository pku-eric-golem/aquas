"""Subtasks A/E, ports and HLS instantiation against the allo_manual golden."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cadl_frontend.allo_sv_gen.burst import generate_burst
from cadl_frontend.allo_sv_gen.instantiate import (KernelParam, classify, generate_hls_instance,
                                                   load_rtl_ports)
from cadl_frontend.allo_sv_gen.ports import (assemble_module, generate_ports, load_ports,
                                             tie_off_unused)
from cadl_frontend.allo_sv_gen.spm import build_layout, generate_spm
from cadl_frontend.parser import parse_proc

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "examples/allo_manual"
GOLDEN = (MANUAL / "main.sv").read_text()
PARAMS = [KernelParam("A", "ap_memory"), KernelParam("BT", "ap_memory"),
          KernelParam("C", "ap_memory"), KernelParam("use_c", "ap_none"),
          KernelParam("D", "ap_memory")]


@pytest.fixture(scope="module")
def proc():
    return parse_proc((ROOT / "examples/allo/gemm.cadl").read_text())


@pytest.fixture(scope="module")
def ports():
    return load_ports()


@pytest.fixture(scope="module")
def instance():
    rtl = MANUAL / "rtl/gemm_fp32.v"
    return classify("gemm_fp32", load_rtl_ports(rtl), PARAMS)


def test_spm_layout_matches_manual_abi(proc):
    layout = build_layout(proc)
    assert [s.name for s in layout.spms] == ["matA", "matBT", "matCD"]
    assert all(s.elements == 131072 and s.elem_bits == 32 for s in layout.spms)
    # main.sv: loadmat funct 0/1/2 add 0, 0x80000, 0x100000; select = addr[20:19].
    assert [layout.base(s.name) for s in layout.spms] == [0, 0x80000, 0x100000]
    assert layout.select("a") == "a[20:19]" and layout.element("a") == "a[18:2]"
    assert generate_spm(layout).count("[0:131071];") == 3


def test_ports_match_golden(ports):
    golden = GOLDEN[GOLDEN.index("module main("):GOLDEN.index(");") + 3]
    assert generate_ports(ports) == golden
    assert len(ports) > 60


def test_tie_off_only_unassigned(ports):
    ties = tie_off_unused("assign burst_write_ready = !rst;\n", ports)
    assert "burst_write_ready" not in ties
    assert "assign hella_cmd_hella_cmd_to_bus_enable = '0;" in ties


def test_hls_instance_matches_golden(instance):
    assert [(m.param, m.index, sorted(m.signals)) for m in instance.memories] == [
        ("A", 0, ["address", "ce", "q"]), ("BT", 0, ["address", "ce", "q"]),
        ("C", 0, ["address", "ce", "q"]), ("D", 0, ["address", "ce", "d", "we"])]
    assert list(instance.scalars) == ["use_c"]
    code = generate_hls_instance(instance)
    golden = GOLDEN[GOLDEN.index("gemm_fp32 accelerator("):]
    golden_ports = re.findall(r"^\s+\.(\w+)\(", golden[:golden.index(");")], re.M)
    assert re.findall(r"^\s+\.(\w+)\(", code, re.M) == golden_ports
    assert "reg [31:0] hls_A_q0;" in code and "wire [31:0] hls_D_d0;" in code


def test_unknown_hls_port_rejected():
    with pytest.raises(ValueError, match="unexpected HLS port"):
        classify("k", load_rtl_ports(MANUAL / "rtl/gemm_fp32.v"), PARAMS[:3])


def _skeleton(proc, ports, instance) -> str:
    layout = build_layout(proc)
    body = generate_spm(layout) + generate_burst(layout, ports) + generate_hls_instance(instance)
    return assemble_module(ports, body)


def _simulate(tmp_path: Path, name: str, main_sv: str) -> str:
    work = tmp_path / name
    work.mkdir()
    (work / "main.sv").write_text(main_sv)
    rtl = sorted(str(p) for p in (MANUAL / "rtl").glob("*.v"))
    subprocess.run(["verilator", "--binary", "-j", "4", "-Wno-fatal", "-Wno-lint",
                    "-Wno-style", "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj",
                    str(Path(__file__).with_name("burst_tb.sv")), "main.sv", *rtl],
                   cwd=work, check=True, capture_output=True)
    return subprocess.run([str(work / "obj/Vtb")], cwd=work, check=True,
                          capture_output=True, text=True).stdout


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_burst_behaviour_matches_golden(tmp_path, proc, ports, instance):
    assert "BURST PASS beats=12" in _simulate(tmp_path, "golden", GOLDEN)
    assert "BURST PASS beats=12" in _simulate(tmp_path, "generated",
                                              _skeleton(proc, ports, instance))
