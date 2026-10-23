"""CADL CSR registers on route 1 (cadl-allo-integration.md 5.C, persistent registers).

`#[csr_address(...)] register r` is a Rocket custom CSR: the bench models it
as Rocket's CSRFile does (value is the register; set loads masked sdata on
the clock edge) and also writes it from the CPU side between instructions.
"""
import shutil
import subprocess

import pytest

from cadl_frontend.allo_sv_gen.build import _csr_config
from cadl_frontend.allo_sv_gen.fsm import build_fsm, generate_fsm
from cadl_frontend.allo_sv_gen.ports import assemble_module, load_ports
from cadl_frontend.parser import parse_proc
from test_fsm import _testbench

SOURCE = """
#[csr_address(0x801)]
register gain: u32;

#[csr_address(0x802)]
register mode: u8;

#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype scale(rs1: u5, rs2: u5, rd: u5) {
    _irf[rd] = _irf[rs1] * _csr[gain];
}

// The new value is visible to later statements on the same edge.
#[opcode(7'b0001011)]
#[funct7(7'b0000001)]
rtype bump(rs1: u5, rs2: u5, rd: u5) {
    _csr[gain] = _csr[gain] + _irf[rs1];
    _irf[rd] = _csr[gain];
}

// The second write to gain waits a cycle.
#[opcode(7'b0001011)]
#[funct7(7'b0000010)]
rtype twice(rs1: u5, rs2: u5, rd: u5) {
    _csr[gain] = _irf[rs1];
    _csr[gain] = _csr[gain] * 2;
    _irf[rd] = _csr[gain];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000011)]
rtype count3(rs1: u5, rs2: u5, rd: u5) {
    with
        k: u32 = (0, k_next)
    do {
        _csr[gain] = _csr[gain] + 1;
        let k_next: u32 = k + 1;
    } while (k_next < 3);
    _irf[rd] = _csr[gain];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000100)]
rtype reset_if(rs1: u5, rs2: u5, rd: u5) {
    if _irf[rs1] > 10 {
        _csr[gain] = 0;
        _irf[rd] = 1;
    } else {
        _irf[rd] = _csr[gain];
    }
}

#[opcode(7'b0001011)]
#[funct7(7'b0000101)]
rtype set_mode(rs1: u5, rs2: u5, rd: u5) {
    _csr[mode] = _irf[rs1];
    _irf[rd] = _csr[mode] + _csr[gain];
}
"""

OP = 0x0b
# (funct7, rs1, expected rd) or ("cpu", gain, None) for a CPU write between instructions.
SEQUENCE = [("cpu", 3, None), (0, 7, 21), (1, 2, 5), (0, 1, 5), (2, 4, 8), (3, 0, 11),
            (4, 1, 11), (4, 20, 1), (0, 9, 0), ("cpu", 6, None), (0, 2, 12),
            (5, 0x1234, 0x34 + 6), (0, 1, 6)]
CSR_TB = [
    ("  wire [31:0] resp_data;\n",
     "  wire [31:0] resp_data;\n"
     "  reg [31:0] gain = 0, mode = 0;\n"
     "  wire gain_set, mode_set;\n"
     "  wire [31:0] gain_sdata, mode_sdata;\n"
     "  always @(posedge clk) begin\n"
     "    if (gain_set) gain <= gain_sdata;\n"
     "    if (mode_set) mode <= mode_sdata & 32'hff;\n"
     "  end\n"),
    ("    .rocc_resp_rocc_resp_to_bus_result_rddata(resp_data));",
     "    .rocc_resp_rocc_resp_to_bus_result_rddata(resp_data),\n"
     "    .csr_gain_value_res0(gain), .csr_gain_set_enable(gain_set),\n"
     "    .csr_gain_set_gain_sdata(gain_sdata),\n"
     "    .csr_mode_value_res0(mode), .csr_mode_set_enable(mode_set),\n"
     "    .csr_mode_set_mode_sdata(mode_sdata));"),
]


def _main(source: str = SOURCE) -> str:
    proc = parse_proc(source)
    ports = load_ports(csrs=list(proc.csrs))
    return assemble_module(ports, generate_fsm(build_fsm(proc, {}), {p.name for p in ports}))


def test_csr_ports_and_config():
    names = [p.name for p in load_ports(csrs=["gain"])][-5:]
    assert names == ["csr_gain_value_res0", "csr_gain_value_ready", "csr_gain_set_gain_sdata",
                     "csr_gain_set_ready", "csr_gain_set_enable"]
    proc = parse_proc(SOURCE)
    assert [_csr_config(n, r) for n, r in proc.csrs.items()] == [
        {"id": 0x801, "name": "gain", "mask": "0xffffffff", "init": 0},
        {"id": 0x802, "name": "mode", "mask": "0xff", "init": 0}]
    sv = _main()
    assert "assign csr_gain_set_enable = fsm_csr_gain_set;" in sv
    assert "assign csr_mode_set_mode_sdata = fsm_csr_mode_sdata;" in sv


def test_bare_csr_name_rejected():
    with pytest.raises(ValueError, match=r"read CSR gain as _csr\[gain\]"):
        _main(SOURCE.replace("_irf[rs1] * _csr[gain]", "_irf[rs1] * gain"))


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_csr_simulation(tmp_path):
    cases, steps = [], []
    tb = _testbench([])
    for old, new in CSR_TB:
        assert tb.count(old) == 1, old
        tb = tb.replace(old, new)
    for i, (funct, a, want) in enumerate(SEQUENCE):
        if funct == "cpu":
            steps.append(f"    gain = 32'd{a};  // csrw from the CPU")
        else:
            cases.append((OP, funct, a, 0, want))
            steps.append(f"    issue(7'h{OP:02x}, 7'h{funct:02x}, 32'h{a:08x}, 32'h0, "
                         f"32'h{want:08x}, {i % 32});")
    steps.append("    if (gain !== 32'd6 || mode !== 32'h34) begin\n"
                 "      $display(\"CSR FAIL gain=%0d mode=%h\", gain, mode); errors = errors + 1;\n"
                 "    end")
    done = '    if (errors == 0) $display("ROCC PASS cases=%0d", 0);'
    assert tb.count(done) == 1
    tb = tb.replace(done, "\n".join(steps) + "\n" + done.replace(", 0)", f", {len(cases)})"))
    (tmp_path / "tb.sv").write_text(tb)
    (tmp_path / "main.sv").write_text(_main())
    subprocess.run(["verilator", "--binary", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv", "main.sv"],
                   cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    assert f"ROCC PASS cases={len(cases)}" in out, out
