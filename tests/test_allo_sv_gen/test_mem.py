"""Single CPU memory accesses `_mem[addr]` through the HellaCache port.

The bench models the port the way ApsWrapper.scala feeds Rocket: requests are
taken when `ready` (random here), responses come back 1-5 cycles later, and a
write answers too. It checks one request in flight and the request fields;
memory is compared with a Python model after the run.
"""
import shutil
import subprocess

import pytest

from cadl_frontend.allo_sv_gen.fsm import build_fsm, generate_fsm
from cadl_frontend.allo_sv_gen.ports import assemble_module, load_ports
from cadl_frontend.parser import parse_proc
from test_fsm import SOURCE as FSM_SOURCE

WORDS = 1024
SOURCE = """
static total: u32 = 0;

#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype ld(rs1: u5, rs2: u5, rd: u5) {
    _irf[rd] = _mem[_irf[rs1]];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000001)]
rtype st(rs1: u5, rs2: u5, rd: u5) {
    let a: u32 = _irf[rs1];
    let v: u32 = _irf[rs2];
    _mem[a] = v;
    _irf[rd] = 0;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000010)]
rtype add2(rs1: u5, rs2: u5, rd: u5) {
    let a: u32 = _irf[rs1];
    _irf[rd] = _mem[a] + _mem[a + 4];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000011)]
rtype copy2x(rs1: u5, rs2: u5, rd: u5) {
    let src: u32 = _irf[rs1];
    let dst: u32 = _irf[rs2];
    with
        k: u32 = (0, k_next)
    do {
        _mem[dst + k * 4] = _mem[src + k * 4] * 2;
        let k_next: u32 = k + 1;
    } while (k_next < 4);
    total = total + 4;
    _irf[rd] = total;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000100)]
rtype chase(rs1: u5, rs2: u5, rd: u5) {
    let p: u32 = _irf[rs1];
    _irf[rd] = _mem[_mem[p]];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000101)]
rtype clamp(rs1: u5, rs2: u5, rd: u5) {
    let a: u32 = _irf[rs1];
    if _mem[a] > 100 {
        _mem[a] = 100;
        _irf[rd] = 1;
    } else {
        _irf[rd] = 0;
    }
}
"""

FUNCT = {"ld": 0, "st": 1, "add2": 2, "copy2x": 3, "chase": 4, "clamp": 5}
SEQUENCE = [("st", 0x100, 0x12345678), ("ld", 0x100, 0), ("add2", 0x200, 0),
            ("copy2x", 0x300, 0x400), ("copy2x", 0x400, 0x500), ("st", 0x600, 0x100),
            ("chase", 0x600, 0), ("st", 0x700, 250), ("clamp", 0x700, 0), ("clamp", 0x700, 0),
            ("ld", 0x700, 0)]


class Model:
    def __init__(self):
        self.mem = [(i * 37 + 11) & 0xFFFFFFFF for i in range(WORDS)]
        self.total = 0

    def ld(self, a, _):
        return self.mem[a // 4]

    def st(self, a, v):
        self.mem[a // 4] = v
        return 0

    def add2(self, a, _):
        return self.mem[a // 4] + self.mem[a // 4 + 1]

    def copy2x(self, src, dst):
        for k in range(4):
            self.mem[dst // 4 + k] = self.mem[src // 4 + k] * 2 & 0xFFFFFFFF
        self.total += 4
        return self.total

    def chase(self, p, _):
        return self.mem[self.mem[p // 4] // 4]

    def clamp(self, a, _):
        if self.mem[a // 4] > 100:
            self.mem[a // 4] = 100
            return 1
        return 0


TB = """module tb;
  reg clk = 0, rst = 1;
  always #1 clk = ~clk;
  reg cmd_enable = 0;
  reg [6:0] opcode = 0, funct = 0;
  reg [31:0] rs1data = 0, rs2data = 0;
  wire cmd_ready, resp_enable;
  wire [31:0] resp_data;

  reg [31:0] mem [0:@WORDS@-1];
  wire h_en, h_signed, h_phys;
  wire [31:0] h_addr, h_data;
  wire [7:0] h_tag;
  wire [4:0] h_cmd;
  wire [1:0] h_size;
  wire [3:0] h_mask;
  reg [15:0] lfsr = 16'hace1;
  wire h_ready = lfsr[0] | lfsr[3];
  reg pending = 0, p_write = 0, r_en = 0;
  reg [31:0] p_addr = 0, p_data = 0, r_data = 0;
  integer delay = 0, errors = 0, cycles, requests = 0;

  main dut(
    .clk(clk), .rst(rst),
    .rocc_cmd_enable(cmd_enable), .rocc_cmd_ready(cmd_ready),
    .rocc_cmd_rocc_cmd_opcode(opcode), .rocc_cmd_rocc_cmd_funct(funct),
    .rocc_cmd_rocc_cmd_rd(5'd1), .rocc_cmd_rocc_cmd_xd(1'b1),
    .rocc_cmd_rocc_cmd_rs1data(rs1data), .rocc_cmd_rocc_cmd_rs2data(rs2data),
    .rocc_resp_rocc_resp_to_bus_enable(resp_enable),
    .rocc_resp_rocc_resp_to_bus_ready(1'b1),
    .rocc_resp_rocc_resp_to_bus_result_rddata(resp_data),
    .hella_cmd_hella_cmd_to_bus_enable(h_en), .hella_cmd_hella_cmd_to_bus_ready(h_ready),
    .hella_cmd_hella_cmd_to_bus_cmd_addr(h_addr), .hella_cmd_hella_cmd_to_bus_cmd_tag(h_tag),
    .hella_cmd_hella_cmd_to_bus_cmd_cmd(h_cmd), .hella_cmd_hella_cmd_to_bus_cmd_size(h_size),
    .hella_cmd_hella_cmd_to_bus_cmd_signed(h_signed), .hella_cmd_hella_cmd_to_bus_cmd_phys(h_phys),
    .hella_cmd_hella_cmd_to_bus_cmd_data(h_data), .hella_cmd_hella_cmd_to_bus_cmd_mask(h_mask),
    .hella_resp_enable(r_en), .hella_resp_hella_resp_data(r_data));

  always @(posedge clk) begin
    lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
    r_en <= 0;
    if (h_en && h_ready) begin
      if (pending) begin $display("HELLA FAIL: second request in flight"); errors = errors + 1; end
      if (h_size !== 2 || h_signed || h_phys || h_cmd > 1 || h_addr[1:0] != 0) begin
        $display("HELLA FAIL: request fields cmd=%h size=%h addr=%h", h_cmd, h_size, h_addr);
        errors = errors + 1;
      end
      pending <= 1; p_write <= h_cmd[0]; p_addr <= h_addr; p_data <= h_data;
      delay <= 1 + lfsr[2:0] % 5;
      requests = requests + 1;
    end else if (pending) begin
      if (delay > 1) delay <= delay - 1;
      else begin
        pending <= 0;
        r_en <= 1;
        r_data <= p_write ? 32'hbad0bad0 : mem[p_addr >> 2];
        if (p_write) mem[p_addr >> 2] <= p_data;
      end
    end
  end

  task automatic issue(input [6:0] f, input [31:0] a, input [31:0] b, input [31:0] want);
    begin
      while (!cmd_ready) @(negedge clk);
      cmd_enable = 1; opcode = 7'h0b; funct = f; rs1data = a; rs2data = b;
      @(negedge clk);
      cmd_enable = 0; rs1data = 32'hdead_beef; rs2data = 32'hdead_beef;
      cycles = 0;
      while (!resp_enable && cycles < 10000) begin @(negedge clk); cycles = cycles + 1; end
      if (!resp_enable || resp_data !== want) begin
        $display("ROCC FAIL f=%h got=%h want=%h", f, resp_data, want);
        errors = errors + 1;
      end
      if (pending) begin $display("HELLA FAIL: answered with a request in flight"); errors = errors + 1; end
      @(negedge clk);
    end
  endtask
  initial begin
    $readmemh("mem.hex", mem);
    repeat (3) @(negedge clk);
    rst = 0;
@STEPS@
    $writememh("out.hex", mem);
    if (errors == 0) $display("MEM PASS requests=%0d", requests);
    $finish;
  end
endmodule
"""


def _main(source: str) -> str:
    ports = load_ports()
    fsm = build_fsm(parse_proc(source), {})
    return assemble_module(ports, generate_fsm(fsm, {p.name for p in ports}))


def test_request_fields_and_states():
    sv = _main(SOURCE)
    assert "assign hella_cmd_hella_cmd_to_bus_cmd_cmd = {4'd0, mem_write};" in sv
    assert "assign hella_cmd_hella_cmd_to_bus_enable = !rst && (" in sv
    assert "S_st_mem_resp" in sv  # the write waits for its response
    # Without single accesses the port stays tied off, as before.
    unused = _main(FSM_SOURCE)
    assert "assign hella_cmd_hella_cmd_to_bus_enable = '0;" in unused and "mem_write" not in unused


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_mem_simulation(tmp_path):
    model = Model()
    (tmp_path / "mem.hex").write_text("\n".join(f"{w:08x}" for w in model.mem) + "\n")
    steps = []
    for name, a, b in SEQUENCE:
        want = getattr(model, name)(a, b) & 0xFFFFFFFF
        steps.append(f"    issue(7'h{FUNCT[name]:02x}, 32'h{a:08x}, 32'h{b:08x}, 32'h{want:08x});")
    (tmp_path / "main.sv").write_text(_main(SOURCE))
    (tmp_path / "tb.sv").write_text(TB.replace("@WORDS@", str(WORDS)).replace("@STEPS@", "\n".join(steps)))
    subprocess.run(["verilator", "--binary", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv", "main.sv"],
                   cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    assert "MEM PASS" in out, out
    got = [int(w, 16) for w in (tmp_path / "out.hex").read_text().split() if not w.startswith("@")]
    assert got == model.mem
