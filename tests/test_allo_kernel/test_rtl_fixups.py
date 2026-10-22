"""rtl_fixups drops Vitis' dangling blackbox ap_ready pins and rejects completion stalls."""
from pathlib import Path

import pytest

from cadl_frontend.allo_kernel.rtl_fixups import fix_rtl

ROOT = Path(__file__).resolve().parents[2]
# Raw Vitis output of the archived manual flow, whose blackboxes declared an
# extra ap_ready output: its MAC pipeline waits for the call to complete.
ARCHIVED_RAW = ROOT / "examples/allo_manual/hls_native/out.prj/solution1/syn/verilog"

# Shape of a Vitis systolic PE (allo.library.systolic) with our blackboxes: the
# MAC pipeline stalls only on its FIFOs and on ap_done_reg.
PE = """module systolic_PE_kernel_3_3();
wire    grp_hf_add_f32_rtl_fu_99_ap_ready;
hf_add_f32_rtl grp_hf_add_f32_rtl_fu_99(
    .ap_clk(ap_clk),
    .result(grp_hf_add_f32_rtl_fu_99_ap_return),
    .ap_ready(grp_hf_add_f32_rtl_fu_99_ap_ready)
);
always @ (*) begin
    ap_block_pp0_stage0_11001 = ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_block_state1_pp0_stage0_iter0));
end
always @ (*) begin
    ap_block_pp0_stage1_subdone = ((ap_done_reg == 1'b1) | ((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_block_state2_pp0_stage1_iter0)));
end
assign ap_block_pp0_stage2_11001 = ~(1'b1 == 1'b1);
endmodule
"""
LOOP_ONLY = "((icmp_ln36_reg_193_pp0_iter2_reg == 1'd0) & (ap_enable_reg_pp0_iter3 == 1'b1)) | "


def _fix(tmp_path, text):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/systolic_PE_kernel_3_3.v").write_text(text)
    return fix_rtl(tmp_path / "src", tmp_path / "out")


def test_dangling_ready_pin_dropped_rest_untouched(tmp_path):
    record = _fix(tmp_path, PE)
    assert record == [dict(file="systolic_PE_kernel_3_3.v", fix="drop dangling ap_ready pin",
                           signal="grp_hf_add_f32_rtl_fu_99_ap_ready")]
    fixed = (tmp_path / "out/systolic_PE_kernel_3_3.v").read_text()
    assert ".ap_ready(" not in fixed
    assert "    .result(grp_hf_add_f32_rtl_fu_99_ap_return)\n);" in fixed
    for line in PE.splitlines()[7:]:  # FIFO, done and constant stalls are kept as is
        assert line in fixed


def test_ready_pin_with_a_reader_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unexpected ap_ready pins"):
        _fix(tmp_path, PE.replace("endmodule", "assign x = grp_hf_add_f32_rtl_fu_99_ap_ready;\nendmodule"))


def test_completion_stall_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="blackbox completion stall in ap_block_pp0_stage0_11001"):
        _fix(tmp_path, PE.replace("ap_block_pp0_stage0_11001 = (", "ap_block_pp0_stage0_11001 = (" + LOOP_ONLY))


def test_archived_manual_blackbox_shape_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="blackbox completion stall"):
        fix_rtl(ARCHIVED_RAW, tmp_path)
