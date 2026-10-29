"""sharing.check_unshared rejects Vitis binding several HardFloat calls to one
blackbox instance, read from the verbose binding reports.

The lines below are cut from the Vitis 2024.1 report of a row-sum loop (II 3)
whose sum and sum of squares shared one adder; that RTL computed the sum wrong.
"""
from pathlib import Path

import pytest

from cadl_frontend.allo_kernel.sharing import check_unshared, shared_blackboxes

CALL = ('ST_{st} : Operation {n} [{k}/4] (7.30ns)   --->   "%{v} = call i32 @{f}, i32 %a, i32 %b" '
        '[./blackbox/hardfloat_model.h:23->kernel.cpp:36]   --->   Operation {n} \'call\' \'{v}\'\n')
TABLE = "        |{i}  |{f}  |        3|        3|  30.000 ns|  30.000 ns|    1|    1|  yes(flp)|\n"


def _report(calls, instances):
    """Each call in all four states of its latency, then the instance table."""
    lines = [CALL.format(st=4 * j + s, n=10 * j + s, k=4 - s, v=v, f=f)
             for j, (v, f) in enumerate(calls) for s in range(4)]
    return "".join(lines) + "".join(TABLE.format(i=i, f=f) for i, f in instances)


def test_two_calls_on_one_instance_reported():
    report = _report([("v11", "hf_add_f32_rtl"), ("v15", "hf_add_f32_rtl"), ("v14", "hf_mul_f32_rtl")],
                     [("grp_hf_add_f32_rtl_fu_114", "hf_add_f32_rtl"),
                      ("grp_hf_mul_f32_rtl_fu_120", "hf_mul_f32_rtl")])
    assert shared_blackboxes(report) == {
        "hf_add_f32_rtl": ({"v11", "v15"}, {"grp_hf_add_f32_rtl_fu_114"})}


def test_call_sites_and_unrolled_copies_on_own_instances_pass():
    calls = [("v11", "hf_add_f32_s0_rtl"), ("v15", "hf_add_f32_s1_rtl"),
             ("v20", "hf_mul_f32_s0_rtl"), ("v20_1", "hf_mul_f32_s0_rtl")]
    instances = [("grp_hf_add_f32_s0_rtl_fu_10", "hf_add_f32_s0_rtl"),
                 ("grp_hf_add_f32_s1_rtl_fu_11", "hf_add_f32_s1_rtl"),
                 ("grp_hf_mul_f32_s0_rtl_fu_12", "hf_mul_f32_s0_rtl"),
                 ("grp_hf_mul_f32_s0_rtl_fu_13", "hf_mul_f32_s0_rtl")]
    assert shared_blackboxes(_report(calls, instances)) == {}


def test_check_unshared_names_module_and_calls(tmp_path):
    (tmp_path / "top.verbose.bind.rpt").write_text("no blackbox calls\n")
    (tmp_path / "top_Pipeline_sum.verbose.bind.rpt").write_text(_report(
        [("v20", "hf_mul_f32_s0_rtl"), ("v20_1", "hf_mul_f32_s0_rtl")],
        [("grp_hf_mul_f32_s0_rtl_fu_12", "hf_mul_f32_s0_rtl")]))
    with pytest.raises(RuntimeError, match=r"top_Pipeline_sum: calls v20, v20_1 of "
                                           r"hf_mul_f32_s0_rtl share 1 instance\(s\)"):
        check_unshared(tmp_path)
    (tmp_path / "top_Pipeline_sum.verbose.bind.rpt").unlink()
    check_unshared(tmp_path)


def test_missing_reports_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="no Vitis binding reports"):
        check_unshared(Path(tmp_path))
