"""Bank mapping between partitioned kernel arrays and banked SPMs, and the
single write block of each SPM (cadl-allo-integration.md 5.B, 5.C).

The kernel port list is the one Vitis gives examples/allo/accept/banked.cadl
(A cyclic on dim 2, C block on dim 1, j unrolled by 2); test_accept runs the
real one end to end.
"""
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest

from cadl_frontend.allo_kernel.interface import (HlsPort, KernelParam, Partition, bank_element,
                                                 classify)
from cadl_frontend.allo_sv_gen.invoke import bank_element_sv
from cadl_frontend.allo_sv_gen.main import generate_main
from cadl_frontend.allo_sv_gen.spm import build_layout
from cadl_frontend.parser import parse_proc

ROOT = Path(__file__).resolve().parents[2]
BANKED = (ROOT / "examples/allo/accept/banked.cadl").read_text()


def _memory(name, bits, roles, index=0):
    widths = {"address": bits, "ce": 1, "we": 1, "q": 32, "d": 32}
    direction = {"q": "input"}
    return [HlsPort(f"{name}_{role}{index}", direction.get(role, "output"), widths[role])
            for role in roles]


def banked_instance():
    ports = [HlsPort(p, "input" if p in ("ap_clk", "ap_rst", "ap_start") else "output", 1)
             for p in ("ap_clk", "ap_rst", "ap_start", "ap_done", "ap_idle", "ap_ready")]
    reads, writes = ("address", "ce", "q"), ("address", "ce", "we", "d")
    ports += _memory("A_0", 3, reads) + _memory("A_1", 3, reads)
    ports += _memory("B", 4, reads) + _memory("B", 4, reads, 1)
    for bank, index in itertools.product((0, 1), (0, 1)):
        ports += _memory(f"C_{bank}", 3, writes, index)
    params = [KernelParam("A", "ap_memory", "i32", 32, (4, 4), Partition("cyclic", 2, 2)),
              KernelParam("B", "ap_memory", "i32", 32, (4, 4)),
              KernelParam("C", "ap_memory", "i32", 32, (4, 4), Partition("block", 1, 2))]
    return classify("banked", ports, params)


@pytest.mark.parametrize("shape, partition", [
    ((16,), Partition("cyclic", 1, 4)), ((16,), Partition("block", 1, 4)),
    ((4, 4), Partition("cyclic", 2, 2)), ((4, 4), Partition("block", 1, 2)),
    ((2, 6, 4), Partition("cyclic", 2, 3)), ((2, 6, 4), Partition("block", 2, 3)),
    ((2, 6, 4), Partition("complete", 3, 4)), ((3, 4), None),
])
def test_bank_element(shape, partition):
    """Banks split the elements as Vitis does, and the RTL mapping agrees with the model."""
    banks = partition.factor if partition else 1
    size = math.prod(partition.bank_shape(shape) if partition else shape)
    seen = []
    for bank, addr in itertools.product(range(banks), range(size)):
        element = bank_element(shape, partition, bank, addr)
        sv = bank_element_sv(shape, partition, bank, str(addr)).replace("/", "//")
        assert eval(sv) == element, (bank, addr, sv)
        if partition:
            k = partition.dim - 1
            index = np.unravel_index(element, shape)[k]
            local = shape[k] // partition.factor
            assert (index % partition.factor if partition.kind == "cyclic" else index // local) == bank
            # Row-major inside the bank: the other dimensions keep their index.
            ei, ai = np.unravel_index(element, shape), np.unravel_index(addr, partition.bank_shape(shape))
            assert ei[:k] == ai[:k] and ei[k + 1:] == ai[k + 1:]
        seen.append(element)
    assert sorted(seen) == list(range(math.prod(shape)))


def test_banked_ports_classified():
    instance = banked_instance()
    assert sorted((m.param, m.bank, m.index) for m in instance.memories) == [
        ("A", 0, 0), ("A", 1, 0), ("B", None, 0), ("B", None, 1),
        ("C", 0, 0), ("C", 0, 1), ("C", 1, 0), ("C", 1, 1)]
    assert instance.memories[0].port("address") == "A_0_address0"


def test_missing_bank_rejected():
    ports = [p for p in banked_instance().ports if not p.name.startswith("A_1_")]
    with pytest.raises(ValueError, match=r"\['A bank 1'\]"):
        classify("banked", ports, banked_instance().params)


def test_spm_banks_from_attributes():
    layout = build_layout(parse_proc(BANKED))
    assert [(s.name, s.banks, s.cyclic) for s in layout.spms] == [
        ("bk_a", 2, True), ("bk_b", 1, True), ("bk_c", 2, False)]
    assert layout["bk_a"].ref("i") == "spm_bk_a[(i) % 2][(i) / 2]"
    assert layout["bk_c"].ref("i") == "spm_bk_c[(i) / 8][(i) % 8]"


@pytest.mark.parametrize("old, new, error", [
    ("#[partition_factor_array([2])]\n#[partition_cyclic_array([1])]\nstatic bk_a",
     "#[partition_factor_array([3])]\n#[partition_cyclic_array([1])]\nstatic bk_a", "must divide 256"),
    ("#[partition_dim_array([0])]\n#[partition_factor_array([2])]\n#[partition_cyclic_array([1])]",
     "#[partition_dim_array([1])]\n#[partition_factor_array([2])]\n#[partition_cyclic_array([1])]",
     "only a partition of the single dimension"),
])
def test_bad_spm_partition(old, new, error):
    assert BANKED.count(old) == 1
    with pytest.raises((ValueError, NotImplementedError), match=re.escape(error)):
        build_layout(parse_proc(BANKED.replace(old, new)))


def test_banked_example_accepted():
    sv = generate_main(parse_proc(BANKED), banked_instance())
    assert "reg [31:0] spm_bk_a [0:1][0:127];" in sv
    assert "hls_A_1_q0 <= spm_bk_a[(hls_A_base + " in sv


@pytest.mark.parametrize("old, new, ports", [
    # C at a runtime base: both C banks may meet in one SPM bank.
    ("C = bk_c[0 +: 16]", "C = bk_c[(a & 0) +: 16]", 4),
    # C's block banks each reach both cyclic SPM banks.
    ("#[partition_cyclic_array([0])]\nstatic bk_c", "#[partition_cyclic_array([1])]\nstatic bk_c", 4),
    # One flat bank for all four C ports.
    ("#[partition_dim_array([0])]\n#[partition_factor_array([2])]\n#[partition_cyclic_array([0])]\n", "", 4),
    # B's two ports reach both banks of bk_a, where one A port per bank already is.
    ("B = bk_b[b +: 16]", "B = bk_a[b +: 16]", 3),
])
def test_bank_port_budget(old, new, ports):
    assert BANKED.count(old) == 1
    with pytest.raises(ValueError, match=rf"up to {ports} kernel ports .* a bank has 2 ports"):
        generate_main(parse_proc(BANKED.replace(old, new)), banked_instance())


def _always_blocks(sv: str) -> list[str]:
    return re.findall(r"^always @.*?^end$", sv, re.M | re.S)


def test_each_spm_written_in_one_block():
    """DMA beats, kernel ports and FSM element writes all sit in one block per SPM."""
    from test_errors import _main
    sv = _main()
    for spm in ("matA", "matBT", "matCD"):
        writers = [b for b in _always_blocks(sv) if re.search(rf"spm_{spm}\[[^;]*\] <=", b)]
        assert len(writers) == 1, (spm, len(writers))
    block = next(b for b in _always_blocks(sv) if "spm_matA_fsm_we" in b and "<=" in b)
    assert "burst_write_enable" in block and "hls_A_we0" not in block  # A is only read
    assert "assert($countones(" in sv
