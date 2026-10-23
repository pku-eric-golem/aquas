"""Subtask E: DMA-side burst access to the scratchpads.

The DMA engine writes one beat per cycle on burst_write and reads by issuing
burst_read_0 with an address; the beat returns on burst_read_1_res0 one
cycle later. A beat carries `beat_bits / elem_bits` consecutive elements.
Beat writes are write sources of the SPM write blocks (spm.py).
"""
from __future__ import annotations

from .ports import WrapperPort, width_of
from .spm import SpmLayout, SpmWrite


def burst_writes(layout: SpmLayout, ports: list[WrapperPort]) -> list[SpmWrite]:
    """One write source per SPM and beat lane: element `addr + lane` of the selected SPM."""
    elem = layout.spms[0].elem_bits
    lanes = width_of(ports, "burst_write_data") // elem
    wsel, wel = layout.select("burst_write_addr"), layout.element("burst_write_addr")
    return [SpmWrite(spm.name, "dma", f"!rst && burst_write_enable && {wsel} == {spm.index}",
                     f"{wel} + {lane}", f"burst_write_data[{lane * elem + elem - 1}:{lane * elem}]")
            for spm in layout.spms for lane in range(lanes)]


def generate_burst(layout: SpmLayout, ports: list[WrapperPort]) -> str:
    beat = width_of(ports, "burst_write_data")
    assert width_of(ports, "burst_read_1_res0") == beat
    elem = layout.spms[0].elem_bits
    lanes = beat // elem
    wsel = layout.select("burst_write_addr")
    rsel, rel = layout.select("burst_read_0_addr"), layout.element("burst_read_0_addr")

    read_cases = "\n".join(
        f"        {spm.index}: spm_burst_read_data[spm_lane*{elem} +: {elem}] <= "
        f"{spm.ref(f'{rel} + spm_lane')};" for spm in layout.spms)
    used_bits = layout.offset_bits + layout.select_bits
    return f"""reg [{beat - 1}:0] spm_burst_read_data;
assign burst_write_ready = !rst;
assign burst_read_0_ready = !rst;
assign burst_read_1_ready = !rst;
assign burst_read_1_res0 = spm_burst_read_data;

integer spm_lane;
always @(posedge clk) begin
  if (!rst && burst_read_0_enable) begin
    for (spm_lane = 0; spm_lane < {lanes}; spm_lane = spm_lane + 1) begin
      case ({rsel})
{read_cases}
        default: spm_burst_read_data[spm_lane*{elem} +: {elem}] <= 0;
      endcase
    end
  end
end

`ifndef SYNTHESIS
always @(posedge clk) if (!rst && burst_write_enable) begin
  assert(burst_write_addr[31:{used_bits}] == 0 && {wsel} < {len(layout.spms)});
  assert(burst_write_addr[{(beat // 8).bit_length() - 2}:0] == 0);
end
`endif
"""
