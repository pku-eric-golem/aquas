// One command in flight. Native HLS memory ports share the DMA scratchpads.
localparam integer SPM_ELEMENTS = 131072;
localparam [3:0] IDLE=0, START=1, RUN=2, DMA_ISSUE=3,
                 DMA_BUSY=4, DMA_DONE=5, RESPONSE=6;
reg [3:0] state;
reg [4:0] saved_rd;
reg [31:0] response_data;
reg [15:0] tile_a, tile_b, tile_c, tile_d;
reg [7:0] use_c;
reg [31:0] cpu_addr, spm_addr;
reg [15:0] row_pitch;
reg [2:0] dma_row;
reg dma_half;
reg dma_store;
reg [31:0] mem_a[0:SPM_ELEMENTS-1];
reg [31:0] mem_b[0:SPM_ELEMENTS-1];
reg [31:0] mem_c[0:SPM_ELEMENTS-1];
reg [63:0] dma_read_data;

assign rocc_cmd_ready = !rst && state == IDLE;
assign rocc_resp_rocc_resp_to_bus_enable = !rst && state == RESPONSE;
assign rocc_resp_rocc_resp_to_bus_result_rd = saved_rd;
assign rocc_resp_rocc_resp_to_bus_result_rddata = response_data;

assign burst_write_ready = !rst;
assign burst_read_0_ready = !rst;
assign burst_read_1_ready = !rst;
assign burst_read_1_res0 = dma_read_data;

assign dma_cpu_to_isax_ch0_enable = !rst && state == DMA_ISSUE && !dma_store;
assign dma_isax_to_cpu_ch0_enable = !rst && state == DMA_ISSUE && dma_store;
assign dma_cpu_to_isax_ch0_cpu_addr = cpu_addr;
assign dma_isax_to_cpu_ch0_cpu_addr = cpu_addr;
assign dma_cpu_to_isax_ch0_isax_addr = spm_addr;
assign dma_isax_to_cpu_ch0_isax_addr = spm_addr;
assign dma_cpu_to_isax_ch0_length = 7; // 128 bytes: four 32-byte FP32 tile rows.
assign dma_isax_to_cpu_ch0_length = 7;
assign dma_cpu_to_isax_ch0_stride_x = 32;
assign dma_isax_to_cpu_ch0_stride_x = 32;
assign dma_cpu_to_isax_ch0_stride_y = 8;
assign dma_isax_to_cpu_ch0_stride_y = 8;

wire dma_idle = dma_poll_for_idle_ch0_ready && dma_poll_for_idle_ch0_res0;
wire command_fire = rocc_cmd_enable && rocc_cmd_ready;

always @(posedge clk) begin
  if (rst) begin
    state <= IDLE;
    saved_rd <= 0;
    response_data <= 0;
    tile_a <= 0; tile_b <= 0; tile_c <= 0; tile_d <= 0;
    use_c <= 0;
    cpu_addr <= 0; spm_addr <= 0; row_pitch <= 0;
    dma_row <= 0; dma_half <= 0; dma_store <= 0;
  end else begin
    case (state)
      IDLE: if (command_fire) begin
        saved_rd <= rocc_cmd_rocc_cmd_rd;
        response_data <= 0;
        if (rocc_cmd_rocc_cmd_opcode == 7'h0b && rocc_cmd_rocc_cmd_funct == 0) begin
          tile_a <= rocc_cmd_rocc_cmd_rs1data[31:16] & 16'hefff;
          tile_b <= {1'b0, rocc_cmd_rocc_cmd_rs1data[14:0]};
          tile_c <= rocc_cmd_rocc_cmd_rs2data[31:16];
          tile_d <= rocc_cmd_rocc_cmd_rs2data[15:0];
          use_c <= {7'b0, rocc_cmd_rocc_cmd_rs1data[15]};
          state <= START;
        end else if (rocc_cmd_rocc_cmd_opcode == 7'h7b && rocc_cmd_rocc_cmd_funct <= 2) begin
          cpu_addr <= rocc_cmd_rocc_cmd_rs1data;
          spm_addr <= ({16'b0, rocc_cmd_rocc_cmd_rs2data[31:16]} << 2)
                    + (rocc_cmd_rocc_cmd_funct == 0 ? 32'h0 :
                       rocc_cmd_rocc_cmd_funct == 1 ? 32'h80000 : 32'h100000);
          row_pitch <= rocc_cmd_rocc_cmd_rs2data[15:0];
          dma_row <= 0;
          dma_half <= 0;
          dma_store <= rocc_cmd_rocc_cmd_funct == 2;
          state <= DMA_ISSUE;
        end else begin
          response_data <= 1;
          state <= RESPONSE;
        end
      end
      START: state <= RUN;
      RUN: if (hls_done) state <= RESPONSE;
      DMA_ISSUE: if (dma_store ? dma_isax_to_cpu_ch0_ready : dma_cpu_to_isax_ch0_ready)
        state <= DMA_BUSY;
      // Wait for idle to fall first: the engine samples enable on this edge.
      DMA_BUSY: if (!dma_idle) state <= DMA_DONE;
      DMA_DONE: if (dma_idle) begin
        if (!dma_half) begin
          dma_half <= 1;
          cpu_addr <= cpu_addr + 128;
          spm_addr <= spm_addr + 1024; // next four tiles
          state <= DMA_ISSUE;
        end else if (dma_row == 7) state <= RESPONSE;
        else begin
          dma_half <= 0;
          dma_row <= dma_row + 1'b1;
          cpu_addr <= cpu_addr + {16'b0, row_pitch} - 128;
          spm_addr <= spm_addr + 32 - 1024;
          state <= DMA_ISSUE;
        end
      end
      RESPONSE: if (rocc_resp_rocc_resp_to_bus_ready) state <= IDLE;
      default: state <= IDLE;
    endcase
  end
end

integer lane;
always @(posedge clk) begin
  if (!rst && burst_write_enable) begin
    for (lane=0; lane<2; lane=lane+1) begin
      case (burst_write_addr[20:19])
        0: mem_a[burst_write_addr[18:2] + lane] <= burst_write_data[lane*32 +: 32];
        1: mem_b[burst_write_addr[18:2] + lane] <= burst_write_data[lane*32 +: 32];
        2: mem_c[burst_write_addr[18:2] + lane] <= burst_write_data[lane*32 +: 32];
        default: ;
      endcase
    end
  end
  if (!rst && burst_read_0_enable) begin
    for (lane=0; lane<2; lane=lane+1) begin
      case (burst_read_0_addr[20:19])
        0: dma_read_data[lane*32 +: 32] <= mem_a[burst_read_0_addr[18:2] + lane];
        1: dma_read_data[lane*32 +: 32] <= mem_b[burst_read_0_addr[18:2] + lane];
        2: dma_read_data[lane*32 +: 32] <= mem_c[burst_read_0_addr[18:2] + lane];
        default: dma_read_data[lane*32 +: 32] <= 0;
      endcase
    end
  end
end

`ifndef SYNTHESIS
always @(posedge clk) if (!rst) begin
  if (command_fire) begin
    assert(rocc_cmd_rocc_cmd_xd) else $fatal(1, "This adapter requires xd=1");
    if (rocc_cmd_rocc_cmd_opcode == 7'h0b && rocc_cmd_rocc_cmd_funct == 0) begin
      assert((rocc_cmd_rocc_cmd_rs1data[31:16] & 16'hefff) < 2048);
      assert(rocc_cmd_rocc_cmd_rs1data[14:0] < 2048);
      assert(rocc_cmd_rocc_cmd_rs2data[31:16] < 2048);
      assert(rocc_cmd_rocc_cmd_rs2data[15:0] < 2048);
    end
  end
  if (burst_write_enable) begin
    assert(burst_write_addr[31:21] == 0 && burst_write_addr[20:19] < 3);
    assert(burst_write_addr[2:0] == 0);
  end
end
`endif
