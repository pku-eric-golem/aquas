// Drives only clk/rst and the DMA burst ports of `main`; other inputs stay 0.
module tb;
  reg clk = 0;
  reg rst = 1;
  always #1 clk = ~clk;

  reg burst_write_enable = 0;
  reg [31:0] burst_write_addr = 0;
  reg [63:0] burst_write_data = 0;
  reg burst_read_0_enable = 0;
  reg [31:0] burst_read_0_addr = 0;
  wire [63:0] burst_read_1_res0;
  wire burst_write_ready, burst_read_0_ready, burst_read_1_ready;

  main dut(
    .clk(clk), .rst(rst),
    .burst_write_enable(burst_write_enable), .burst_write_ready(burst_write_ready),
    .burst_write_addr(burst_write_addr), .burst_write_data(burst_write_data),
    .burst_read_0_enable(burst_read_0_enable), .burst_read_0_ready(burst_read_0_ready),
    .burst_read_0_addr(burst_read_0_addr),
    .burst_read_1_enable(1'b1), .burst_read_1_ready(burst_read_1_ready),
    .burst_read_1_res0(burst_read_1_res0)
  );

  // Region base, first beat, a middle beat and the last beat of each SPM.
  localparam integer N = 12;
  function automatic [31:0] addr_of(input integer i);
    integer region, which;
    begin
      region = i / 4;
      which = i % 4;
      addr_of = (region << 19) + (which == 0 ? 0 : which == 1 ? 8 :
                                   which == 2 ? 32'h3_0a08 : 32'h7_fff8);
    end
  endfunction
  function automatic [63:0] data_of(input integer i);
    data_of = {32'hc0de_0000 + i, 32'h5eed_0000 + (i * 7)};
  endfunction

  integer i, errors = 0;
  initial begin
    repeat (3) @(negedge clk);
    rst = 0;
    @(negedge clk);
    if (!burst_write_ready || !burst_read_0_ready || !burst_read_1_ready) begin
      $display("BURST FAIL: not ready");
      $finish;
    end
    for (i = 0; i < N; i = i + 1) begin
      burst_write_enable = 1;
      burst_write_addr = addr_of(i);
      burst_write_data = data_of(i);
      @(negedge clk);
    end
    burst_write_enable = 0;
    for (i = 0; i < N; i = i + 1) begin
      burst_read_0_enable = 1;
      burst_read_0_addr = addr_of(i);
      @(negedge clk);
      burst_read_0_enable = 0;
      if (burst_read_1_res0 !== data_of(i)) begin
        $display("BURST FAIL: addr=%h got=%h want=%h", addr_of(i), burst_read_1_res0, data_of(i));
        errors = errors + 1;
      end
    end
    if (errors == 0) $display("BURST PASS beats=%0d", N);
    $finish;
  end
endmodule
