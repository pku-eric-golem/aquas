// Same native stage boundaries, with HLS CE and reset adaptation.
module hf_mul_f32(
 input wire ap_clk, ap_rst, ap_ce,
 input wire [31:0] a, b,
 output wire [31:0] result,
 output wire ap_ready
);
 wire [36:0] payload;
 HFNativePipelineCE #(.WIDTH(32), .OP(2), .LATENCY(3)) arithmetic(
  .clock(ap_clk), .clock_enable(ap_ce && !ap_rst),
  .a(a), .b(b), .c(32'b0), .rm(3'b000), .payload(payload));
 // HLS tracks result validity. Three enabled edges refill native stages.
 assign result = ap_rst ? 32'b0 : payload[31:0];
 // Vitis 2024.1 connects this unused pin for a scalar-return call.
 // Register it as an auxiliary scalar output, not a chain control signal.
 assign ap_ready = 1'b1;
endmodule
