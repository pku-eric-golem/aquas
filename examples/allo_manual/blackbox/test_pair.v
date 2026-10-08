module test_pair(input wire ap_clk, ap_rst, ap_ce,
 input wire [31:0] a,b, output wire [31:0] sum, product,
 output wire ready);
 hf_add_f32 add(.ap_clk(ap_clk),.ap_rst(ap_rst),.ap_ce(ap_ce),.a(a),.b(b),.result(sum),.ap_ready(ready));
 hf_mul_f32 mul(.ap_clk(ap_clk),.ap_rst(ap_rst),.ap_ce(ap_ce),.a(a),.b(b),.result(product),.ap_ready());
endmodule
