// Exercise the actual HLS RTL through the handwritten adapter, without a CPU.
#include "Vmain.h"
#include "verilated.h"
#include "software/vectors.h"
#include <cstdio>
#include <cstdlib>
#include <random>

static unsigned long long ticks;
static void require(bool ok, const char *message) {
  if (!ok) { std::fprintf(stderr, "FAIL cycle=%llu: %s\n", ticks, message); std::exit(1); }
}
static void edge(Vmain &dut) {
  dut.clk=0; dut.eval(); dut.clk=1; dut.eval(); dut.clk=0; dut.eval(); ++ticks;
}
static void write_tile(Vmain &dut, unsigned bank, unsigned tile, const uint32_t *data,
                       bool transpose=false) {
  for (unsigned n=0; n<64; n+=2) {
    uint64_t word=0;
    for (unsigned lane=0; lane<2; ++lane) {
      unsigned at=n+lane;
      word |= uint64_t(data[transpose ? (at%8)*8+at/8 : at]) << (lane*32);
    }
    dut.burst_write_enable=1;
    dut.burst_write_addr=bank*0x80000+tile*256+n*4;
    dut.burst_write_data=word;
    edge(dut);
  }
  dut.burst_write_enable=0;
}
static void compute(Vmain &dut, unsigned a, unsigned b, unsigned c, unsigned d,
                    bool accumulate, unsigned rd) {
  dut.rocc_cmd_rocc_cmd_opcode=0x0b;
  dut.rocc_cmd_rocc_cmd_funct=0;
  dut.rocc_cmd_rocc_cmd_rd=rd;
  dut.rocc_cmd_rocc_cmd_rs1data=(a<<16)|b|(accumulate?0x8000:0);
  dut.rocc_cmd_rocc_cmd_rs2data=(c<<16)|d;
  dut.rocc_cmd_enable=1;
  dut.eval(); require(dut.rocc_cmd_ready, "command not admitted"); edge(dut);
  dut.rocc_cmd_enable=0;
  unsigned wait=0;
  while (!dut.rocc_resp_rocc_resp_to_bus_enable) {
    require(wait++<200000, "HLS completion timeout"); edge(dut);
  }
  for (unsigned stall=0; stall<17; ++stall) {
    require(!dut.rocc_cmd_ready, "accepted another command while response pending");
    require(dut.rocc_resp_rocc_resp_to_bus_enable, "dropped response under backpressure");
    require(dut.rocc_resp_rocc_resp_to_bus_result_rd==rd, "incorrect rd");
    require(dut.rocc_resp_rocc_resp_to_bus_result_rddata==0, "nonzero acknowledgement");
    edge(dut);
  }
  dut.rocc_resp_rocc_resp_to_bus_ready=1; edge(dut);
  dut.rocc_resp_rocc_resp_to_bus_ready=0; dut.eval();
  require(!dut.rocc_resp_rocc_resp_to_bus_enable, "duplicate response");
}
static void check_tile(Vmain &dut, unsigned tile, const uint32_t *expected) {
  for (unsigned n=0; n<64; n+=2) {
    dut.burst_read_0_enable=1;
    dut.burst_read_0_addr=0x100000+tile*256+n*4;
    edge(dut);
    uint64_t word=dut.burst_read_1_res0;
    for (unsigned lane=0; lane<2; ++lane) {
      unsigned got=(word>>(lane*32))&0xffffffffULL;
      if (got!=expected[n+lane]) {
        std::fprintf(stderr,"tile=%u index=%u expected=%08x got=%08x\n",
                     tile,n+lane,expected[n+lane],got);
        require(false, "FP32 GEMM mismatch");
      }
    }
  }
  dut.burst_read_0_enable=0;
}
int main(int argc, char **argv) {
  Verilated::commandArgs(argc,argv);
  Vmain dut;
  dut.rst=1; dut.rocc_cmd_enable=0;
  dut.rocc_cmd_rocc_cmd_xd=1;
  dut.rocc_cmd_rocc_cmd_xs1=dut.rocc_cmd_rocc_cmd_xs2=1;
  dut.rocc_resp_rocc_resp_to_bus_ready=0;
  dut.burst_write_enable=dut.burst_read_0_enable=dut.burst_read_1_enable=0;
  dut.dma_poll_for_idle_ch0_ready=dut.dma_poll_for_idle_ch0_res0=1;
  edge(dut); edge(dut); dut.rst=0; edge(dut);
  unsigned checked=0;
  for (unsigned t=0; t<sizeof(cases)/sizeof(cases[0]); ++t) {
    unsigned a=1008+t,b=1016+t,c=1000+t,d=1100+t;
    write_tile(dut,0,a,cases[t].a);
    write_tile(dut,1,b,cases[t].b,true);
    compute(dut,a,b,c,c,false,t%31+1);
    check_tile(dut,c,cases[t].first);
    compute(dut,a,b,c,d,true,(t+7)%31+1);
    check_tile(dut,d,cases[t].accumulated);
    // Also exercise C == D, which must read each element before overwriting it.
    compute(dut,a,b,c,c,true,(t+13)%31+1);
    check_tile(dut,c,cases[t].accumulated);
    checked+=192;
  }
  // An in-flight transaction must not produce a stale response after reset.
  dut.rocc_cmd_enable=1; edge(dut); dut.rocc_cmd_enable=0;
  for(unsigned n=0;n<9;++n)edge(dut);
  dut.rst=1; edge(dut); dut.rst=0;
  for(unsigned n=0;n<100;++n) {
    edge(dut); require(!dut.rocc_resp_rocc_resp_to_bus_enable,"stale response after reset");
  }
  std::printf("ALLO WRAPPER PASS checked=%u cycles=%llu backpressure=17 reset=PASS\n",checked,ticks);
  dut.final();
}
