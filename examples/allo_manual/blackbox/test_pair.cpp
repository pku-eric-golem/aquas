#include "Vtest_pair.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <stdexcept>
#include <array>

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vtest_pair dut;
    std::ifstream vectors(argv[1]);
    if (!vectors) throw std::runtime_error("Missing vectors");
    std::deque<std::array<uint32_t,2>> pending;
    unsigned checked=0, stalls=0, resets=0, index=0;
    auto edge=[&](){dut.ap_clk=0;dut.eval();dut.ap_clk=1;dut.eval();};
    uint32_t a,b,sum,product;
    while(vectors >> std::hex >> a >> b >> sum >> product) {
        if (index % 113 == 0) {
            dut.ap_rst=1; dut.ap_ce=1; edge();
            if(dut.sum || dut.product) throw std::runtime_error("Reset masking");
            pending.clear(); dut.ap_rst=0; ++resets;
        }
        if(index % 3 == 0) {
            dut.ap_ce=0; dut.a=~a; dut.b=~b; dut.eval();
            const uint32_t previous_sum=dut.sum, previous_product=dut.product;
            for(unsigned i=0;i<(index%5)+1;++i) {
                edge(); ++stalls;
                if(dut.sum!=previous_sum || dut.product!=previous_product)
                    throw std::runtime_error("Clock enable failed to hold stages");
            }
        }
        dut.ap_ce=1; dut.a=a; dut.b=b;
        pending.push_back({sum,product}); edge();
        if(!dut.ready) throw std::runtime_error("Auxiliary constant output");
        if(pending.size()==3) {
            const auto expected=pending.front(); pending.pop_front();
            if(dut.sum!=expected[0] || dut.product!=expected[1]) {
                fprintf(stderr,"index=%u sum=%08x/%08x product=%08x/%08x\n",
                        index,dut.sum,expected[0],dut.product,expected[1]);
                return 1;
            }
            checked+=2;
        }
        ++index;
    }
    printf("NATIVE HARDFLOAT PASS checked=%u latency=3 II=1 stalls=%u resets=%u\n",
           checked,stalls,resets);
    return checked > 1000 ? 0 : 1;
}
