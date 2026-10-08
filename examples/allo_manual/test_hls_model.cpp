#include <cstdint>
#include <cstring>
#include <cstdio>
#include "software/vectors.h"
extern "C" void gemm_fp32(float*,float*,float*,uint8_t,float*);

int main() {
    unsigned checked=0;
    for(const auto &test:cases) {
        float a[64],bt[64],c[64]={},d[64];
        memcpy(a,test.a,sizeof(a));
        for(unsigned i=0;i<8;++i)for(unsigned j=0;j<8;++j)
            memcpy(&bt[i*8+j],&test.b[j*8+i],4);
        for(unsigned pass=0;pass<3;++pass) {
            if(pass) memcpy(c,test.first,sizeof(c));
            float *dest=pass==2 ? c : d;
            gemm_fp32(a,bt,c,pass!=0,dest);
            for(unsigned i=0;i<64;++i) {
                uint32_t got;memcpy(&got,dest+i,4);
                uint32_t expected=pass ? test.accumulated[i] : test.first[i];
                if(got!=expected) {
                    fprintf(stderr,"pass=%u element=%u got=%08x expected=%08x\n",pass,i,got,expected);
                    return 1;
                }
                ++checked;
            }
        }
    }
    printf("HLS C MODEL PASS checked=%u (including in-place accumulation)\n",checked);
}
