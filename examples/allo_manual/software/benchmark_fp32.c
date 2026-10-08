/* Derived from the existing BF16 GEMM benchmark's RoCC ABI and tiled layout. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "bench_vectors.h"

#define ROCC_COMMAND(name, opcode, funct) \
static inline uint32_t name(uint32_t a, uint32_t b) { \
    uint32_t rd; \
    __asm__ volatile(".insn r " #opcode ", 7, " #funct ", %0, %1, %2\naddi %0,%0,0" \
                     : "=r"(rd) : "r"(a), "r"(b) : "memory"); \
    return rd; \
}
ROCC_COMMAND(compute, 0x0b, 0)
ROCC_COMMAND(load_a, 0x7b, 0)
ROCC_COMMAND(load_b, 0x7b, 1)
ROCC_COMMAND(store_c, 0x7b, 2)

static uint32_t a[512] __attribute__((aligned(128)));
static uint32_t bt[512] __attribute__((aligned(128)));
static uint32_t d[512] __attribute__((aligned(128)));
static float fa[512], fbt[512], fd[512];
static inline float as_float(uint32_t bits) {union {uint32_t u;float f;} v={.u=bits};return v.f;}
static inline uint32_t as_bits(float value) {union {uint32_t u;float f;} v={.f=value};return v.u;}
static inline uint32_t cycles(void) {
    uint32_t n; __asm__ volatile("fence rw,rw\ncsrr %0,mcycle":"=r"(n)::"memory");return n;
}
__attribute__((noinline)) static void software_gemm(void) {
    for(unsigned tile=0;tile<8;++tile)
        for(unsigned i=0;i<8;++i)
            for(unsigned j=0;j<8;++j) {
                float acc=0.0f;
                for(unsigned k=0;k<8;++k) {
                    float product=fa[i*64+tile*8+k]*fbt[j*64+tile*8+k];
                    acc=acc+product;
                }
                fd[i*64+tile*8+j]=acc+0.0f;
            }
}
static int run_software(unsigned trial) {
    for(unsigned n=0;n<512;++n) {fa[n]=as_float(bench_a[n]);fbt[n]=as_float(bench_bt[n]);}
    uint32_t start=cycles();software_gemm();uint32_t end=cycles();
    for(unsigned n=0;n<512;++n)
        if(as_bits(fd[n])!=bench_expected[n]) {
            printf("ALLO FP32 FAIL software trial=%u index=%u expected=%08lx got=%08lx\n",
                   trial,n,(unsigned long)bench_expected[n],(unsigned long)as_bits(fd[n]));return 1;
        }
    printf("ALLO_BENCH trial=%u engine=software tiles=8 elements=512 cycles=%lu\n",trial,(unsigned long)(end-start));
    return 0;
}
static int run_isax(unsigned trial) {
    memcpy(a,bench_a,sizeof(a));memcpy(bt,bench_bt,sizeof(bt));memset(d,0xa5,sizeof(d));
    unsigned ack=0,at=8,btile=24,dt=40;
    uint32_t start=cycles();
    ack|=load_a((uint32_t)(uintptr_t)a,(at*64<<16)|256);
    ack|=load_b((uint32_t)(uintptr_t)bt,(btile*64<<16)|256);
    uint32_t loaded=cycles();
    for(unsigned t=0;t<8;++t)ack|=compute(((at+t)<<16)|(btile+t),((dt+t)<<16)|(dt+t));
    uint32_t computed=cycles();
    ack|=store_c((uint32_t)(uintptr_t)d,(dt*64<<16)|256);
    uint32_t end=cycles();
    if(ack) {printf("ALLO FP32 FAIL acknowledgement=%u\n",ack);return 1;}
    for(unsigned n=0;n<512;++n)
        if(d[n]!=bench_expected[n]) {
            printf("ALLO FP32 FAIL isax trial=%u index=%u expected=%08lx got=%08lx\n",
                   trial,n,(unsigned long)bench_expected[n],(unsigned long)d[n]);return 1;
        }
    printf("ALLO_BENCH trial=%u engine=isax tiles=8 elements=512 load=%lu compute=%lu store=%lu total=%lu\n",
           trial,(unsigned long)(loaded-start),(unsigned long)(computed-loaded),
           (unsigned long)(end-computed),(unsigned long)(end-start));
    return 0;
}
int main(void) {
    __asm__ volatile("csrw fcsr,zero");
    puts("ALLO FP32 GEMM: Rocket software vs Allo/Vitis HLS RTL, separate mul/add RNE");
    if(run_software(0)||run_isax(0)||run_isax(1)||run_software(1))return 1;
    puts("ALLO FP32 BENCH PASS trials=2 checked=2048");return 0;
}
