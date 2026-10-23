/* RV32 Rocket program for examples/allo/glue.cadl: single _mem accesses
 * through the HellaCache port, if, a persistent register, single SPM
 * elements around the Allo vec_add kernel, and the error ABI (a failed
 * runtime check answers 0 and sets a sticky bit that 0x7b/0x7f returns and
 * clears). Single accesses share Rocket's data cache with the CPU, so they
 * run without fences; DMA transfers are fenced as in the other programs. */
#include <stdint.h>
#include <stdio.h>

#define ROCC_COMMAND(name, opcode, funct)                                     \
    static inline uint32_t name(uint32_t rs1, uint32_t rs2) {                   \
        uint32_t rd;                                                         \
        __asm__ volatile (".insn r " #opcode ", 7, " #funct ", %0, %1, %2"    \
                          : "=r"(rd) : "r"(rs1), "r"(rs2) : "memory");        \
        return rd;                                                           \
    }
ROCC_COMMAND(vadd, 0x0b, 0)
ROCC_COMMAND(ld, 0x0b, 1)
ROCC_COMMAND(st, 0x0b, 2)
ROCC_COMMAND(add2, 0x0b, 3)
ROCC_COMMAND(copy2x, 0x0b, 4)
ROCC_COMMAND(chase, 0x0b, 5)
ROCC_COMMAND(clamp, 0x0b, 6)
ROCC_COMMAND(elem_rd, 0x0b, 7)
ROCC_COMMAND(elem_wr, 0x0b, 8)
ROCC_COMMAND(load_b, 0x7b, 0)
ROCC_COMMAND(store_c, 0x7b, 1)
ROCC_COMMAND(error_status_rocc, 0x7b, 0x7f)

/* Error bits, as cadl_frontend/allo_sv_gen/fsm.py ERRORS. */
#define E_INVOKE_RANGE 1u
#define E_SPM_INDEX 2u
#define E_DMA_RANGE 4u
#define E_MEM_ALIGN 8u
#define SPM 256u
#define GUARD 16
#define DEAD 0xdeadbeefu

struct words {
    uint32_t before[GUARD];
    uint32_t w[64];
    uint32_t after[GUARD];
};
static volatile struct words mem __attribute__((aligned(64)));
static volatile struct words out __attribute__((aligned(64)));
static volatile int32_t b_src[2][16] __attribute__((aligned(64)));

static unsigned checks;

static void fence_memory(void) { __asm__ volatile ("fence rw, rw" ::: "memory"); }
static uint32_t addr(volatile const void *p) { return (uint32_t)(uintptr_t)p; }
static uint32_t error_status(void) { return error_status_rocc(0, 0); }

static int expect(const char *what, uint32_t got, uint32_t want) {
#ifdef GLUE_FORCE_FAIL
    if (checks == 0) want ^= 1;
#endif
    ++checks;
    if (got == want) return 0;
    printf("ALLO GLUE FAIL: %s got=%08x want=%08x\n", what, (unsigned)got, (unsigned)want);
    return 1;
}
#define EXPECT(what, got, want) do { if (expect(what, got, want)) return 1; } while (0)

static int guards(const char *what, volatile const struct words *buffer) {
    for (unsigned i = 0; i < GUARD; ++i)
        if (buffer->before[i] != DEAD || buffer->after[i] != DEAD) {
            printf("ALLO GLUE FAIL: %s guard word %u\n", what, i);
            return 1;
        }
    return 0;
}

static int memory_glue(void) {
    volatile uint32_t *w = mem.w;
    for (unsigned i = 0; i < 64; ++i) w[i] = i * 37u + 11u;

    EXPECT("st rd", st(addr(&w[0]), 0x12345678u), 1);
    EXPECT("st seen by CPU", w[0], 0x12345678u);
    EXPECT("ld", ld(addr(&w[0]), 0), 0x12345678u);
    w[1] = 0xcafe0001u;
    EXPECT("CPU store seen by ld", ld(addr(&w[1]), 0), 0xcafe0001u);
    EXPECT("add2", add2(addr(&w[2]), 0), w[2] + w[3]);

    EXPECT("copy2x count", copy2x(addr(&w[8]), addr(&w[16])), 4);
    for (unsigned k = 0; k < 4; ++k) EXPECT("copy2x data", w[16 + k], 2u * w[8 + k]);
    EXPECT("copy2x count again", copy2x(addr(&w[16]), addr(&w[24])), 8);
    for (unsigned k = 0; k < 4; ++k) EXPECT("copy2x chained", w[24 + k], 4u * w[8 + k]);

    w[30] = addr(&w[5]);
    EXPECT("chase", chase(addr(&w[30]), 0), w[5]);

    w[40] = 250;
    EXPECT("clamp taken", clamp(addr(&w[40]), 0), 1);
    EXPECT("clamp wrote", w[40], 100);
    EXPECT("clamp not taken", clamp(addr(&w[40]), 0), 0);
    EXPECT("clamp kept", w[40], 100);
    return guards("single access", &mem);
}

static int32_t a_val(unsigned i) { return (int32_t)(i * 3u) - 20; }
static int32_t b_val(unsigned batch, unsigned i) { return (int32_t)(batch * 1000u + i * 7u) - 50; }

static int spm_glue(void) {
    for (unsigned i = 0; i < 16; ++i) EXPECT("elem_wr", elem_wr(32 + i, (uint32_t)a_val(i)), 1);
    for (unsigned i = 0; i < 16; ++i) b_src[0][i] = b_val(0, i);
    fence_memory();
    EXPECT("load_b", load_b(addr(b_src[0]), 48), 1);
    EXPECT("vadd", vadd((32u << 16) | 48u, 64), 1);
    for (unsigned i = 0; i < 16; ++i)
        EXPECT("vadd elem_rd", elem_rd(64 + i, 0), (uint32_t)(a_val(i) + b_val(0, i)));

    for (unsigned i = 0; i < GUARD; ++i) out.before[i] = out.after[i] = DEAD;
    for (unsigned i = 0; i < 64; ++i) out.w[i] = DEAD;
    fence_memory();
    EXPECT("store_c", store_c(addr(out.w), 64), 1);
    fence_memory();
    for (unsigned i = 0; i < 16; ++i)
        EXPECT("store_c data", out.w[i], (uint32_t)(a_val(i) + b_val(0, i)));
    for (unsigned i = 16; i < 64; ++i) EXPECT("store_c tail", out.w[i], DEAD);
    return guards("store_c", &out);
}

static int errors(void) {
    volatile uint32_t *w = mem.w;
    EXPECT("status clean", error_status(), 0);

    /* Single SPM elements. */
    EXPECT("elem_rd past end", elem_rd(SPM, 0), 0);
    EXPECT("status spm_index", error_status(), E_SPM_INDEX);
    EXPECT("status cleared", error_status(), 0);
    EXPECT("elem_wr past end", elem_wr(SPM + 44, 7), 0);
    EXPECT("status spm_index wr", error_status(), E_SPM_INDEX);

    /* Single memory accesses: aborted before reaching the cache. */
    w[50] = 0x50505050u;
    w[51] = 0x51515151u;
    EXPECT("ld misaligned", ld(addr(&w[50]) + 1, 0), 0);
    EXPECT("status mem_align", error_status(), E_MEM_ALIGN);
    EXPECT("st misaligned", st(addr(&w[50]) + 2, 0xbadu), 0);
    EXPECT("status mem_align st", error_status(), E_MEM_ALIGN);
    EXPECT("st misaligned kept w50", w[50], 0x50505050u);
    EXPECT("st misaligned kept w51", w[51], 0x51515151u);

    /* invoke: C would run past vec_c; vec_c must keep its values. */
    uint32_t before = elem_rd(SPM - 6, 0);
    EXPECT("vadd C past end", vadd((32u << 16) | 48u, SPM - 6), 0);
    EXPECT("status invoke_range", error_status(), E_INVOKE_RANGE);
    EXPECT("vadd aborted kept vec_c", elem_rd(SPM - 6, 0), before);
    EXPECT("vadd A past end", vadd(((SPM - 15) << 16) | 48u, 64), 0);
    EXPECT("status invoke_range A", error_status(), E_INVOKE_RANGE);

    /* DMA: an aborted load must not write vec_b[240..255]. */
    for (unsigned i = 0; i < 16; ++i) b_src[1][i] = b_val(1, i);
    fence_memory();
    EXPECT("load_b tail", load_b(addr(b_src[0]), SPM - 16), 1);
    EXPECT("load_b past end", load_b(addr(b_src[1]), SPM - 6), 0);
    EXPECT("status dma_range", error_status(), E_DMA_RANGE);
    EXPECT("vadd after aborted load", vadd((32u << 16) | (SPM - 16), 80), 1);
    for (unsigned i = 0; i < 16; ++i)
        EXPECT("aborted load kept vec_b", elem_rd(80 + i, 0), (uint32_t)(a_val(i) + b_val(0, i)));

    /* DMA: an aborted store must not write memory. */
    for (unsigned i = 0; i < 64; ++i) out.w[i] = DEAD;
    fence_memory();
    EXPECT("store_c past end", store_c(addr(out.w), SPM - 8), 0);
    EXPECT("status dma_range store", error_status(), E_DMA_RANGE);
    fence_memory();
    for (unsigned i = 0; i < 64; ++i) EXPECT("aborted store kept memory", out.w[i], DEAD);
    if (guards("aborted store", &out)) return 1;

    /* Sticky: causes accumulate until read. */
    EXPECT("elem_rd far", elem_rd(1u << 20, 0), 0);
    EXPECT("ld odd", ld(addr(&w[0]) + 3, 0), 0);
    EXPECT("status sticky", error_status(), E_SPM_INDEX | E_MEM_ALIGN);
    EXPECT("status sticky cleared", error_status(), 0);

    /* State survives the aborted instructions. */
    EXPECT("total survives", copy2x(addr(&w[8]), addr(&w[32])), 12);
    EXPECT("vec_c survives", elem_rd(64, 0), (uint32_t)(a_val(0) + b_val(0, 0)));
    EXPECT("status end", error_status(), 0);
    return guards("errors", &mem);
}

int main(void) {
    printf("ALLO GLUE Rocket/RoCC: _mem single access, SPM elements, error ABI\n");
    for (unsigned i = 0; i < GUARD; ++i) mem.before[i] = mem.after[i] = DEAD;
    if (expect("status after reset", error_status(), 0)) return 1;
    if (memory_glue() || spm_glue() || errors()) return 1;
    printf("ALLO GLUE PASS checks=%u\n", checks);
    return 0;
}
