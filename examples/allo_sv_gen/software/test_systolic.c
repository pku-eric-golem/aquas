/* RV32 Rocket program for examples/allo/systolic.cadl: C = A * B with
 * A[M][4], B[4][N] in DRAM, 4x4 tiles moved by DMA and computed by the Allo
 * library systolic array. The reference is computed on the Rocket FPU in the
 * PE order (v = v + a * b, k ascending, separately rounded), so results must
 * match bit for bit. */
#include <stdint.h>
#include <stdio.h>

#define ROCC_COMMAND(name, opcode, funct)                                     \
    static inline uint32_t name(uint32_t rs1, uint32_t rs2) {                   \
        uint32_t rd;                                                         \
        __asm__ volatile (".insn r " #opcode ", 7, " #funct ", %0, %1, %2"    \
                          : "=r"(rd) : "r"(rs1), "r"(rs2) : "memory");        \
        return rd;                                                           \
    }
ROCC_COMMAND(systolic4x4, 0x0b, 0)
ROCC_COMMAND(loadtile_a, 0x7b, 0)
ROCC_COMMAND(loadtile_b, 0x7b, 1)
ROCC_COMMAND(storetile_c, 0x7b, 2)

#define M 8
#define N 8
#define K 4
#define GUARD 64
/* Row pitches in words; batch 1 pads every row. All pitches are multiples of
 * four words so each 16-byte DMA row stays 16-byte aligned. */
static const unsigned pitches[2][3] = {{K, N, N}, {K + 4, N + 8, N + 12}};
/* SPM tiles per batch: nonzero offsets and the top tile (63). */
static const unsigned spm_a[2][2] = {{5, 63}, {12, 0}};
static const unsigned spm_b[2][2] = {{1, 62}, {33, 7}};
static const unsigned spm_c[2][4] = {{0, 7, 40, 63}, {63, 2, 3, 4}};

struct buffer {
    uint32_t before[GUARD];
    float data[M * (N + 12)];
    uint32_t after[GUARD];
};
static volatile struct buffer a_buf __attribute__((aligned(64)));
static volatile struct buffer b_buf __attribute__((aligned(64)));
static volatile struct buffer c_buf __attribute__((aligned(64)));
static float reference[M][N];

static uint32_t state = 0x2545f491u;
static float random_float(void) {
    /* Normal values with random sign and mantissa in roughly [2^-4, 2^4). */
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    uint32_t bits = (state & 0x807fffffu) | ((123u + (state >> 23) % 8u) << 23);
    union { uint32_t u; float f; } value = {bits};
    return value.f;
}

static void fence_memory(void) { __asm__ volatile ("fence rw, rw" ::: "memory"); }
static uint32_t cycles(void) { uint32_t c; __asm__ volatile ("rdcycle %0" : "=r"(c)); return c; }
static uint32_t bits(float f) { union { float f; uint32_t u; } value = {f}; return value.u; }

static void fill(volatile struct buffer *buffer) {
    for (unsigned i = 0; i < GUARD; ++i) buffer->before[i] = buffer->after[i] = 0xdeadbeef;
    for (unsigned i = 0; i < M * (N + 12); ++i) buffer->data[i] = 0.0f;
}

static uint32_t dma_args(unsigned tile, unsigned pitch_words) {
    /* SPM address is in elements, DRAM pitch is in bytes. */
    return ((tile * 16) << 16) | (pitch_words * sizeof(float));
}

static int check(unsigned batch, unsigned pc) {
    for (unsigned i = 0; i < GUARD; ++i)
        if (c_buf.before[i] != 0xdeadbeef || c_buf.after[i] != 0xdeadbeef) {
            printf("ALLO SYSTOLIC FAIL: DMA guard batch=%u\n", batch);
            return 1;
        }
    for (unsigned r = 0; r < M; ++r)
        for (unsigned col = 0; col < pc; ++col) {
            uint32_t expected = col < N ? bits(reference[r][col]) : 0;
#ifdef SYSTOLIC_FORCE_FAIL
            if (batch == 0 && r == 0 && col == 0) expected ^= 1;
#endif
            uint32_t got = bits(c_buf.data[r * pc + col]);
            if (got != expected) {
                printf("ALLO SYSTOLIC FAIL: batch=%u row=%u col=%u expected=%08x got=%08x\n",
                       batch, r, col, (unsigned)expected, (unsigned)got);
                return 1;
            }
        }
    return 0;
}

int main(void) {
    printf("ALLO SYSTOLIC Rocket/RoCC: 4x4 FP32 PE grid, C[%d][%d] = A[%d][%d] * B[%d][%d]\n",
           M, N, M, K, K, N);
    for (unsigned batch = 0; batch < 2; ++batch) {
        const unsigned pa = pitches[batch][0], pb = pitches[batch][1], pc = pitches[batch][2];
        fill(&a_buf);
        fill(&b_buf);
        fill(&c_buf);
        for (unsigned r = 0; r < M; ++r)
            for (unsigned k = 0; k < K; ++k) a_buf.data[r * pa + k] = random_float();
        for (unsigned k = 0; k < K; ++k)
            for (unsigned col = 0; col < N; ++col) b_buf.data[k * pb + col] = random_float();

        uint32_t start = cycles();
        for (unsigned r = 0; r < M; ++r)
            for (unsigned col = 0; col < N; ++col) {
                float v = 0.0f;
                for (unsigned k = 0; k < K; ++k) v = v + a_buf.data[r * pa + k] * b_buf.data[k * pb + col];
                reference[r][col] = v;
            }
        uint32_t software = cycles() - start;

        fence_memory();
        start = cycles();
        uint32_t ack = 0;
        for (unsigned t = 0; t < 2; ++t) {
            ack |= loadtile_a((uint32_t)(uintptr_t)&a_buf.data[t * 4 * pa], dma_args(spm_a[batch][t], pa));
            ack |= loadtile_b((uint32_t)(uintptr_t)&b_buf.data[t * 4], dma_args(spm_b[batch][t], pb));
        }
        uint32_t load = cycles() - start;
        start = cycles();
        for (unsigned ti = 0; ti < 2; ++ti)
            for (unsigned tj = 0; tj < 2; ++tj)
                ack |= systolic4x4((spm_a[batch][ti] << 16) | spm_b[batch][tj], spm_c[batch][ti * 2 + tj]);
        uint32_t compute = cycles() - start;
        start = cycles();
        for (unsigned ti = 0; ti < 2; ++ti)
            for (unsigned tj = 0; tj < 2; ++tj)
                ack |= storetile_c((uint32_t)(uintptr_t)&c_buf.data[ti * 4 * pc + tj * 4],
                                   dma_args(spm_c[batch][ti * 2 + tj], pc));
        uint32_t store = cycles() - start;
        fence_memory();
        if (ack) {
            printf("ALLO SYSTOLIC FAIL: nonzero acknowledgement batch=%u\n", batch);
            return 1;
        }
        if (check(batch, pc)) return 1;
        printf("ALLO SYSTOLIC batch=%u PASS software=%u load=%u compute=%u store=%u cycles\n",
               batch, (unsigned)software, (unsigned)load, (unsigned)compute, (unsigned)store);
    }
    printf("ALLO SYSTOLIC PASS batches=2 elements=%d\n", 2 * M * N);
    return 0;
}
