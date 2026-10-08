#ifndef ALLO_HARDFLOAT_MODEL_H
#define ALLO_HARDFLOAT_MODEL_H
#include <cstdint>
#include <cstring>

// The native RISC-V specialization returns canonical NaNs. C models are used
// only by host simulation; the blackbox RTL implements the arithmetic.
static inline float hf_canonical(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        bits = 0x7fc00000U;
        std::memcpy(&value, &bits, sizeof(bits));
    }
    return value;
}
extern "C" float hf_add_f32(float a, float b, bool &ap_ready);
extern "C" float hf_mul_f32(float a, float b, bool &ap_ready);
#endif
