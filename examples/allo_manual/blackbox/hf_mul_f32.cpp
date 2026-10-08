#include "hardfloat_model.h"
extern "C" float hf_mul_f32(float a, float b, bool &ap_ready) {
    #pragma HLS interface ap_ctrl_none port=return
    ap_ready = true;
    volatile float result = a * b;
    return hf_canonical(result);
}
