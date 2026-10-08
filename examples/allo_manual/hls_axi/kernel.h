#ifndef KERNEL_H
#define KERNEL_H

extern "C" {
void gemm_fp32(
  float *v0,
  float *v1,
  float *v2,
  uint8_t v3,
  float *v4
);
} // extern "C"

#endif // KERNEL_H
