"""Allo FP32 tile GEMM: D = A * BT.T + (use_c ? C : 0), 8x8x8."""
import allo
from allo.ir.types import float32, uint8


def gemm_fp32(A: float32[64], BT: float32[64], C: float32[64],
              use_c: uint8, D: float32[64]):
    # An explicit accumulation tensor exposes the perfect reduction loop nest
    # used by the official Allo Vivado/Vitis GEMM scheduling tutorial.
    acc: float32[8, 8] = 0.0
    for i, j in allo.grid(8, 8, name="mac"):
        for k in allo.reduction(8):
            product: float32 = A[i * 8 + k] * BT[j * 8 + k]
            acc[i, j] = acc[i, j] + product
    for u, v in allo.grid(8, 8, name="outputs"):
        initial: float32 = 0.0
        if use_c != 0:
            initial = C[u * 8 + v]
        D[u * 8 + v] = acc[u, v] + initial
