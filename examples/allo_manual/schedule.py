"""Schedule primitives from the official Allo FP32 Vitis GEMM tutorial.

https://cornell-zhang.github.io/allo/gallery/tutorial_02_vhls.html
The k order for each output remains unchanged (no reassociation or fast math).
"""
import allo
from gemm import gemm_fp32

PRIMITIVES = [
    'reorder("k", "j")',
    'buffer_at(s.acc, axis="i")',
    'pipeline("j", initiation_interval=1)',
    'pipeline("v", initiation_interval=1)',
]


def build_schedule(partition_acc=True):
    s = allo.customize(gemm_fp32)
    s.reorder("k", "j")
    s.buffer_at(s.acc, axis="i")
    s.pipeline("j", initiation_interval=1)
    s.pipeline("v", initiation_interval=1)
    if partition_acc:
        s.partition(s.acc, dim=0)
    return s
