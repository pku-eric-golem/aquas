"""Bit-exact checks of the Allo FP32 LLVM kernel against an integer oracle."""
import json
import random
from pathlib import Path
import numpy as np
import allo
from gemm import gemm_fp32
from generate_vectors import gemm
from schedule import build_schedule

HERE = Path(__file__).resolve().parent


def main():
    rng = random.Random(0xA110F032)
    module = build_schedule().build()
    for trial in range(8):
        def values():
            return [(rng.randrange(2)<<31)|(rng.randrange(123,132)<<23)|rng.randrange(1<<23) for _ in range(64)]
        aa,bb,cc=values(),values(),values()
        a=np.array(aa,dtype=np.uint32).view(np.float32)
        b=np.array(bb,dtype=np.uint32).view(np.float32)
        c=np.array(cc,dtype=np.uint32).view(np.float32)
        d=np.zeros(64,dtype=np.float32)
        module(a,b,c,trial%2,d)
        expected=np.array(gemm(aa,bb,cc if trial%2 else None),dtype=np.uint32)
        np.testing.assert_array_equal(d.view(np.uint32),expected)
    result={"passed":True,"precision":"FP32","checked":512,"implementation":"Allo LLVM JIT",
            "reference":"integer IEEE oracle; separate multiply/add RNE"}
    (HERE/"allo-validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":main()
