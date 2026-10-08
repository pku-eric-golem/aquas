"""FP32 bit-exact references; multiplication/addition are rounded separately."""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests/hardfloat"))
from ieee_oracle import IEEE


def emit(name, values):
    return f"static const uint32_t {name}[{len(values)}] = {{\n" + "\n".join(
        "  " + ", ".join(f"0x{x:08x}U" for x in values[n:n+8]) + ","
        for n in range(0, len(values), 8)) + "\n};\n"


def gemm(a, bt, c=None):
    ieee = IEEE(32)
    result = []
    for i in range(8):
        for j in range(8):
            acc = 0
            for k in range(8):
                product, _ = ieee.operation("mul", a[i*8+k], bt[j*8+k])
                acc, _ = ieee.operation("add", acc, product)
            value, _ = ieee.operation("add", acc, c[i*8+j] if c else 0)
            result.append(value)
    return result


def main():
    out = HERE / "software"
    out.mkdir(exist_ok=True)
    rng = random.Random(0xA110F032)
    def finite():
        return (rng.randrange(2)<<31) | (rng.randrange(123,132)<<23) | rng.randrange(1<<23)
    identity = [0x3f800000 if i==j else 0 for i in range(8) for j in range(8)]
    ordinary = [finite() for _ in range(64)]
    cases = [([0]*64, ordinary), (identity, ordinary), (ordinary, identity)]
    pools = [[0x3f800000,0xbf800000,0x3f000000,0xbf000000],
             [0x3f800001,0x3f800003,0x3f000001,0xbf800001],
             [1,0x007fffff,0x00800000,0x80800000],
             [0x7f7fffff,0xff7fffff,0x40000000,0x3f000000],
             [0,0x80000000,0x7f800000,0xff800000,0x7fc00000,0x7f800001]]
    for pool in pools:
        cases.append(([rng.choice(pool) for _ in range(64)], [rng.choice(pool) for _ in range(64)]))
    for test in range(8):
        value = finite if test<4 else lambda: rng.getrandbits(32)
        cases.append(([value() for _ in range(64)], [value() for _ in range(64)]))
    text = "#include <stdint.h>\nstruct gemm_case {uint32_t a[64],b[64],first[64],accumulated[64];};\n"
    text += "static const struct gemm_case cases[] = {\n"
    for a,b in cases:
        bt=[b[k*8+j] for j in range(8) for k in range(8)]
        first=gemm(a,bt)
        matrices=(a,b,first,gemm(a,bt,first))
        text += "  {\n" + ",\n".join("    {"+", ".join(f"0x{x:08x}U" for x in m)+"}" for m in matrices)+"\n  },\n"
    (out / "vectors.h").write_text(text+"};\n")
    a,bt,expected=([0]*512 for _ in range(3))
    for tile in range(8):
        aa,bb=([finite() for _ in range(64)] for _ in range(2))
        dd=gemm(aa,bb)
        for i in range(8):
            for j in range(8):
                at=i*64+tile*8+j
                a[at],bt[at],expected[at]=aa[i*8+j],bb[i*8+j],dd[i*8+j]
    (out / "bench_vectors.h").write_text("#include <stdint.h>\n"+emit("bench_a",a)+emit("bench_bt",bt)+emit("bench_expected",expected))
    (HERE / "vectors.json").write_text(json.dumps({"seed":"0xA110F032","precision":"FP32",
        "rounding":"RNE, separate multiply/add", "test_tiles":16,"benchmark_tiles":8},indent=2)+"\n")
    print("Generated 16 FP32 test cases and 8 dense benchmark tiles")


if __name__ == "__main__": main()
