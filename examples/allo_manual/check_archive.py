"""Verify the archive and the exact source fingerprints in its historical run."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    manifest = json.loads((HERE / "artifact-manifest.json").read_text())
    for name, entry in manifest["files"].items():
        if sha256(HERE / name) != entry["sha256"]:
            raise RuntimeError(f"Archive checksum mismatch: {name}")
    proofs = [HERE / "evidence/result.json", HERE / "evidence/reproduction/result.json"]
    inputs = 0
    for proof in proofs:
        result = json.loads(proof.read_text())
        if not result["passed"] or result["asic_backend_run"]:
            raise RuntimeError(f"Unexpected run status: {proof}")
        for name, expected in result["files_sha256"].items():
            candidates = [HERE / name, HERE / "evidence/original-inputs" / name,
                          proof.parent / name]
            if not any(sha256(path) == expected for path in candidates):
                raise RuntimeError(f"Missing validated input for {proof.name}: {name}")
            inputs += 1
    fp = result["fp_blackboxes"]
    for section, folder in [("original_hls_verilog_sha256", "hls_native/out.prj/solution1/syn/verilog"),
                            ("packaged_verilog_sha256", "rtl"),
                            ("external_rtl_sha256", "blackbox")]:
        for name, expected in fp[section].items():
            if sha256(HERE / folder / name) != expected:
                raise RuntimeError(f"RTL differs from the validated run: {folder}/{name}")
    print(f"ARCHIVE PASS files={len(manifest['files'])} validated_inputs={inputs} runs={len(proofs)}")


if __name__ == "__main__":
    main()
