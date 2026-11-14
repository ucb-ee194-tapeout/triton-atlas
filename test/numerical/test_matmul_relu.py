"""Opt-in numerical execution, separate from the compile-only handoff regression."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compile_matmul_relu import DEFAULT_TRITON_ATLAS_OPT, ROOT, probe
from reference import deterministic_inputs, f32_values_from_bf16, matmul_relu


def check_case(output, atlas_bin, model_python, model_root, dram_base, inputs, expected,
               triton_atlas_opt=DEFAULT_TRITON_ATLAS_OPT):
    from triton_atlas_backend.boundary import pack_input_tile, unpack_output_tile
    sys.path.insert(0, str(ROOT / "third_party/atlas-mlir/python"))
    from atlas_mlir_native.layout import unpack_bf16_pair_halves

    output.mkdir()
    probe(output, atlas_bin, dram_base, triton_atlas_opt=triton_atlas_opt)
    manifest = json.loads((output / "kernel.json").read_text())
    if (manifest["grid"] != [1, 1, 1] or len(manifest["inputs"]) != 2 or len(manifest["outputs"]) != 1):
        raise ValueError("expected the one-tile matmul_relu boundary")
    for slot in manifest["inputs"] + manifest["outputs"]:
        if slot["layout"] != "row_major" or slot["logical_shape"] != [32, 32]:
            raise ValueError("expected full row-major tiles")
    slot = manifest["outputs"][0]
    if {(entry["arg"], entry["role"]) for entry in manifest["inputs"]} != {(0, "activation_fp8"), (1, "weight_fp8_kn")}:
        raise ValueError("expected A activation and row-major B[K,N] weight slots")
    if slot["arg"] != 2 or slot["role"] != "output_bf16" or slot["host_dtype"] != "f32":
        raise ValueError("expected BF16 device output with f32 host dtype")
    program = json.loads((output / "kernel.program.json").read_text())
    matmuls = [row for row in program["instructions"] if row["operation"] == "atlas.mxu_matmul"]
    if len(matmuls) != 1 or matmuls[0]["fields"]["unit"] != 0 or matmuls[0]["fields"]["accumulate"]:
        raise ValueError("reference requires one reset MXU0 contraction")

    # Every device address is derived from the exported manifest. Guard both
    # sides of the slot region and the unused second half of each FP8 slot.
    input_base, output_base = manifest["dram_base"]["input"], manifest["dram_base"]["output"]
    base = min(input_base, output_base) - 1024
    end = max(input_base + 2048 * len(manifest["inputs"]), output_base + 2048 * len(manifest["outputs"])) + 1024
    initial = bytearray(b"\xa5" * (end - base))
    addresses = []
    for entry in manifest["inputs"]:
        if entry["arg"] not in (0, 1) or entry["host_dtype"] != "f8E4M3FN":
            raise ValueError("expected FP8 A/B input arguments")
        address = input_base + entry["index"] * 2048
        packed = pack_input_tile(inputs[entry["arg"]], (32, 32), entry)
        initial[address - base:address - base + len(packed)] = packed
        addresses.append(dict(arg=entry["arg"], address=address, bytes=len(packed)))
    output_address = output_base + slot["index"] * 2048
    offset = output_address - base
    initial[offset:offset + 2048] = b"\xcd" * 2048
    request = output / "model-request.json"
    request.write_text(json.dumps(dict(base=base, initial_hex=initial.hex())) + "\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(model_root) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run([
        str(model_python),
        str(ROOT / "test/numerical/run_model.py"), "--program",
        str(output / "kernel.program.json"), "--request",
        str(request), "--out",
        str(output / "memory.bin"), "--stats",
        str(output / "model-stats.json")
    ], env=environment, check=True)
    memory = (output / "memory.bin").read_bytes()
    if len(memory) != len(initial):
        raise ValueError("model returned an incomplete memory image")
    physical = memory[offset:offset + 2048]
    # DMA writes the two 32x16 BF16 register halves consecutively. Reuse
    # Atlas's physical ABI conversion before our row-major crop/widen helper.
    actual = unpack_bf16_pair_halves(physical)
    host = bytearray(4096)
    unpack_output_tile(actual, host, (32, 32), slot)
    values = list(struct.unpack("<1024f", host))
    reference = f32_values_from_bf16(expected)
    expected_host = b"".join(b"\0\0" + expected[i:i + 2] for i in range(0, 2048, 2))
    mismatches = [
        dict(row=i // 32, column=i % 32, actual=values[i], expected=reference[i],
             actual_bf16=actual[2 * i:2 * i + 2].hex(), expected_bf16=expected[2 * i:2 * i + 2].hex())
        for i in range(1024)
        if actual[2 * i:2 * i + 2] != expected[2 * i:2 * i + 2] or host[4 * i:4 * i + 4] != expected_host[4 * i:4 * i +
                                                                                                          4]
    ]
    guard_mismatches = [
        base + i
        for i, (before, after) in enumerate(zip(initial, memory))
        if not offset <= i < offset + 2048 and before != after
    ]
    errors = [abs(a - b) if math.isfinite(a) else math.inf for a, b in zip(values, reference)]
    report = dict(kernel="matmul_relu", arithmetic="MXU0 K-ordered BF16 RNE per MAC, then ReLU", inputs=addresses,
                  output_address=output_address, device_dtype="bf16", host_dtype="f32",
                  physical_layout="bf16_pair_halves", exact_matches=1024 - len(mismatches), max_abs_error=max(errors),
                  mismatches=mismatches, guard_mismatches=guard_mismatches, model=json.loads(
                      (output / "model-stats.json").read_text()))
    (output / "expected.bf16").write_bytes(expected)
    (output / "actual.physical.bf16").write_bytes(physical)
    (output / "actual.bf16").write_bytes(actual)
    (output / "actual.f32").write_bytes(host)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key not in ("mismatches", "guard_mismatches")}
    summary.update(mismatch_count=len(mismatches), first_mismatches=mismatches[:5],
                   guard_mismatch_count=len(guard_mismatches), full_report=str(output / "comparison.json"))
    print(json.dumps(summary, indent=2), flush=True)
    return not mismatches and not guard_mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh artifact directory")
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--atlas-bin", type=Path, default=ROOT / "build/atlas/bin")
    parser.add_argument("--triton-atlas-opt", type=Path, default=DEFAULT_TRITON_ATLAS_OPT)
    parser.add_argument("--dram-base", type=lambda value: int(value, 0), action="append")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = deterministic_inputs()
    expected = matmul_relu(*inputs)
    passed = True
    with tempfile.TemporaryDirectory(prefix="ttatlas-numerical-cache-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        for base in args.dram_base or (0x90000000, 0x91000000):
            passed = check_case(args.out.resolve() / f"{base:08x}", args.atlas_bin.resolve(),
                                args.model_python.absolute(), args.model_root.resolve(), base, inputs, expected,
                                triton_atlas_opt=args.triton_atlas_opt.absolute()) and passed
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
