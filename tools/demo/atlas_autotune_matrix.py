#!/usr/bin/env python3
"""Run model-backed autotuning across small kernels used by larger models.

The matrix intentionally stays on static, single-tile kernels first.  This
lets us separate compiler support from model timing while exercising the same
boundary packing, guard validation, and candidate ranking as the demo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test/numerical"))

from reference import bf16, deterministic_inputs, fp8, round_bf16  # noqa: E402
from triton_atlas_backend.autotune import AtlasAutotuner, Candidate  # noqa: E402
from triton_atlas_backend.boundary import pack_input_tile, unpack_outputs  # noqa: E402


@triton.jit
def add16(X, Y):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 16)
    off = rows[:, None] * 16 + cols[None, :]
    tl.store(Y + off, (tl.load(X + off).to(tl.float32) * 2.0).to(tl.bfloat16))


@triton.jit
def add16_reassociated(X, Y):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 16)
    off = rows[:, None] * 16 + cols[None, :]
    x = tl.load(X + off).to(tl.float32)
    x = (x + 1.0) + (x - 1.0)
    tl.store(Y + off, x.to(tl.bfloat16))


@triton.jit
def mlp32(X, W, B, Y):
    rows = tl.arange(0, 32)
    off = rows[:, None] * 32 + rows[None, :]
    x = tl.load(X + off)
    w = tl.load(W + off)
    bias = tl.load(B + rows).to(tl.float32)
    acc = tl.dot(x, tl.trans(w)) + bias[None, :]
    tl.store(Y + off, tl.maximum(acc, 0.0).to(tl.bfloat16))


@triton.jit
def mlp32_redundant_relu(X, W, B, Y):
    rows = tl.arange(0, 32)
    off = rows[:, None] * 32 + rows[None, :]
    x = tl.load(X + off)
    w = tl.load(W + off)
    bias = tl.load(B + rows).to(tl.float32)
    acc = tl.dot(x, tl.trans(w)) + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    acc = tl.maximum(acc, 0.0)
    tl.store(Y + off, acc.to(tl.bfloat16))


@triton.jit
def row_reduce32(X, Y):
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    off = rows[:, None] * 32 + cols[None, :]
    x = tl.load(X + off).to(tl.float32)
    maximum = tl.max(x, axis=1)
    total = tl.sum(x - maximum[:, None], axis=1)
    tl.store(Y + rows, total.to(tl.bfloat16))


@triton.jit
def row_reduce32_reassociated(X, Y):
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    off = rows[:, None] * 32 + cols[None, :]
    x = tl.load(X + off).to(tl.float32)
    maximum = tl.max(x, axis=1)
    total = tl.sum(x - maximum[:, None], axis=1) + 0.0
    tl.store(Y + rows, total.to(tl.bfloat16))


@triton.jit
def softmax16(X, Y):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 16)
    off = rows[:, None] * 16 + cols[None, :]
    x = tl.load(X + off).to(tl.float32)
    exponent = tl.exp(x - tl.max(x, axis=1)[:, None])
    probability = exponent * (1.0 / tl.sum(exponent, axis=1))[:, None]
    tl.store(Y + off, probability.to(tl.bfloat16))


@triton.jit
def softmax16_reassociated(X, Y):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 16)
    off = rows[:, None] * 16 + cols[None, :]
    x = tl.load(X + off).to(tl.float32)
    exponent = tl.exp(x - tl.max(x, axis=1)[:, None])
    probability = exponent * (1.0 / tl.sum(exponent, axis=1))[:, None]
    tl.store(Y + off, (probability + 0.0).to(tl.bfloat16))


@triton.jit
def exp16(X, Y):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 16)
    off = rows[:, None] * 16 + cols[None, :]
    tl.store(Y + off, tl.exp(tl.load(X + off).to(tl.float32)).to(tl.bfloat16))


@triton.jit
def exp16_reassociated(X, Y):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 16)
    off = rows[:, None] * 16 + cols[None, :]
    value = tl.exp(tl.load(X + off).to(tl.float32))
    tl.store(Y + off, (value + 0.0).to(tl.bfloat16))


@dataclass(frozen=True)
class Workload:
    name: str
    candidates: tuple[Candidate, ...]
    inputs: Mapping[int, bytes]
    expected: Mapping[int, bytes]
    max_bf16_ulp: int = 0


def _bf16_buffer(codes: list[int]) -> bytes:
    return b"".join(code.to_bytes(2, "little") for code in codes)


def _add16_workload() -> Workload:
    values = [0x3F80, 0x3F00, 0xBF80, 0x3E80] * 64
    source = _bf16_buffer(values)
    expected = _bf16_buffer([round_bf16(bf16(value) * 2) for value in values])
    signature = {"X": "*bf16", "Y": "*bf16"}
    return Workload(
        "add16",
        (Candidate("add16", add16, signature), Candidate("add16_reassociated", add16_reassociated, signature)),
        {0: source},
        {1: expected},
    )


def _mlp_expected(x: bytes, w: bytes, bias: bytes) -> bytes:
    x_values, w_values = list(map(fp8, x)), list(map(fp8, w))
    bias_values = [bf16(int.from_bytes(bias[index:index + 2], "little")) for index in range(0, len(bias), 2)]
    output: list[int] = []
    for row in range(32):
        for col in range(32):
            accumulator: Any = 0
            for inner in range(32):
                accumulator = round_bf16(
                    bf16(accumulator) + x_values[row * 32 + inner] * w_values[col * 32 + inner]
                )
            value = bf16(accumulator) + bias_values[col]
            output.append(0 if value <= 0 else round_bf16(value))
    return _bf16_buffer(output)


def _mlp_workload() -> Workload:
    x, w = deterministic_inputs()
    bias = _bf16_buffer([0x3F80, 0xBF00, 0x3E80, 0x0000] * 8)
    expected = _mlp_expected(x, w, bias)
    signature = {"X": "*fp8e4nv", "W": "*fp8e4nv", "B": "*bf16", "Y": "*bf16"}
    return Workload(
        "mlp32",
        (Candidate("mlp32", mlp32, signature), Candidate("mlp32_redundant_relu", mlp32_redundant_relu, signature)),
        {0: x, 1: w, 2: bias},
        {3: expected},
        max_bf16_ulp=1,
    )


def _row_reduce_workload() -> Workload:
    values = [0x3F80, 0x3F00, 0xBF80, 0x3E80] * 256
    source = _bf16_buffer(values)
    expected = []
    for row in range(32):
        row_values = [bf16(values[row * 32 + col]) for col in range(32)]
        maximum = max(row_values)
        expected.append(round_bf16(sum(value - maximum for value in row_values)))
    signature = {"X": "*bf16", "Y": "*bf16"}
    return Workload(
        "row_reduce32",
        (
            Candidate("row_reduce32", row_reduce32, signature),
            Candidate("row_reduce32_reassociated", row_reduce32_reassociated, signature),
        ),
        {0: source},
        {1: _bf16_buffer(expected)},
        max_bf16_ulp=1,
    )


def _softmax_workload() -> Workload:
    values = [0x0000, 0x3D80, 0xBD80, 0x3E80] * 64
    source = _bf16_buffer(values)
    expected = []
    for row in range(16):
        row_values = [float(bf16(values[row * 16 + col])) for col in range(16)]
        maximum = max(row_values)
        exponent = [math.exp(value - maximum) for value in row_values]
        denominator = sum(exponent)
        expected.extend(round_bf16(value / denominator) for value in exponent)
    signature = {"X": "*bf16", "Y": "*bf16"}
    return Workload(
        "softmax16",
        (
            Candidate("softmax16", softmax16, signature),
            Candidate("softmax16_reassociated", softmax16_reassociated, signature),
        ),
        {0: source},
        {1: _bf16_buffer(expected)},
        max_bf16_ulp=2,
    )


def _exp_workload() -> Workload:
    values = [0x0000, 0x3D80, 0xBD80, 0x3E80] * 64
    source = _bf16_buffer(values)
    expected = _bf16_buffer([round_bf16(math.exp(float(bf16(value)))) for value in values])
    signature = {"X": "*bf16", "Y": "*bf16"}
    return Workload(
        "exp16",
        (Candidate("exp16", exp16, signature), Candidate("exp16_reassociated", exp16_reassociated, signature)),
        {0: source},
        {1: expected},
        max_bf16_ulp=2,
    )


def _initial_image(workload: Workload, program: Mapping[str, Any]) -> bytes:
    manifest = json.loads(program["kernel_manifest"])
    input_base = manifest["dram_base"]["input"]
    output_base = manifest["dram_base"]["output"]
    model_base = input_base - 1024
    end = max(
        input_base + 2048 * max((slot["index"] for slot in manifest["inputs"]), default=-1) + 2048,
        output_base + 2048 * max((slot["index"] for slot in manifest["outputs"]), default=-1) + 2048,
    ) + 1024
    image = bytearray(b"\xa5" * (end - model_base))
    for slot in manifest["inputs"]:
        if slot["kind"] == "memory":
            packed = pack_input_tile(slot, workload.inputs[slot["arg"]])
            offset = input_base + slot["index"] * 2048 - model_base
            image[offset:offset + len(packed)] = packed
    for slot in manifest["outputs"]:
        offset = output_base + slot["index"] * 2048 - model_base
        image[offset:offset + 2048] = b"\xcd" * 2048
    return bytes(image)


def _validator(workload: Workload):
    def ordered_bf16(bits: int) -> int:
        return (~bits & 0xFFFF) if bits & 0x8000 else bits | 0x8000

    def validate(_candidate: Candidate, program: Mapping[str, Any], memory: bytes) -> None:
        manifest = json.loads(program["kernel_manifest"])
        input_base = manifest["dram_base"]["input"]
        output_base = manifest["dram_base"]["output"]
        model_base = input_base - 1024
        host = {arg: bytearray(value) for arg, value in workload.expected.items()}
        output_window = memory[output_base - model_base:]
        unpack_outputs(manifest, output_window, host)
        for arg, expected in workload.expected.items():
            actual = bytes(host[arg])
            bad = []
            for offset in range(0, min(len(actual), len(expected)), 2):
                got = int.from_bytes(actual[offset:offset + 2], "little")
                want = int.from_bytes(expected[offset:offset + 2], "little")
                if abs(ordered_bf16(got) - ordered_bf16(want)) > workload.max_bf16_ulp:
                    bad.append(offset)
            if len(actual) != len(expected) or bad:
                first_diff = bad[0] if bad else min(len(actual), len(expected))
                raise ValueError(
                    f"output argument {arg} differs from the reference; "
                    f"first_diff_byte={first_diff}; "
                    f"actual_window={actual[max(0, first_diff - 8):first_diff + 8].hex()} "
                    f"expected_window={expected[max(0, first_diff - 8):first_diff + 8].hex()}; "
                    f"max_allowed_ulp={workload.max_bf16_ulp}; "
                    f"output_slot={manifest['outputs']}; "
                    f"operations={[row['operation'] for row in program['instructions']]}"
                )
        output_end = output_base - model_base + 2048 * len(manifest["outputs"])
        if any(byte != 0xA5 for byte in memory[:input_base - model_base]):
            raise ValueError("pre-kernel guard changed")
        if any(byte != 0xA5 for byte in memory[output_end:]):
            raise ValueError("post-kernel guard changed")

    return validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--atlas-bin", type=Path, default=os.environ.get("ATLAS_OOT_BIN_DIR"))
    parser.add_argument("--triton-atlas-opt", type=Path, default=os.environ.get("TRITON_ATLAS_OPT"))
    args = parser.parse_args()
    workloads = (_add16_workload(), _mlp_workload(), _row_reduce_workload(), _softmax_workload(), _exp_workload())
    reports = []
    for workload in workloads:
        tuner = AtlasAutotuner(workload.candidates, atlas_bin=args.atlas_bin, triton_atlas_opt=args.triton_atlas_opt)
        try:
            result = tuner.tune(
                lambda _candidate, program: _initial_image(workload, program), validator=_validator(workload)
            )
            reports.append({
                "workload": workload.name,
                "status": "ok",
                "selected": {"name": result.selected.name, "cycles": result.selected.cycles},
                "measurements": [item.as_dict() for item in result.measurements],
            })
        except RuntimeError as error:
            reports.append({"workload": workload.name, "status": "no_valid_candidate", "error": str(error)})
    report = {"model_revision": "337e9f249f938a1dff007ddf8a763e5cd4b3aa2b", "workloads": reports}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
