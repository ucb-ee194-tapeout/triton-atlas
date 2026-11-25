#!/usr/bin/env python3
"""Tune equivalent Triton kernels with Atlas's real model-backed cycle count.

This is intentionally small enough to show live.  Each candidate is ordinary
``@triton.jit`` code.  Atlas's existing lowering heuristics (tiling, stream
scheduling, delays, slot assignment, and spilling) run for every candidate;
the tuner ranks the resulting programs by execution-driven npu_model cycles.

Run on BWRC from the repository root, after building the backend and Atlas
tools, with ``ATLAS_MODEL_ROOT`` and ``ATLAS_MLIR_ROOT`` set to the documented
model/compiler checkouts.  ``ATLAS_OOT_BIN_DIR`` and ``TRITON_ATLAS_OPT`` may
select the tool directories explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test/numerical"))

from reference import deterministic_inputs, matmul_relu  # noqa: E402
from triton_atlas_backend.autotune import AtlasAutotuner, Candidate  # noqa: E402
from triton_atlas_backend.boundary import pack_input_tile, unpack_outputs  # noqa: E402


@triton.jit
def baseline(A, B, C):
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    a = tl.load(A + rows[:, None] * 32 + cols[None, :])
    b = tl.load(B + rows[:, None] * 32 + cols[None, :])
    acc = tl.dot(a, b, tl.full((32, 32), 0.0, tl.float32))
    tl.store(C + rows[:, None] * 32 + cols[None, :], tl.maximum(acc, 0.0))


@triton.jit
def redundant_relu(A, B, C):
    """A legal but deliberately redundant implementation for the tuner."""
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    a = tl.load(A + rows[:, None] * 32 + cols[None, :])
    b = tl.load(B + rows[:, None] * 32 + cols[None, :])
    acc = tl.dot(a, b, tl.full((32, 32), 0.0, tl.float32))
    acc = tl.maximum(acc, 0.0)
    acc = tl.maximum(acc, 0.0)
    tl.store(C + rows[:, None] * 32 + cols[None, :], acc)


@triton.jit
def triple_relu(A, B, C):
    """Another legal implementation with two redundant ReLU stages."""
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    a = tl.load(A + rows[:, None] * 32 + cols[None, :])
    b = tl.load(B + rows[:, None] * 32 + cols[None, :])
    acc = tl.dot(a, b, tl.full((32, 32), 0.0, tl.float32))
    acc = tl.maximum(acc, 0.0)
    acc = tl.maximum(acc, 0.0)
    acc = tl.maximum(acc, 0.0)
    tl.store(C + rows[:, None] * 32 + cols[None, :], acc)


SIGNATURE = {"A": "*fp8e4nv", "B": "*fp8e4nv", "C": "*fp32"}
INPUTS = deterministic_inputs()
EXPECTED = matmul_relu(*INPUTS)


def _initial_image(_candidate: Candidate, program: Mapping[str, Any]) -> bytes:
    """Pack host FP8 inputs and guards for one candidate's manifest."""
    manifest = json.loads(program["kernel_manifest"])
    input_base = manifest["dram_base"]["input"]
    output_base = manifest["dram_base"]["output"]
    model_base = input_base - 1024
    end = output_base + 2048 * len(manifest["outputs"]) + 1024
    image = bytearray(b"\xa5" * (end - model_base))
    for slot in manifest["inputs"]:
        packed = pack_input_tile(slot, INPUTS[slot["arg"]])
        offset = input_base + slot["index"] * 2048 - model_base
        image[offset:offset + len(packed)] = packed
    for slot in manifest["outputs"]:
        offset = output_base + slot["index"] * 2048 - model_base
        image[offset:offset + 2048] = b"\xcd" * 2048
    return bytes(image)


def _validate(candidate: Candidate, program: Mapping[str, Any], memory: bytes) -> None:
    """Require exact BF16 output and untouched guards before ranking cycles."""
    manifest = json.loads(program["kernel_manifest"])
    if len(manifest["inputs"]) != 2 or len(manifest["outputs"]) != 1:
        raise ValueError("demo expects two input slots and one output slot")
    input_base = manifest["dram_base"]["input"]
    output_base = manifest["dram_base"]["output"]
    model_base = input_base - 1024
    output = manifest["outputs"][0]
    host = bytearray(4096)
    window = memory[output_base - model_base:output_base - model_base + 2048]
    unpack_outputs(manifest, window, {output["arg"]: host})
    actual = b"".join(host[index + 2:index + 4] for index in range(0, 4096, 4))
    if actual != EXPECTED:
        raise ValueError(f"{candidate.name} produced incorrect BF16 output")
    input_start = input_base - model_base
    output_end = output_base - model_base + 2048
    if any(byte != 0xA5 for byte in memory[:input_start]):
        raise ValueError(f"{candidate.name} modified the pre-kernel guard")
    if any(byte != 0xA5 for byte in memory[output_end:]):
        raise ValueError(f"{candidate.name} modified the post-kernel guard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="write the JSON tuning report here")
    parser.add_argument("--atlas-bin", type=Path, default=os.environ.get("ATLAS_OOT_BIN_DIR"))
    parser.add_argument("--triton-atlas-opt", type=Path, default=os.environ.get("TRITON_ATLAS_OPT"))
    args = parser.parse_args()
    candidates = [
        Candidate("baseline", baseline, SIGNATURE),
        Candidate("redundant_relu", redundant_relu, SIGNATURE),
        Candidate("triple_relu", triple_relu, SIGNATURE),
    ]
    tuner = AtlasAutotuner(
        candidates,
        atlas_bin=args.atlas_bin,
        triton_atlas_opt=args.triton_atlas_opt,
    )
    result = tuner.tune(_initial_image, validator=_validate)
    report = result.as_dict()
    report["model_revision"] = "337e9f249f938a1dff007ddf8a763e5cd4b3aa2b"
    report["heuristic_pipeline"] = [
        "atlas-compile",
        "virtual scheduling",
        "DMA wait insertion",
        "stream scheduling and delay insertion",
        "slot assignment and spilling",
    ]
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "selected": {
            "name": result.selected.name,
            "cycles": result.selected.cycles,
        },
        "measurements": [
            {"name": item.name, "status": item.status, "cycles": item.cycles, "error": item.error}
            for item in result.measurements
        ],
        "model_revision": report["model_revision"],
        "report": str(args.report) if args.report else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
