"""Compile the two-layer MLP through triton-atlas without launching it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

# Triton's active driver is selected during import. The Atlas virtual driver
# compiles only; this script creates no tensors and launches no device kernel.
os.environ["TRITON_ATLAS_TARGET"] = "virtual"

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from kernel import fp8_two_layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    source = ASTSource(
        fn=fp8_two_layer,
        signature={
            "X": "*fp8e4nv",
            "W0": "*fp8e4nv",
            "W1": "*fp8e4nv",
            "Out": "*bf16",
        },
        constexprs={},
    )
    compiled = triton.compile(source, target=GPUTarget("atlas", "virtual", 1))
    ttir = compiled.asm["ttir"]
    atlas_stage = compiled.asm["atlas_virtual"]
    if isinstance(ttir, str):
        ttir = ttir.encode()
    if isinstance(atlas_stage, str):
        atlas_stage = atlas_stage.encode()
    (out_dir / "kernel.ttir").write_bytes(ttir)
    atlas_path = out_dir / "atlas_virtual.mlir"
    atlas_path.write_bytes(atlas_stage)

    atlas_text = atlas_stage.decode()
    if "atlas.virtual_" not in atlas_text:
        raise RuntimeError(
            "the Atlas backend stage returned TTIR without Atlas virtual ops; "
            "check that the C++ conversion pass is built and registered"
        )
    repo_root = Path(__file__).resolve().parents[3]
    verify_tool = Path(os.environ.get(
        "TRITON_ATLAS_OPT", repo_root / "build/bin/triton-atlas-opt"
    ))
    if not verify_tool.is_file():
        raise FileNotFoundError(
            f"Atlas verifier not found at {verify_tool}; run `make build.tools`"
        )
    verified = subprocess.run(
        [str(verify_tool), str(atlas_path), "--verify-atlas-virtual-stream"],
        capture_output=True, text=True,
    )
    (out_dir / "atlas_verify.log").write_text(verified.stdout + verified.stderr)
    if verified.returncode:
        raise RuntimeError("Atlas virtual verification failed; see atlas_verify.log")

    report = {
        "kernel": "fp8_two_layer",
        "origin": "triton.compile(ASTSource).asm",
        "target": {"backend": "atlas", "arch": "virtual", "warp_size": 1},
        "triton_version": triton.__version__,
        "python_version": platform.python_version(),
        "gpu_launched": False,
        "ttir_sha256": hashlib.sha256(ttir).hexdigest(),
        "atlas_stage_sha256": hashlib.sha256(atlas_stage).hexdigest(),
        "atlas_virtual_ops_emitted": True,
        "atlas_virtual_operation_counts": {
            name: len(re.findall(r'"atlas\.virtual_' + re.escape(name) + r'"', atlas_text))
            for name in ("mxu_matmul", "xlu_transpose", "vpu_unary", "pack_fp8", "output_bf16")
        },
        "atlas_virtual_verified": True,
        "status": "lowered_and_verified",
    }
    (out_dir / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
