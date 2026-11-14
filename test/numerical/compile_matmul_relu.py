"""Compile-only connection probe. Success never establishes numerical correctness."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRITON_ATLAS_OPT = Path(os.environ.get("TRITON_ATLAS_OPT", ROOT / "build/bin/triton-atlas-opt"))


def run(command):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, command))}\n{result.stderr.strip()}")
    return result.stdout


def _lower_and_emit(output, atlas_bin, report, debug=False, triton_atlas_opt=DEFAULT_TRITON_ATLAS_OPT):
    """Own the complete machine handoff; callers never run cleanup separately."""
    printing = ["--mlir-print-debuginfo"] if debug else []
    # Atlas-only tools do not register the ttatlas metadata dialect.
    run([
        atlas_bin / "atlas-opt", output / "kernel.virtual.mlir", "--allow-unregistered-dialect", "--atlas-compile",
        *printing, "-o", output / "kernel.machine.mlir"
    ])
    report["machine_lowering"] = True
    run([
        triton_atlas_opt, output / "kernel.machine.mlir", "--ttatlas-finalize-machine-handoff", *printing, "-o",
        output / "kernel.emit.mlir"
    ])
    program = run([atlas_bin / "atlas-emit", "--program-json", output / "kernel.emit.mlir"])
    (output / "kernel.program.json").write_text(program)
    report["instruction_emission"] = True


def probe(output, atlas_bin, dram_base, debug=False, triton_atlas_opt=DEFAULT_TRITON_ATLAS_OPT):
    # Set the cache before importing Triton, including through the backend.
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    sys.path.insert(0, str(ROOT / "test/from-jit"))
    from kernels import matmul_relu

    report = {
        "triton_compile": False, "machine_lowering": False, "instruction_emission": False, "numerical_execution": False,
        "debug": debug
    }
    try:
        compiled = triton.compile(ASTSource(matmul_relu, {"A": "*fp8e4nv", "B": "*fp8e4nv", "C": "*fp32"}),
                                  target=GPUTarget("atlas", "virtual", 1), options={"dram_base": dram_base})
        virtual = compiled.asm["atlas_virtual"]
        if isinstance(virtual, bytes):
            virtual = virtual.decode()
        (output / "kernel.virtual.mlir").write_text(virtual)
        (output / "kernel.json").write_text(compiled.metadata.ttatlas_kernel)
        report["triton_compile"] = True
        _lower_and_emit(output, atlas_bin, report, debug=debug, triton_atlas_opt=triton_atlas_opt)
    except (RuntimeError, OSError) as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "compile-report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh artifact directory")
    parser.add_argument("--atlas-bin", type=Path, default=ROOT / "build/atlas/bin")
    parser.add_argument("--triton-atlas-opt", type=Path, default=DEFAULT_TRITON_ATLAS_OPT,
                        help="cleanup tool (default: TRITON_ATLAS_OPT or this repo's build/bin)")
    parser.add_argument("--dram-base", type=lambda value: int(value, 0), default=0x90000000)
    parser.add_argument("--debug", action="store_true", help="preserve source locations in machine MLIR artifacts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ttatlas-numerical-cache-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        try:
            probe(args.out.resolve(), args.atlas_bin.resolve(), args.dram_base, debug=args.debug,
                  triton_atlas_opt=args.triton_atlas_opt.absolute())
        except (RuntimeError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 1
    print("Compile-only probe passed; no numerical execution performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
