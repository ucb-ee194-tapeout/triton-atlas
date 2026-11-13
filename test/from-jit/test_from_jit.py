# RUN: %PYTHON %s
#
# Front-door test: real @triton.jit kernels go through Triton's own front end and our backend's ttir stage, then through
# our passes stage by stage. Each row says how far its kernel must get (the last passing stage) and, if it stops early,
# the error the next stage must give. A row that gets further or stops sooner than it says fails, so the table is the
# record of which kernels the pipeline takes today; flip a row when a pass learns something new.
#
# End-to-end rows also run a whole triton.compile, which must produce the "atlas_virtual" output.
#
# Run it with lit (make test T=test/from-jit) or alone: python test/from-jit/test_from_jit.py, with triton-atlas-opt
# on PATH (source env.sh after a build).
import os
import re
import shutil
import subprocess
import sys
import tempfile

# A private Triton cache, so a run never reuses or leaves compiled kernels behind.
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="ttatlas-from-jit-")

import triton  # noqa: E402
from triton._C.libtriton import ir  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler.compiler import ASTSource, make_backend  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels  # noqa: E402

TARGET = GPUTarget("atlas", "virtual", 1)

# The passes each stage runs, with the cleanup Pipeline.cpp puts between them. Stage k ends with the stage-k contract
# check (--ttatlas-verify-contract=stage=k).
CLEANUP = ["--canonicalize", "--cse"]
STAGES = {
    1: ["--ttatlas-validate-ttir"],
    2: ["--ttatlas-validate-ttir", "--ttatlas-normalize", *CLEANUP],
    3: ["--ttatlas-validate-ttir", "--ttatlas-normalize", *CLEANUP, "--convert-triton-compute-to-linalg"],
    4: [
        "--ttatlas-validate-ttir", "--ttatlas-normalize", *CLEANUP, "--convert-triton-compute-to-linalg",
        "--ttatlas-legalize-shapes", *CLEANUP
    ],
    5: [
        "--ttatlas-validate-ttir", "--ttatlas-normalize", *CLEANUP, "--convert-triton-compute-to-linalg",
        "--ttatlas-legalize-shapes", *CLEANUP, "--ttatlas-legalize-dtypes", *CLEANUP
    ],
}
LAST = max(STAGES)

FP8, BF16, FP32 = "*fp8e4nv", "*bf16", "*fp32"


def row(name, fn, signature, passes_ok=LAST, first_error=None, constexprs=None, end_to_end=False,
        end_to_end_error=None):
    return dict(name=name, fn=fn, signature=signature, constexprs=constexprs or {}, passes_ok=passes_ok,
                first_error=first_error, end_to_end=end_to_end or end_to_end_error is not None,
                end_to_end_error=end_to_end_error)


# passes_ok: the last stage the kernel passes (0 = not even stage 1). first_error: a regex the next stage's first
# "error:" line must match; required when passes_ok < LAST. end_to_end_error: a known gap in the full triton.compile,
# as a regex its error must match (an expected failure); the row fails once the compile succeeds, so flip it then.
ROWS = [
    row("mlp", kernels.mlp, {"X": FP8, "W": FP8, "B": BF16, "Y": BF16}, end_to_end=True),
    row("matmul_relu", kernels.matmul_relu, {"A": FP8, "B": FP8, "C": FP32}, end_to_end=True),
    # Expected end-to-end failure: pass 6 does not read a tile with a row stride other than 32 yet (A's stride is
    # K = 128), a planned gap for the pass 6 lane's next PR; stages 1-5 run as usual.
    row("mm32", kernels.mm32, {"A": FP8, "Bm": FP8, "C": BF16, "K": "constexpr"}, constexprs={"K": 128},
        end_to_end_error=r"cannot prove contiguous row-major tile"),
    row("bias_seeded_k_loop", kernels.bias_seeded_k_loop,
        {"A": FP8, "Bm": FP8, "Bias": BF16, "C": BF16, "K": "constexpr"}, constexprs={"K": 128}),
    row("unrolled_by_triton", kernels.unrolled_by_triton, {"A": FP8, "Bm": FP8, "C": BF16, "K": "constexpr"},
        constexprs={"K": 128}),
    row("dot_plus_dot", kernels.dot_plus_dot, {"A": FP8, "Bm": FP8, "C": BF16}),
    row("row_reduce", kernels.row_reduce, {"X": BF16, "Y": BF16}),
    # Pass 1 rejects packing BF16 back to FP8 in the kernel: no FP8 scale-code rule yet.
    row("fp8_two_layer", kernels.fp8_two_layer, {"X": FP8, "W0": FP8, "W1": FP8, "Out": BF16}, passes_ok=0,
        first_error=r"unsupported tt\.fp_to_fp: converts bf16 to f8E4M3FN"),
    # Pass 1 rejects the BF16 transpose of K (Atlas transposes FP8 only), before it reaches the BF16 dots.
    row("bf16_simple_attention", kernels.bf16_simple_attention, {"Q": BF16, "K": BF16, "V": BF16, "Out": BF16},
        passes_ok=0, first_error=r"unsupported tt\.trans: transposes bf16"),
    # Expected end-to-end failure: pass 7 splices one K chain at a time, and four output tiles each have a chain
    # (planned pass 7 fix).
    row("matmul_64", kernels.matmul_64, {"A": FP8, "Bm": FP8, "C": BF16},
        end_to_end_error=r"K-chain state must have one successor"),
    # Expected end-to-end failure: pass 6 does not read a stored tensor assembled from several tiles yet (planned for
    # the next pass 6 PR).
    row("matmul_relu_64x32", kernels.matmul_relu_64x32, {"A": FP8, "Bm": FP8, "C": BF16},
        end_to_end_error=r"must fully cover a fresh tensor.empty"),
    row("row_sum_scaled_64x16", kernels.row_sum_scaled_64x16, {"X": BF16, "Y": BF16}),
    row("softmax_16", kernels.softmax_16, {"X": BF16, "Y": BF16}),
    row("softmax_masked_20", kernels.softmax_masked_20, {"X": BF16, "Y": BF16}),
    row("row_max_64", kernels.row_max_64, {"X": BF16, "Y": BF16}),
    row("add_16", kernels.add_16, {"X": BF16, "Y": BF16}),
    row("add_rows_50", kernels.add_rows_50, {"X": BF16, "Y": BF16}),
    row("add_64x16", kernels.add_64x16, {"X": BF16, "Y": BF16}),
    row("copy_16", kernels.copy_16, {"X": BF16, "Y": BF16}),
    row("copy_rows_50", kernels.copy_rows_50, {"X": BF16, "Y": BF16}),
]


def make_ttir(r):
    """Triton's front end plus our backend's ttir stage, as text."""
    backend = make_backend(TARGET)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    src = ASTSource(fn=r["fn"], signature=r["signature"], constexprs=r["constexprs"])
    mod = src.make_ir(TARGET, options, backend.get_codegen_implementation(options), backend.get_module_map(), context)
    mod = backend.make_ttir(mod, {}, options)
    return mod.str_nodebug() if hasattr(mod, "str_nodebug") else str(mod)


def run_stage(opt, path, k):
    """Runs stages 1..k on the TTIR file; returns None on success, else the first error line."""
    p = subprocess.run([opt, path, *STAGES[k], f"--ttatlas-verify-contract=stage={k}", "-o", os.devnull],
                       capture_output=True, text=True)
    if p.returncode == 0:
        return None
    if p.returncode < 0:  # killed by a signal: a crash, whose first line is LLVM's bug-report banner
        return f"crashed (signal {-p.returncode})"
    return next((line for line in p.stderr.splitlines() if "error:" in line), p.stderr.strip() or "(no output)")


def check_row(opt, r, tmp):
    """Returns a list of failure messages for the row (empty when it behaves as the table says)."""
    if r["passes_ok"] < LAST and not r["first_error"]:
        return [f"{r['name']}: the table says it stops after stage {r['passes_ok']} but gives no first_error"]
    path = os.path.join(tmp, r["name"] + ".ttir.mlir")
    with open(path, "w") as f:
        f.write(make_ttir(r))
    failures = []
    for k in STAGES:
        err = run_stage(opt, path, k)
        if k <= r["passes_ok"] and err is not None:
            failures.append(f"{r['name']}: stage {k} should pass but fails: {err}")
            break
        if k == r["passes_ok"] + 1:
            if err is None:
                failures.append(f"{r['name']}: stage {k} should fail (/{r['first_error']}/) but passes; "
                                f"raise passes_ok in the table")
            elif not re.search(r["first_error"], err):
                failures.append(f"{r['name']}: stage {k} fails with an unexpected error: {err}")
            break
    if r["end_to_end"]:
        # MLIR prints its diagnostics to the process's stderr, not into the exception, so capture fd 2 to match an
        # expected error.
        with tempfile.TemporaryFile(mode="w+") as log:
            saved = os.dup(2)
            os.dup2(log.fileno(), 2)
            try:
                compiled = triton.compile(ASTSource(fn=r["fn"], signature=r["signature"], constexprs=r["constexprs"]),
                                          target=TARGET)
                error = None
            except Exception as e:  # noqa: BLE001 (any compile failure is a test failure)
                error = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            finally:
                os.dup2(saved, 2)
                os.close(saved)
            log.seek(0)
            stderr = log.read()
        if r["end_to_end_error"]:
            if error is None:
                failures.append(f"{r['name']}: triton.compile now succeeds; drop end_to_end_error from the table")
            elif not re.search(r["end_to_end_error"], stderr + error):
                failures.append(f"{r['name']}: triton.compile failed with an unexpected error: {error}")
        elif error is not None:
            sys.stderr.write(stderr)
            failures.append(f"{r['name']}: triton.compile failed: {error}")
        elif not compiled.asm.get("atlas_virtual"):
            failures.append(f"{r['name']}: triton.compile gave no atlas_virtual output")
    return failures


def main():
    opt = shutil.which("triton-atlas-opt")
    if opt is None:
        sys.exit("triton-atlas-opt not on PATH (build it, then source env.sh)")
    tmp = tempfile.mkdtemp(prefix="ttatlas-from-jit-ttir-")
    failures = []
    for r in ROWS:
        got = check_row(opt, r, tmp)
        print(("FAIL " if got else "ok   ") + r["name"])
        failures += got
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(os.environ["TRITON_CACHE_DIR"], ignore_errors=True)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
