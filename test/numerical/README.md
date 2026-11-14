# Numerical verification of the real matmul_relu kernel

`test_matmul_relu.py` compiles the existing `@triton.jit` kernel from
`test/from-jit/kernels.py`, emits Atlas machine instructions, executes them on
npu_model with `numerics="rtl"`, and compares all 1024 outputs against an
independent CPU reference. Numerical execution is opt-in and separate from
compile-only tests. The Triton runtime launcher remains unimplemented.

## Build and run

From the repository root, build matching tools from the vendored source:

```sh
source env.sh
export PATH="$PWD/.venv/bin:$PATH"
cmake -S third_party/atlas-mlir -B build/atlas -G Ninja \
  -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_C_COMPILER="$LLVM_INSTALL_DIR/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_INSTALL_DIR/bin/clang++"
cmake --build build/atlas --target atlas-opt atlas-emit -j 4
cmake --build build/plugin --target triton-atlas-opt -j 4
cmake --install build/plugin --component plugin \
  --prefix "$PWD/.venv/lib/python3.14/site-packages"

# Portable reference, boundary, and compile-only regression tests.
.venv/bin/python test/numerical/test_reference.py
.venv/bin/python test/compiler/test_boundary_buffers.py
.venv/bin/python test/numerical/test_handoff.py

# Actual instruction execution in the separate model Python environment.
.venv/bin/python test/numerical/test_matmul_relu.py \
  --out /tmp/matmul-relu-numerical \
  --model-python /bwrcq/home/amy.yao/atlas-eval-venv/bin/python \
  --model-root /bwrcq/home/amy.yao/npu-model-eval
```

Use a fresh output directory. By default execution runs at input bases
`0x90000000` and `0x91000000`; `--dram-base` can select other bases. Each case
preserves virtual MLIR, the verbatim `ttatlas.kernel` manifest, machine MLIR
before/after handoff, emitted program JSON, the memory request and returned
memory, model statistics/trace, physical and logical BF16 output, widened
FP32 output, and `comparison.json` with every mismatch. Nonzero mismatches,
changed guard/input bytes, failed halt, or runtime errors fail the test.

Tool and model locations are configurable: `--atlas-bin` selects the Atlas
tools, `--triton-atlas-opt` selects the internal cleanup tool, and
`--model-python` / `--model-root` select the model environment and source.
The cleanup tool defaults to `TRITON_ATLAS_OPT` when set, otherwise this
repository's `build/bin/triton-atlas-opt`. Both compile-only and numerical
commands accept these tool overrides. The documented BWRC model paths are
examples, not defaults. The numerical test's two default input bases are test
placements; actual input/output addresses are always read from each compiled
kernel's manifest.

`compile_matmul_relu.py --out <fresh-directory>` performs only compilation
and emission, including automatic internal metadata cleanup. Debug printing
is off by default. Enable `--debug` to retain source locations in both the
lowered and emission-ready machine MLIR:

```sh
# Normal machine MLIR, without extra printed source locations.
.venv/bin/python test/numerical/compile_matmul_relu.py --out /tmp/matmul-relu-compile

# Machine MLIR with source/debug locations preserved through both tools.
.venv/bin/python test/numerical/compile_matmul_relu.py --debug --out /tmp/matmul-relu-debug
```

Both modes automatically run Atlas compilation, internal MLIR-aware cleanup,
and instruction emission. Neither requires a separate cleanup command.
`--debug` passes `--mlir-print-debuginfo` to both `atlas-opt` and
`triton-atlas-opt`; it does not change instructions, addresses, arithmetic,
or the host manifest. The native cleanup never removes locations or debug
attributes: in default mode source locations are simply omitted from the
printed machine MLIR. Emitter-supported debugging attributes remain intact
in either mode. The original Triton-produced virtual MLIR is saved as-is.

`compile-report.json` records the selected debug mode and deliberately records no numerical
execution; execution evidence is in `model-stats.json` and `comparison.json`.
`handoff.mlir` registers the compile-only regression with lit. In restricted
sandboxes where Python 3.14's default forkserver cannot bind a Unix socket,
run lit with the fork multiprocessing context, without changing the tests:

```sh
.venv/bin/python -c 'import multiprocessing; multiprocessing.set_start_method("fork"); from lit.main import main; main()' \
  -j 1 -sv test/numerical/handoff.mlir test/compiler/boundary-buffers.mlir
```

Additional validation commands used for PR review:

```sh
# Existing Atlas compilation pipeline (three tests).
ATLAS_OOT_BIN_DIR="$PWD/build/atlas/bin" .venv/bin/python -m unittest discover \
  -s third_party/atlas-mlir/test -p test_atlas_compile.py -v

# Broader existing compiler tests, including metadata export.
.venv/bin/python -c 'import multiprocessing; multiprocessing.set_start_method("fork"); from lit.main import main; main()' \
  -j 1 -sv test/numerical/handoff.mlir test/compiler

# Repository-pinned tools: Ruff 0.9.1, YAPF 0.43.0, clang-format 19.1.6.
ruff check python/triton_atlas_backend/boundary.py test/compiler/test_boundary_buffers.py test/numerical/*.py
yapf --diff python/triton_atlas_backend/boundary.py test/compiler/test_boundary_buffers.py test/numerical/*.py
clang-format --dry-run --Werror include/TTAtlas/TTAtlas.h plugin.cpp lib/TTAtlas/FinalizeMachineHandoff.cpp
git diff --check
```

The broader compiler, handoff and boundary tests pass when the standalone
`triton-atlas-opt` target is rebuilt explicitly. It is excluded from the
default CMake build, so rebuilding only the plugin can leave a stale tool.
An older local binary caused a `linalg.generic` interface-registration
crash; both current remote main (`a1c4606`) and this branch pass with their
matching rebuilt tools.

## Machine handoff

The orchestration owns the following internal sequence after generating
`kernel.virtual.mlir` (shown for explanation, not as required manual steps):

```sh
build/atlas/bin/atlas-opt kernel.virtual.mlir --allow-unregistered-dialect \
  --atlas-compile -o kernel.machine.mlir
build/bin/triton-atlas-opt kernel.machine.mlir \
  --ttatlas-finalize-machine-handoff -o kernel.emit.mlir
build/atlas/bin/atlas-emit --program-json kernel.emit.mlir
```

With `--debug`, the orchestration adds `--mlir-print-debuginfo` to the first
two commands. The emitter receives the same verified instruction stream.

Pass 7 creates typed module placement tags and the kernel manifest.
Atlas machine lowering copies all module attributes, including the frontend
`#ttatlas<layout row_major>` attribute. The standalone Atlas emitter owns only
the Atlas dialect and cannot parse that custom attribute.

The explicit frontend-owned handoff pass runs after Atlas compilation. It
requires a generated, timed, verified machine artifact and removes only the
obsolete module tags `ttatlas.layout`, `ttatlas.dram_base`,
`ttatlas.fold_bias_into_dot`, `ttatlas.precision_policy`, and optional
`ttatlas.legalized`. It rejects other TTAtlas tags or TTAtlas tags on
instructions. It preserves all Atlas instructions and attributes, including
addresses, generated contracts, timing state and provider. The host manifest
remains in the sidecar. The pass is not added to Passes 1–7.

The regression exercises the CLI in both default and debug modes, compiles
the real JIT kernel at a relocated base, parses its
machine IR with `triton-atlas-opt`, verifies exact textual identity of every
instruction and all non-TTAtlas module attributes after normalizing with the
same parser/printer, emits the result, tests idempotence, and checks rejection
of premature handoff and unexpected metadata. It also checks preservation of
explicit source, fused and named locations and emitter-supported debugging
attributes. Both modes must produce identical host manifests and emitted
programs, and default machine output must omit extra location annotations.

## Arithmetic and layout

The selected RTL is `atlas-npu` revision
`0079c0541111197741a231c002e3843fa6f545b2`. Its
`SystolicArrayParams.scala` defaults to `PEArchitecture.CustomFMA`.
`sa/PE.scala` instantiates `E4M3FMA` with a BF16 addend and output.
`E4M3FMA.scala` chains `E4M3Mul` with `E4M3ProdAddBF16`, which rounds each PE
result using guard/round/sticky bits, nearest with ties to even. The normal
FP8 product fits exactly in the product's seven fraction bits. This is BF16
accumulation at every K step, not an FP32 accumulator rounded only on readout.
The custom unit excludes NaN, infinity and subnormals and clamps overflow.

The local npu_model revision is
`6eb3894032b4ac0c50a7fe4eef667015e9dfc73b`.
`configs/isa_definition.py:_vmatmul` starts a reset contraction at BF16 zero,
interprets weights as `A @ W.T`, and selects
`RtlNumerics.systolic_matmul` for MXU0. That implementation loops over K and
applies its custom FMA at every step. The test requires exactly one reset
MXU0 contraction in the emitted instructions. The compiler inserts the
transpose required by the row-major `weight_fp8_kn` B input.

`reference.py` uses exact Python rational arithmetic, independently of Torch
and npu_model. It rounds every exact FP8-product-plus-BF16-accumulator to BF16
nearest-even in ascending K order, then applies ReLU. A fixed LCG generates
encoded FP8 zeros and signed normal values with magnitudes .5–2. There is no
input quantization ambiguity. This bounded diagnostic matches the executed
fixture exactly; it does not claim to emulate all custom FMA exceptional or
shift-right-jam corner cases. Five portable tests cover decoding, ties, carry,
K-order rounding, determinism, ReLU and row-major orientation.

Every input/output address comes from the kernel manifest's base plus
`slot.index * 2048`; `slot.arg` selects the host buffer. FP8 payloads are
1024 bytes in row-major order. BF16 device output is 2048 bytes in the
physical **bf16_pair_halves** layout: all 32 rows of columns 0–15, then all
32 rows of columns 16–31. This follows the machine lowering's two VSTOREs and
npu_model's `write_mrf_bf16_tile`. The test reuses atlas-mlir's
`atlas_mlir_native.layout.unpack_bf16_pair_halves` before calling the existing
row-major `unpack_output_tile` crop/widen helper. The manifest's `layout` is
the logical host layout, not the physical register-pair representation.

The C pointer signature and manifest declare FP32 host output, while the
Atlas device result is BF16. Widening inserts the low 16 zero bits without
rounding. Both BF16 result bits and widened FP32 bits are compared. Existing
BF16 output callers retain their row-major copy behavior.

The model adapter reuses the vendored `model_instruction` helper and
`decode_binary`. Each model instruction must re-encode to the emitted word.
Expected outputs never enter the model subprocess. Program memory is
initialized through `InstantiableProgram.memory_regions`; results are read
from `state.read_dram` after successful halt and zero runtime errors. Inputs,
unused FP8 slot halves and leading/trailing guard regions must remain intact.

## Verified results

For the deterministic fixture, both runs halted with zero runtime errors in
3249 model cycles:

| A address | B address | C address | Exact matches | Max absolute error | Mismatches |
| --- | --- | --- | --- | --- | --- |
| `0x90000000` | `0x90000800` | `0x90001000` | 1024/1024 | 0 | 0 |
| `0x91000000` | `0x91000800` | `0x91001000` | 1024/1024 | 0 | 0 |

All guards and inputs were preserved. An initial readback that treated
physical BF16 bytes as a row-major tile gave 245 matches and 779 mismatches
(max error 22.75); applying the documented physical ABI conversion resolves
them without changing instructions or reference arithmetic.

For an independent infrastructure smoke check of the pre-existing Atlas
fixture (not a Triton kernel):

```sh
PYTHONPATH=/bwrcq/home/amy.yao/npu-model-eval \
  /bwrcq/home/amy.yao/atlas-eval-venv/bin/python test/numerical/smoke_existing_model.py
```
