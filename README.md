# triton-atlas

A Triton backend for Atlas, the NPU in the Penguin SoC. It compiles `@triton.jit` kernels to Atlas virtual IR, the
input of [atlas-mlir](https://github.com/ucb-bar/atlas-mlir).

**Status:** the pipeline runs all seven passes. Pass 4 tiles statically sized linalg operations at 32, fuses
tile producers, fully unrolls static tile loops, recognizes supported compile-time edge masks, and tags matmuls
for MXU0. Pass 7 retains the initial static 32×32 two-layer FP8 MLP lowering; several other passes remain
placeholders. Launching a kernel still stops at a stub.

```text
@triton.jit kernel
  → ttir           Triton's passes (compiler.py, make_ttir)
  → atlas_virtual  our passes (the plugin library)
  → atlas-mlir     verify, schedule, emit (not wired in yet)
```

It is a plugin for the stock Triton 3.9.0 wheel; Triton isn't patched or rebuilt.

## Using the backend

Needs Linux x86-64 and Python 3.12+. Build the wheel with `make build.wheel` (in `build/dist/`) or get one from the team.

```bash
python -m venv .venv
.venv/bin/pip install triton_atlas_backend-0.1.0-py3-none-linux_x86_64.whl
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
export TRITON_ATLAS_TARGET=virtual
```

Use the CPU torch: PyPI's torch pins an older Triton, which breaks the plugin. Without `TRITON_ATLAS_TARGET`, Triton
uses the GPU backends.

```python
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def add_tile(x_ptr, y_ptr, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    tl.store(y_ptr + r, tl.load(x_ptr + r) + 1.0)


src = triton.compiler.ASTSource(fn=add_tile, signature={"x_ptr": "*bf16", "y_ptr": "*bf16", "BLOCK": "constexpr"},
                                constexprs={"BLOCK": 32})
kernel = triton.compile(src, target=GPUTarget("atlas", "virtual", 1))
print(kernel.asm["atlas_virtual"].decode())
```

To dump each stage: `TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=$PWD/dump TRITON_ALWAYS_COMPILE=1`.

## Developing the backend

Needs Linux x86-64, git and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo>
cd triton-atlas
make setup
source env.sh
```

`make setup` fetches Triton's prebuilt LLVM (2 GB) and builds everything. `env.sh` is per machine; see
`env.example.sh`.

```text
triton-atlas/
├── python/triton_atlas_backend/
│   ├── __init__.py                loads the plugin into Triton
│   ├── compiler.py                stages, passes, options
│   └── driver.py                  device hooks and the launcher (a stub)
├── plugin.cpp                     what the plugin registers
├── include/TritonToAtlasVirtual/  pass declarations (Passes.td)
├── lib/TritonToAtlasVirtual/      pass code; Pipeline.cpp sets the order
├── test/                          lit tests, one folder per pass
├── tools/                         triton-atlas-opt, language servers
├── third_party/                   atlas-mlir, triton, llvm-project (pinned)
├── scripts/                       what make runs
└── Makefile                       make help lists every command
```

| Task | Command |
|---|---|
| Rebuild after a C++ change | `make` (Python changes need nothing) |
| Rebuild only the plugin | `make build.plugin` |
| Run the tests | `make test` |
| Run some tests | `make test T=<file or folder>`; `LIT_FILTER=<regex>` picks by name |
| Run passes on one file | `triton-atlas-opt kernel.ttir --ttatlas-pipeline` (all of them) or `--<pass-name>` (one) |
| Run pass 4 tiling, mask checks, and MXU tagging | `triton-atlas-opt input.mlir --ttatlas-legalize-shapes` |
| Compile a kernel | `uv run python my_kernel.py` |
| Format and lint | `make format` (the git hooks run it on commit) |
| Build the wheel | `make build.wheel` |
| Build `triton-atlas-opt` | `make build.tools` |
| Language server for editors | `make build.lsp`; `.td` files: `make build.lsp.tblgen` |
| Also build atlas-mlir's own tools | `make setup.atlas.full` |
| Also build Triton (`triton-opt`, full server) | `make setup.triton` |
| Build LLVM from source (`mlir-opt`, all tools) | `make setup.llvm.source` (22 min, 9 GB) |
| Clean up | `make clean`; `make clean.<part>` for one part |
| List every command | `make help` |

To add a pass: declare it in its own file under `include/TritonToAtlasVirtual/Passes/` (included from `Passes.td`),
write it in its own folder under `lib/TritonToAtlasVirtual/`, add it to `Pipeline.cpp`, and add a test folder
`test/<pass>/` (see `test/plugin/verify-atlas-virtual.mlir`).

Atlas hardware facts (tile size, slot size, MXU limits, scale codes, DRAM window) live in
`include/TTAtlas/AtlasTarget.h`, each with its source; passes use those constants instead of spelling numbers. Logic
that depends on the hardware in a way a constant can't hold gets an `atlas-hw:` comment with its source.
`git grep -n "atlas-hw:\|target::"` lists every place the hardware reaches into the passes.

Pass 4 accepts static rank-1 and rank-2 linalg tensors whose dimensions and iteration extents are multiples of
32. Larger linalg operations are tiled with MLIR's linalg tiling interface; static tile loops are unrolled, so a
split-K matmul becomes an explicit chain. Tensor slices are fused into their linalg consumers where MLIR supports
the pattern. Every matmul is tagged `ttatlas.mxu = 0 : i32` (MXU0 is the initial policy).

For masked `tt.load` and `tt.store`, pass 4 recognizes `slt` comparisons of `tt.make_range` against a scalar
constant, including `andi` combinations and the usual `expand_dims`/`broadcast` wrappers. The load's explicit
other value or the stored value's inferred consumer neutral must be zero (add/matmul) or negative infinity
(max-reduction); the load neutral must agree with its consumers. The pass attaches `ttatlas.pad` metadata with
the logical shape. Dynamic shapes, non-multiple-of-32 linalg extents, unsupported mask expressions, and padding
whose neutral element cannot be established are rejected with an unsupported diagnostic. This first slice does
not yet turn arbitrary sub-32 or irregular linalg shapes into padded 32×32 compute tiles.

Focused Pass 4 lit coverage is in `test/legalize-shapes/`. Run it with
`make test T=test/legalize-shapes`; run the stage-contract fixtures separately with
`make test T=test/verify-contract`.

Versions: Triton exactly 3.9.0 and its LLVM commit (both checked at build time); atlas-mlir pinned on its default
branch.
