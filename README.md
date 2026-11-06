# triton-atlas

A Triton backend for Atlas, the NPU in the Penguin SoC. It compiles `@triton.jit` kernels to Atlas virtual IR, the
input of [atlas-mlir](https://github.com/ucb-bar/atlas-mlir).

**Status:** the native pass has an initial lowering for one static 32x32 two-layer FP8 MLP. Other kernels are not yet
lowered, and launching an Atlas kernel still stops at a stub.

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
| Run passes on one file | `triton-atlas-opt kernel.ttir --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream` |
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

To add a pass: declare it in `Passes.td`, write it in `lib/TritonToAtlasVirtual/`, add it to `Pipeline.cpp`, and add a
test in `test/<pass>/` (see `test/plugin/verify-atlas-virtual.mlir`).

Versions: Triton exactly 3.9.0 and its LLVM commit (both checked at build time); atlas-mlir pinned on its default
branch.
