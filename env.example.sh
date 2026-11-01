# shellcheck shell=bash
# Settings for this machine. `make setup` creates env.sh from this file; env.sh is not committed.
# Uncomment what you need. Run `source env.sh` from the repo root before running kernels.
# The build reads these settings only, not a shell's CC, CXX, CFLAGS, CXXFLAGS, CPPFLAGS or LDFLAGS, which conda and
# others export for their own builds.

# The LLVM to build against, which `make setup LLVM=<dir>` records for you. By default, setup downloads Triton's here.
export LLVM_INSTALL_DIR="$PWD/build/llvm"

# Compilers, if you don't want the clang from that LLVM. CMake keeps the first ones it sees, so run
# `make clean clean.triton` after changing them.
# export TRITON_ATLAS_CC=/path/to/cc
# export TRITON_ATLAS_CXX=/path/to/c++

# Tools from make setup.atlas.full and make setup.triton, such as atlas-opt and triton-opt.
export PATH="$PWD/build/bin:$PATH"

# Your shell's python stays yours, such as a conda environment's. Run the repo's with uv run python or .venv/bin/python,
# or uncomment this to make it the shell's python.
# export PATH="$PWD/.venv/bin:$PATH"

# What the Atlas driver does: virtual (compile only) or npu (run on npu_model; not implemented yet).
export TRITON_ATLAS_TARGET="${TRITON_ATLAS_TARGET:-virtual}"
