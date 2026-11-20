# shellcheck shell=bash
# The start of every script here: the repo root, env.sh, the venv, and what to do when a step fails.

die() {
    trap - EXIT
    echo "error: $*" >&2
    exit 1
}

# A failure with a known fix calls die. This covers the rest, such as a compile error.
trap '[ $? -eq 0 ] || cat >&2 <<EOF
next: fix what the error above says, then rerun.
      If it is not clear, check the settings in env.sh, then run make clean clean.triton and make.
      After moving the repo, run make clean clean.triton clean.venv and make setup.
EOF' EXIT

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Only env.sh sets up the build. A shell's values, such as conda's, are for its own builds.
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LLVM_INSTALL_DIR
# Before strict mode, since env.sh is also written for an ordinary shell.
# shellcheck source=/dev/null
source ./env.sh || die "env.sh failed; fix it"
set -euo pipefail

# Compile with the clang from our LLVM, as Triton does, and link with its lld if it has one, unless env.sh names other
# compilers. Until setup has that LLVM, the default compiler is used. CMakeLists.txt does the same for the plugin.
if [ -n "${TRITON_ATLAS_CXX:-}" ]; then
    export CC="${TRITON_ATLAS_CC:-cc}" CXX="$TRITON_ATLAS_CXX"
elif [ -x "$LLVM_INSTALL_DIR/bin/clang++" ]; then
    export CC="$LLVM_INSTALL_DIR/bin/clang" CXX="$LLVM_INSTALL_DIR/bin/clang++"
    if [ -x "$LLVM_INSTALL_DIR/bin/ld.lld" ]; then
        export LDFLAGS="-fuse-ld=lld"
    fi
fi

command -v uv >/dev/null || die "uv not found; see https://docs.astral.sh/uv/"
# uv prefers VIRTUAL_ENV to ./.venv, so a venv active in the shell would otherwise receive our packages.
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Triton, cmake, ninja and the other tools in pyproject.toml, at the versions in uv.lock. --inexact keeps the plugin,
# which this step doesn't install and a plain uv sync would remove.
dev_tools() {
    uv sync -q --inexact --no-install-project
}

# The npu_model settings make test.numerical needs. Prints what is wrong and fails, for build.sh to stop and doctor to
# report.
npu_model_check() {
    if [ -z "${NPU_MODEL_ROOT:-}" ] || [ -z "${NPU_MODEL_PYTHON:-}" ]; then
        echo "NPU_MODEL_ROOT or NPU_MODEL_PYTHON not set; set both in env.sh, as env.example.sh shows"
        return 1
    fi
    PYTHONPATH="$NPU_MODEL_ROOT" "$NPU_MODEL_PYTHON" -c 'import torch, npu_model' 2>/dev/null || {
        echo "$NPU_MODEL_PYTHON can't import torch and npu_model from $NPU_MODEL_ROOT; fix NPU_MODEL_* in env.sh"
        return 1
    }
}

# Fetch only some paths of submodule $1, at our pin, into $2. llvm-project and triton are far larger than what we need
# from them, and Git records their pins without a checkout.
fetch_files() {
    local sub="$1" dest="$2" pin
    shift 2
    pin="$(git ls-tree HEAD "$sub" | awk '{print $3}')"
    [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" != "$pin" ] || return 0
    rm -rf "$dest"
    git init -q "$dest" &&
        git -C "$dest" sparse-checkout set --no-cone "$@" &&
        git -C "$dest" fetch -q --depth 1 --filter=blob:none "$(git config -f .gitmodules "submodule.$sub.url")" "$pin" &&
        git -C "$dest" checkout -q FETCH_HEAD
}
