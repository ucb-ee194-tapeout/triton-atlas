#!/usr/bin/env bash
# The build.* targets: this repo's code. One function per target, named after the part behind the dot.
# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

# Install the tools and the pinned Triton, then our package. The second step configures build/plugin and builds it.
# The build reads Triton from the venv (pyproject.toml's [tool.uv]), so the first step has to come first. --inexact keeps
# what isn't in uv.lock, such as tools installed by hand for an editor.
package() {
    dev_tools
    uv sync --inexact --reinstall-package triton-atlas-backend
}

# Rebuild with Ninja and copy the plugin into the venv. The Python stays in the source tree, so it needs no copy.
plugin() {
    cmake --build build/plugin
    # A failed install can leave the package uninstalled. This lookup fails then, and Ninja wouldn't notice.
    local site
    site="$(python -c 'import importlib.metadata as m; print(m.distribution("triton-atlas-backend").locate_file(""))' 2>/dev/null)" ||
        die "triton-atlas-backend isn't installed; run make clean, then make"
    cmake --install build/plugin --component plugin --prefix "$site"
}

# The file to hand out. It compiles in build/plugin, like the rest.
wheel() {
    uv build --wheel --out-dir build/dist
}

# triton-opt appears only after a finished Triton build. build/triton-atlas-lsp --version reports which server it is.
lsp() {
    if [ -x build/triton/triton/bin/triton-opt ]; then lsp_full; else lsp_small; fi
    lsp_tblgen
}

lsp_small() {
    link_lsp build/plugin
}

lsp_full() {
    link_lsp build/triton
}

# The server for .td files, which not every LLVM has. Most LLVMs lack its library too, so compile its sources at our pin.
lsp_tblgen() {
    if ! fetch_files third_party/llvm-project build/fetched/llvm-project \
        /mlir/lib/Tools/tblgen-lsp-server/ /mlir/tools/tblgen-lsp-server/; then
        echo "warning: couldn't fetch the .td language server's sources; skipping it" >&2
        return 0
    fi
    [ -f build/tblgen-lsp/build.ninja ] || cmake -S tools/tblgen-lsp -B build/tblgen-lsp -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
        -DLLVM_PROJECT_DIR="$PWD/build/fetched/llvm-project"
    cmake --build build/tblgen-lsp
    ln -sf "$PWD/build/tblgen-lsp/bin/triton-atlas-tblgen-lsp" build/tblgen-lsp-server
}

# Build triton-atlas-opt, keep the full atlas-mlir build current, and link every tool that exists into build/bin, which
# env.sh puts on PATH.
tools() {
    [ ! -f build/atlas/build.ninja ] || cmake --build build/atlas
    cmake --build build/plugin --target triton-atlas-opt
    rm -rf build/bin
    mkdir -p build/bin
    local tool
    for tool in build/plugin/bin/triton-atlas-opt build/atlas/bin/* build/triton/triton/bin/*; do
        case "$tool" in *-test) continue ;; esac
        if [ -f "$tool" ] && [ -x "$tool" ]; then ln -sf "$PWD/$tool" build/bin/; fi
    done
}

# Build the server in folder $1 and link it where editors look.
link_lsp() {
    cmake --build "$1" --target triton-atlas-lsp
    ln -sf "$PWD/$1/bin/triton-atlas-lsp" build/triton-atlas-lsp
}

"$@"
