#!/usr/bin/env bash
# The setup.* targets: what this repo depends on. One function per target, named after the part behind the dot.
# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

# llvm-project and triton are `update = none`, so they need an explicit --checkout.
fetch() {
    git submodule update --init --checkout --depth 1 "$@"
}

# To change Python, edit .python-version, then run make clean.venv and make setup.
venv() {
    [ -f .venv/pyvenv.cfg ] || uv venv --clear
}

# Record which LLVM env.sh uses. $1 can name $PWD, for env.sh to expand.
use_llvm() {
    sed -i '/LLVM_INSTALL_DIR/d' env.sh
    # shellcheck disable=SC2016 # the ${...} is for env.sh to expand
    printf 'export LLVM_INSTALL_DIR="%s"\n' "$1" >>env.sh
    LLVM_INSTALL_DIR="${1//\$PWD/$PWD}"
}

# A source build fills build/llvm as it goes, so it counts only once it has finished.
llvm_ready() {
    [ -d "$LLVM_INSTALL_DIR/lib/cmake/mlir" ] &&
        { [ "$LLVM_INSTALL_DIR" != "$PWD/build/llvm" ] || [ ! -f build/llvm/CMakeCache.txt ] || [ -f build/llvm/.done ]; }
}

# $1 is an existing install, and $2 asks for a source build. With neither, use the LLVM in env.sh, or download one.
llvm() {
    [ -z "${1:-}" ] || [ -z "${2:-}" ] || die "LLVM=<dir> and LLVM_SOURCE=1 ask for different LLVMs; pass one"
    if [ -n "${1:-}" ]; then
        [ -d "$1/lib/cmake/mlir" ] || die "no MLIR in $1"
        use_llvm "$(realpath "$1")"
    elif [ -n "${2:-}" ]; then
        llvm_source
    elif ! llvm_ready; then
        [ "$LLVM_INSTALL_DIR" = "$PWD/build/llvm" ] ||
            die "no MLIR in $LLVM_INSTALL_DIR; fix LLVM_INSTALL_DIR in env.sh or run make setup.llvm LLVM=<dir>"
        llvm_prebuilt
    fi
}

# Download Triton's own LLVM. The URL and sha256 come from Triton at our pin. Triton's script picks Ubuntu's build for
# any newer Linux, which needs a newer glibc than some have, so Linux takes the AlmaLinux 8 one: glibc 2.28.
llvm_prebuilt() {
    [ ! -f build/llvm/CMakeCache.txt ] || die "build/llvm holds a source build; run make setup LLVM_SOURCE=1, or \
make clean clean.triton clean.llvm first"
    # shellcheck disable=SC2016 # env.sh expands it
    use_llvm '$PWD/build/llvm'
    llvm_ready && return 0
    local os arch archive=build/llvm.tar.gz url sha
    case "$(uname -s)" in
        Linux) os=almalinux ;;
        Darwin) os=macos ;;
        *) die "Triton has no LLVM download for $(uname -s); run make setup LLVM_SOURCE=1" ;;
    esac
    case "$(uname -m)" in
        x86_64) arch=x64 ;;
        arm64 | aarch64) arch=arm64 ;;
        *) die "Triton has no LLVM download for $(uname -m); run make setup LLVM_SOURCE=1" ;;
    esac
    mkdir -p build
    fetch_files third_party/triton build/fetched/triton /cmake/llvm-info.json /python/build_helpers.py ||
        die "couldn't fetch Triton's LLVM details; check the network, or run make setup LLVM_SOURCE=1"
    read -r url sha < <(cd build/fetched/triton/python && python -c "
import build_helpers
archive = build_helpers.get_llvm_archive('$os-$arch')
print(archive.source_url, archive.sha256sum)")
    echo "Downloading LLVM: $url"
    curl -fL --retry 3 -o "$archive" "$url" || die "LLVM download failed; check the network, or run make setup LLVM_SOURCE=1"
    python -c 'import hashlib, sys; sys.exit(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != sys.argv[2])' \
        "$archive" "$sha" || die "$archive doesn't match Triton's sha256; delete it and rerun"
    rm -rf build/llvm build/llvm.tmp
    mkdir build/llvm.tmp
    tar -xzf "$archive" -C build/llvm.tmp --strip-components=1
    build/llvm.tmp/bin/llvm-config --version >/dev/null ||
        die "the downloaded LLVM doesn't run here; it needs glibc 2.27 or newer. Run make setup LLVM_SOURCE=1"
    mv build/llvm.tmp build/llvm
    rm "$archive"
}

# Reuses Triton's own script, unmodified (triton v3.9.0 scripts/build-llvm-project.sh, the one `make dev-install-llvm`
# runs): same projects (mlir;llvm;lld;clang), targets (Native;NVPTX;AMDGPU) and assertions as Triton's prebuilt LLVM. We
# only point it at our submodule and our build folder.
llvm_source() {
    [ ! -d build/llvm ] || [ -f build/llvm/CMakeCache.txt ] ||
        die "build/llvm holds a downloaded LLVM; run make clean clean.triton clean.llvm first"
    # shellcheck disable=SC2016 # env.sh expands it
    use_llvm '$PWD/build/llvm'
    llvm_ready && return 0
    echo "Building LLVM from the pinned submodule. This takes a while and several GB."

    dev_tools
    fetch third_party/llvm-project third_party/triton
    local pin want repo="$PWD"
    pin="$(git ls-tree HEAD third_party/llvm-project | awk '{print $3}')"
    want="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["llvm_hash"])' third_party/triton/cmake/llvm-info.json)"
    [ "$pin" = "$want" ] ||
        die "third_party/llvm-project is pinned at $pin, but Triton wants $want (cmake/llvm-info.json); update the pin"
    (
        cd third_party/triton
        LLVM_PROJECT_PATH="$repo/third_party/llvm-project" \
        LLVM_BUILD_PATH="$repo/build/llvm" \
        LLVM_COMMIT_HASH="$pin" \
        LLVM_BUILD_TYPE="${LLVM_BUILD_TYPE:-Release}" \
            scripts/build-llvm-project.sh
    )
    touch build/llvm/.done
}

# Report by tool, not by origin: an install doesn't record where it came from, and a build can leave tools out too.
llvm_check() {
    local bin="$LLVM_INSTALL_DIR/bin" tools hint=""
    [ -d "$LLVM_INSTALL_DIR/lib/cmake/mlir" ] || die "no MLIR in $LLVM_INSTALL_DIR; run make setup"
    tools=("$bin"/*)
    [ "${#tools[@]}" -gt 30 ] || hint=", looks like Triton's prebuilt"
    echo "LLVM $("$bin/llvm-config" --version) at $LLVM_INSTALL_DIR (${#tools[@]} tools$hint)"
    report_tool clang "compiling with the default compiler"
    report_tool ld.lld "linking with the system linker"
    report_tool mlir-opt "use the Python route for stock passes, or rebuild with LLVM_SOURCE=1"
}

report_tool() {
    [ -x "$LLVM_INSTALL_DIR/bin/$1" ] || echo "  missing  $1: $2"
}

# atlas-mlir's own build, as in its README, in its own folder. The plugin compiles only the library.
atlas_full() {
    dev_tools
    cmake -S third_party/atlas-mlir -B build/atlas -G Ninja \
        -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
    cmake --build build/atlas
}

# Compile the unpatched submodule through the tools/source wrapper. Kernels keep running on the wheel in the venv.
triton() {
    dev_tools
    fetch third_party/triton
    # Triton's own defaults for a tools build (triton v3.9.0 setup.py), without the Python module or unit tests.
    cmake -S tools/source -B build/triton -G Ninja \
        -DLLVM_SYSPATH="$LLVM_INSTALL_DIR" \
        -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
        -DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts \
        -DCMAKE_CXX_STANDARD=20 \
        -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
        -DTRITON_BUILD_UT=OFF \
        -DTRITON_BUILD_WITH_CCACHE=OFF \
        -DTRITON_CACHE_PATH="$PWD/build/triton-cache" \
        -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
    cmake --build build/triton
}

"$@"
