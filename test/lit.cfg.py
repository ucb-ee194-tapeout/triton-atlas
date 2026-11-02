# Runs the .mlir tests under test/: each file's RUN lines, checked by FileCheck
# (https://llvm.org/docs/CommandGuide/lit.html). The tools come from this build: build/bin (triton-atlas-opt) and the
# LLVM in env.sh (FileCheck, not). make test runs it; T=<file or folder> picks tests.
# lit defines config before running this file.
# ruff: noqa: F821
import os

import lit.formats

config.name = "triton-atlas"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
_repo = os.path.dirname(config.test_source_root)
config.test_exec_root = os.path.join(_repo, "build", "test")
config.environment["PATH"] = os.pathsep.join([
    os.path.join(_repo, "build", "bin"),
    os.path.join(os.environ["LLVM_INSTALL_DIR"], "bin"),
    os.environ["PATH"],
])
