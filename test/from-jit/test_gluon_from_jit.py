# RUN: %PYTHON %s
#
# End-to-end Gluon front-door test.  The kernel intentionally uses AutoLayout
# followed by set_auto_layout so that the test exercises both Atlas-only
# frontend passes before it enters the ordinary TTIR pipeline.
import os
import sys
import tempfile

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="ttatlas-gluon-")

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as ttgl  # noqa: E402


@gluon.jit
def copy_16(X, Y, BLOCK: ttgl.constexpr, LAYOUT: ttgl.constexpr):
    offsets = ttgl.arange(0, BLOCK)
    ptrs = X + offsets
    value = ttgl.load(ttgl.set_auto_layout(ptrs, LAYOUT))
    ttgl.store(ttgl.set_auto_layout(Y + offsets, LAYOUT), value)


source = gluon.GluonASTSource(
    copy_16,
    signature={"X": "*bf16", "Y": "*bf16", "BLOCK": "constexpr", "LAYOUT": "constexpr"},
    constexprs={"BLOCK": 16, "LAYOUT": ttgl.BlockedLayout([16], [1], [1], [0])},
)
compiled = triton.compile(source, target=GPUTarget("atlas", "virtual", 1))
assert b"ttatlas.kernel" in compiled.asm["atlas_virtual"]
