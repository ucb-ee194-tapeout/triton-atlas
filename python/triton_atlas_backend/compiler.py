# Compiler half of the Atlas backend for Triton (stub).
#
# Triton finds this class through our `triton.backends` entry point (pyproject.toml; triton v3.9.0
# python/triton/backends/__init__.py:57-62). The hooks are BaseBackend's (python/triton/backends/compiler.py:23-75).
# make_ttir follows Triton's two backends, NVIDIA and AMD, pass for pass (see the comment there). A line marked
# "Atlas:" is where we differ from them; every other line is theirs.
#
# Stub for now: the "atlas_virtual" stage only tags the TTIR text. Our TTIR -> Atlas virtual ops passes replace it
# in phase 1 of the plan; run mode (lowering to program.bin) comes in phase 2.
import dataclasses
import hashlib

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language


@dataclasses.dataclass(frozen=True)
class AtlasOptions:
    # Fields Triton's frontend and CompiledKernel read from every backend's options.
    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 1
    cluster_dims: tuple = (1, 1, 1)
    debug: bool = False
    sanitize_overflow: bool = False
    arch: str = "virtual"
    supported_fp8_dtypes: tuple = ("fp8e4nv", )
    deprecated_fp8_dot_operand_dtypes: tuple = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee", )
    max_num_imprecise_acc_default: int = 0
    extern_libs: tuple = ()
    backend_name: str = "atlas"
    instrumentation_mode: str = ""
    fpsan_homomorphic_casts: bool = False  # JITFunction.run always passes it (triton v3.9.0 runtime/jit.py:734-736)
    enable_fp_fusion: bool = False
    launch_cooperative_grid: bool = False  # Triton's compile hooks read it (triton v3.9.0 runtime/jit.py:658, 672)

    def hash(self):
        return hashlib.sha256(str(self).encode()).hexdigest()


class AtlasBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "atlas"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "atlas_virtual"

    def hash(self) -> str:
        return f"atlas-{self.target.arch}-stub-0"

    def parse_options(self, options: dict) -> object:
        args = {k: v for k, v in options.items() if k in AtlasOptions.__dataclass_fields__}
        args["arch"] = self.target.arch
        return AtlasOptions(**args)

    def pack_metadata(self, metadata):
        return ()

    def get_codegen_implementation(self, options):
        # tl.dot asks the backend for its smallest (M, N, K) (triton v3.9.0 python/triton/language/semantic.py:1596-1601;
        # NVIDIA's version: third_party/nvidia/backend/compiler.py:22-34). Atlas's MXU works on whole 32x32 tiles,
        # so ask for 32 in every dimension; a smaller dot fails in Triton's frontend with Triton's own message.
        return {"min_dot_size": lambda lhs_type, rhs_type: (32, 32, 32)}

    def get_module_map(self):
        return {}

    def load_dialects(self, context):
        from triton._C.libtriton import atlas
        atlas.load_dialects(context)

    # Upstream Triton has two backends, and their TTIR lists agree in order (triton v3.9.0): NVIDIA
    # third_party/nvidia/backend/compiler.py:285-298, AMD third_party/amd/backend/compiler.py:443-457.
    # A pass only one of them runs says so. Summaries are from include/triton/Dialect/Triton/Transforms/Passes.td
    # for the ttir passes and from MLIR for the common ones.
    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # Inline calls to other @triton.jit functions.
        passes.common.add_inliner(pm)
        # Rewrite tensor-descriptor loads and stores into pointer loads and stores. Upstream runs it only on hardware
        # without descriptor support: NVIDIA below capability 9 (no TMA), AMD without TDM.
        # Atlas: always, since Atlas has neither.
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        # MLIR's folding and simplification patterns.
        passes.common.add_canonicalizer(pm)
        # Triton's peephole rewrites, e.g. dot(a, b, 0) + c => dot(a, b, c), and nested addptr into one.
        passes.ttir.add_combine(pm)
        # Move broadcast and splat after elementwise ops, so those run on the smaller tensor.
        passes.ttir.add_reorder_broadcast(pm)
        # Common subexpression elimination.
        passes.common.add_cse(pm)
        # AMD only: MLIR's loop-invariant code motion, plus hoisting masked loads out of loops.
        # Atlas: off for now; whether to run it is open.
        # passes.ttir.add_triton_licm(pm)
        # Delete functions nothing calls any more, such as inlined helpers.
        passes.common.add_symbol_dce(pm)
        # Unroll loops that carry tt.loop_unroll_factor, set by tl.range(loop_unroll_factor=...).
        passes.ttir.add_loop_unroll(pm)
        # Not in either make_ttir. These run on TTIR's ops but upstream runs them later, in make_ttgir, which we
        # don't have, so their cleanup is lost unless we add them here. Atlas: off; whether to add them is open.
        # Constant propagation. NVIDIA make_ttgir (compiler.py:375).
        # passes.common.add_sccp(pm)
        # CSE that also removes loop-carried values that never change. NVIDIA make_ttgir (compiler.py:325, 355, 370).
        # passes.ttir.add_loop_aware_cse(pm)
        # Loop-invariant code motion, as above. NVIDIA make_ttgir (compiler.py:329, 339, 353), AMD make_ttgir (482).
        # passes.ttir.add_triton_licm(pm)
        # Not listed: passes.ttir.add_convert_to_ttgpuir is the step into TTGIR itself, and the ttgpuir, convert,
        # llvmir and gluon groups work on GPU layouts, LLVM or Gluon, none of which this stage has.
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_atlas_virtual(mod, metadata, options):
        # Stand-in for our TTIR -> Atlas virtual ops passes: keeps the TTIR text, tagged.
        text = str(mod)
        metadata["name"] = next(line for line in text.splitlines() if "tt.func" in line).split("@")[1].split("(")[0]
        metadata["shared"] = 0
        return "// atlas stub: TTIR, not yet Atlas virtual ops\n" + text

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError("the Atlas backend compiles Triton kernels only")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["atlas_virtual"] = lambda src, metadata: self.make_atlas_virtual(src, metadata, options)
