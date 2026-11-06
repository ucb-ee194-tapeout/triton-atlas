# Compiler half of the Atlas backend for Triton.
#
# Triton finds this class through our `triton.backends` entry point (pyproject.toml; triton v3.9.0
# python/triton/backends/__init__.py:57-62). The hooks are BaseBackend's (python/triton/backends/compiler.py:23-75).
# make_ttir follows Triton's two backends, NVIDIA and AMD, pass for pass (see the comment there). A line marked
# "Atlas:" is where we differ from them; every other line is theirs.
#
# The "atlas_virtual" stage runs our C++ pipeline (lib/TritonToAtlasVirtual/Pipeline.cpp) and exports its
# validated kernel manifest. Boundary conversion is implemented; upstream compute/legalization passes are pending.
# Run mode (lowering to program.bin) comes in phase 2.
import dataclasses
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language

from . import PACKAGE_DIR, PLUGIN_LIBRARY


@lru_cache(maxsize=1)
def _kernel_validator():
    schema = json.loads((Path(__file__).parent / "schemas/ttatlas.kernel.v1.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_kernel_metadata(manifest, name):
    parsed = json.loads(manifest)
    if not isinstance(parsed, dict) or parsed.get("version") != 1 or parsed.get("name") != name:
        raise ValueError("ttatlas.kernel must be a version-1 manifest matching the Atlas function name")
    _kernel_validator().validate(parsed)
    for table in (parsed["inputs"], parsed["outputs"]):
        if [entry["index"] for entry in table] != list(range(len(table))):
            raise ValueError("ttatlas.kernel slot indices must be dense and sorted")
    bases = parsed["dram_base"]
    if bases["output"] != bases["input"] + len(parsed["inputs"]) * 2048:
        raise ValueError("ttatlas.kernel output base must follow the input slot region")
    if bases["output"] + len(parsed["outputs"]) * 2048 > 1 << 32:
        raise ValueError("ttatlas.kernel slot windows exceed 32-bit DRAM")
    return parsed


def _export_kernel_metadata(mod, metadata):
    """Copy the finalized MLIR manifest verbatim; the driver owns its use."""
    functions = []

    def collect(op):
        if op.get_name() == "func.func":
            functions.append((op.get_str_attr("sym_name"), op.get_str_attr("ttatlas.kernel")))

    mod.walk(collect)
    if len(functions) != 1:
        raise ValueError("Atlas metadata export requires exactly one converted func.func")
    name, manifest = functions[0]
    if manifest is None:
        raise ValueError("Converted Atlas function is missing finalized ttatlas.kernel metadata")
    _validate_kernel_metadata(manifest, name)
    # Triton constructs a namedtuple from metadata keys, so a dotted key is
    # invalid. Keep the original JSON string under an identifier-safe key.
    metadata["ttatlas_kernel"] = manifest
    metadata["name"] = name
    metadata["shared"] = 0


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

    # Backend placement choices participate in Triton's option/cache hash.
    precision_policy: str = "fp8-bf16-v1"
    layout: str = "row_major"
    dram_base: int = 0x90000000

    def __post_init__(self):
        if self.precision_policy != "fp8-bf16-v1":
            raise ValueError("unsupported Atlas precision policy")
        if self.layout not in ("row_major", "tile_major"):
            raise ValueError("unsupported Atlas layout")
        if type(self.dram_base) is not int or not 0x80000000 <= self.dram_base < 1 << 32 or self.dram_base % 1024:
            raise ValueError("Atlas DRAM base must be an aligned address in selected 32-bit DRAM")

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
        # Triton's cache key covers only Triton's own package (triton v3.9.0 python/triton/runtime/cache.py:284-315),
        # so without this an edited .py or a rebuilt plugin silently reuses the old cached kernel.
        digest = hashlib.sha256()
        for path in [
                *sorted(PACKAGE_DIR.glob("*.py")), *sorted((PACKAGE_DIR / "schemas").glob("*.json")), PLUGIN_LIBRARY
        ]:
            digest.update(path.read_bytes())
        return f"atlas-{self.target.arch}-{digest.hexdigest()}"

    def parse_options(self, options: dict) -> object:
        args = {k: v for k, v in options.items() if k in AtlasOptions.__dataclass_fields__}
        args["arch"] = self.target.arch
        return AtlasOptions(**args)

    def pack_metadata(self, metadata):
        return ()

    def get_codegen_implementation(self, options):
        # atlas-hw: the 32x32 tile (include/TTAtlas/AtlasTarget.h, kTileSize; atlas-mlir docs/compiler-pipeline.md:109).
        # tl.dot asks the backend for its smallest (M, N, K) (triton v3.9.0 python/triton/language/semantic.py:1596-1601;
        # NVIDIA's version: third_party/nvidia/backend/compiler.py:22-34). Atlas's MXU works on whole 32x32 tiles,
        # so ask for 32 in every dimension; a smaller dot fails in Triton's frontend with Triton's own message.
        return {"min_dot_size": lambda lhs_type, rhs_type: (32, 32, 32)}

    def get_module_map(self):
        return {}

    def load_dialects(self, context):
        # Nothing to do: Triton loads plugin dialects itself (extend_dialects_with in __init__.py;
        # triton v3.9.0 python/src/ir.cc:360-366).
        pass

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
        metadata["name"] = mod.get_entry_func_name()
        metadata["shared"] = 0
        if options is not None:
            builder = ir.builder(mod.context)
            mod.set_attr("ttatlas.precision_policy", builder.get_string_attr(options.precision_policy))
            # Triton's Python IR bindings cannot build i64 or custom enum
            # attributes. This private handoff is normalized by our metadata
            # producer; shared typed encodings remain unchanged.
            mod.set_attr(
                "atlas_backend.options",
                builder.get_string_attr(
                    json.dumps({"dram_base": options.dram_base, "layout": options.layout}, sort_keys=True)))
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.add_atlas_virtual_pipeline(pm)
        pm.run(mod, "make_atlas_virtual")
        _export_kernel_metadata(mod, metadata)
        return str(mod)

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError("the Atlas backend compiles Triton kernels only")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["atlas_virtual"] = lambda src, metadata: self.make_atlas_virtual(src, metadata, options)
