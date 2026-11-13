"""Tests for metadata transport, independently of unfinished compute passes."""
import importlib.util
import importlib.metadata
import json
import copy
import subprocess
import shutil
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from jsonschema import ValidationError

from triton._C.libtriton import ir
from triton._C import libtriton

ROOT = Path(__file__).resolve().parents[2]
# Use the installed library also loaded by Triton's backend discovery. Loading
# a second build copy registers different C++ dialect identities for ttatlas.
PLUGIN = Path(
    importlib.metadata.distribution("triton-atlas-backend").locate_file(
        "triton_atlas_backend/libtriton_atlas_backend.so"))
# Import only this lane's Python module without requiring package installation.
package = types.ModuleType("triton_atlas_backend")
package.PACKAGE_DIR = ROOT / "python/triton_atlas_backend"
package.PLUGIN_LIBRARY = PLUGIN
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location("triton_atlas_backend.compiler", package.PACKAGE_DIR / "compiler.py")
compiler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compiler
spec.loader.exec_module(compiler)

libtriton.passes.plugin.extend_with(str(PLUGIN))
libtriton.ir.extend_dialects_with(str(PLUGIN))


class MetadataExportTest(unittest.TestCase):

    def manifest(self):
        golden = json.loads((ROOT / "docs/examples/ttatlas.kernel.v1.json").read_text())
        golden["name"] = "copy"
        return json.dumps(golden)

    def test_schema_and_semantic_rejections(self):
        original = json.loads(self.manifest())
        for mutate in (
                lambda m: m["inputs"][0].pop("host_dtype"),
                lambda m: m["inputs"][0].update(index=2),
                lambda m: m["inputs"][0].update(tile_origin="unknown"),
                lambda m: m.update(grid=[2, 1, 1]),
                lambda m: m["dram_base"].update(output=0x90002000),
                lambda m: m["outputs"][0].update(role="bias_bf16"),
        ):
            manifest = copy.deepcopy(original)
            mutate(manifest)
            with self.subTest(manifest=manifest), self.assertRaises((ValueError, ValidationError)):
                compiler._validate_kernel_metadata(json.dumps(manifest), "copy")

    def test_native_producer_and_determinism(self):
        command = [
            shutil.which("triton-atlas-opt"),
            str(ROOT / "test/inputs/boundary-slices.ttir.mlir"), "--ttatlas-assign-slots",
            "--convert-triton-to-atlas-virtual", "--ttatlas-verify-contract=stage=7", "--verify-atlas-virtual-stream"
        ]
        first = subprocess.check_output(command, text=True)
        self.assertEqual(first, subprocess.check_output(command, text=True))
        context, module = self.parse(first)
        metadata = {}
        compiler._export_kernel_metadata(module, metadata)
        manifest = json.loads(metadata["ttatlas_kernel"])
        self.assertEqual(
            [entry["tile_origin"] for entry in manifest["inputs"]],
            ["affine_map<()[s0, s1, s2, s3, s4, s5] -> (1, 0)>", "affine_map<()[s0, s1, s2, s3, s4, s5] -> (1, 32)>"])
        self.assertEqual([entry["host_dtype"] for entry in manifest["outputs"]], ["f32", "f32"])
        self.assertEqual(manifest["dram_base"]["output"], 0x90001000)
        self.assertEqual(manifest["legalized"][0]["what"], "f32 output to bf16")

    def test_native_options_handoff(self):
        # Exercise the Python entry point on direct TTIR tiles. Mixed Tensor/
        # Linalg dialect registration in the Python plugin belongs upstream.
        source = (ROOT / "test/assign-slots/bf16.mlir").read_text().replace(
            "tt.return", "tt.store %p, %x {ttatlas.host_dtype = f32} : tensor<32x32x!tt.ptr<bf16>>\n    tt.return")
        context, module = self.parse(source)
        module.context = context
        metadata = {"num_warps": 1}
        compiler.AtlasBackend.make_atlas_virtual(module, metadata,
                                                 compiler.AtlasOptions(layout="tile_major", dram_base=0x90002000))
        manifest = json.loads(metadata["ttatlas_kernel"])
        self.assertEqual(manifest["dram_base"], {"input": 0x90002000, "output": 0x90002800})
        self.assertEqual({entry["layout"] for entry in manifest["inputs"] + manifest["outputs"]}, {"tile_major"})
        self.assertEqual(metadata["num_warps"], 1)

    def test_options_validation_and_cache(self):
        defaults = compiler.AtlasOptions()
        self.assertNotEqual(defaults.hash(), compiler.AtlasOptions(layout="tile_major").hash())
        self.assertNotEqual(defaults.hash(), compiler.AtlasOptions(dram_base=0x90002000).hash())
        for kwargs in ({"dram_base": -1}, {"dram_base": 0x90000001}, {"layout":
                                                                      "unknown"}, {"precision_policy": "unknown"}):
            with self.assertRaises(ValueError):
                compiler.AtlasOptions(**kwargs)

    def parse(self, text):
        context = ir.context()
        ir.load_dialects(context)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir") as source:
            source.write(text)
            source.flush()
            module = ir.parse_mlir_module(source.name, context)
        # Keep the native context alive for the module's lifetime.
        return context, module

    def function(self, manifest=None, name="copy"):
        attr = ""
        if manifest is not None:
            escaped = manifest.replace("\\", "\\\\").replace('"', '\\22')
            attr = f' attributes {{ttatlas.kernel = "{escaped}"}}'
        return f"func.func @{name}(){attr} {{return}}"

    def test_verbatim_native_attribute_and_cache_roundtrip(self):
        manifest = self.manifest()
        context, module = self.parse(self.function(manifest))
        metadata = {"name": "old", "shared": 8, "num_warps": 1}
        compiler._export_kernel_metadata(module, metadata)
        self.assertEqual(metadata["ttatlas_kernel"], manifest)
        self.assertEqual(metadata["name"], "copy")
        self.assertEqual(metadata["shared"], 0)
        cached = json.loads(json.dumps(metadata))
        self.assertEqual(cached, metadata)
        from collections import namedtuple
        self.assertEqual(namedtuple("KernelMetadata", cached)(**cached).ttatlas_kernel, manifest)

    def test_missing_manifest(self):
        context, module = self.parse(self.function())
        with self.assertRaisesRegex(ValueError, "missing finalized"):
            compiler._export_kernel_metadata(module, {})

    def test_mismatched_name(self):
        context, module = self.parse(self.function('{"version":1,"name":"other"}'))
        with self.assertRaisesRegex(ValueError, "matching the Atlas function name"):
            compiler._export_kernel_metadata(module, {})

    def test_multiple_functions(self):
        context, module = self.parse(self.function(name="a") + self.function(name="b"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            compiler._export_kernel_metadata(module, {})

    def test_stage_exports_after_pipeline(self):
        manifest = self.manifest()

        class Module:
            context = object()
            converted = False

            def get_entry_func_name(self):
                return "copy"

            def walk(self, callback):
                if not self.converted:
                    raise AssertionError("export happened before conversion")
                callback(
                    types.SimpleNamespace(
                        get_name=lambda: "func.func",
                        get_str_attr=lambda key: {"sym_name": "copy", "ttatlas.kernel": manifest}.get(key)))

            def __str__(self):
                return "converted"

        class Manager:

            def enable_debug(self):
                pass

            def run(self, mod, stage):
                self.stage = stage
                mod.converted = True

        manager = Manager()
        module = Module()
        metadata = {}
        with patch.object(compiler, "ir", types.SimpleNamespace(pass_manager=lambda ctx: manager)), \
             patch.object(compiler, "passes", types.SimpleNamespace(
                 plugin=types.SimpleNamespace(add_atlas_virtual_pipeline=lambda pm: None))):
            self.assertEqual(compiler.AtlasBackend.make_atlas_virtual(module, metadata, None), "converted")
        self.assertEqual(metadata["ttatlas_kernel"], manifest)
        self.assertEqual(manager.stage, "make_atlas_virtual")


if __name__ == "__main__":
    unittest.main()
