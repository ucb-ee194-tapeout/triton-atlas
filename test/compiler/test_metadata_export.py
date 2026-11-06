"""Tests for metadata transport, independently of unfinished compute passes."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from triton._C.libtriton import ir
from triton._C import libtriton

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = Path(shutil.which("triton-atlas-opt")).resolve().parent.parent / "libtriton_atlas_backend.so"
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
        # This tests transport, not the pending producer schema. Only the
        # already-defined version/name fields are interpreted by the exporter.
        manifest = '{ "version": 1, "name": "copy", "grid": [1,1,1], "legalized": [] }'
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
        manifest = '{"version":1,"name":"copy"}'

        class Module:
            context = object()
            converted = False

            def get_entry_func_name(self):
                return "copy"

            def walk(self, callback):
                if not self.converted:
                    raise AssertionError("export happened before conversion")
                callback(types.SimpleNamespace(
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
