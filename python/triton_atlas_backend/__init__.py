# Loads our plugin into Triton. Importing this package is enough: Triton imports it through our `triton.backends`
# entry point, and the plugin resolves MLIR and LLVM from the libtriton already in memory.
# Follows triton-lang/triton-ext@268252b backend/AppleGPU/python/triton_apple_backend/__init__.py (MIT): an editable
# install serves this file from the source tree but puts the library in site-packages, so both are searched.
import sysconfig
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PLUGIN_NAME = "libtriton_atlas_backend.so"
PLUGIN_LIBRARY = next(
    (p for p in (PACKAGE_DIR / PLUGIN_NAME, Path(sysconfig.get_paths()["purelib"]) / PACKAGE_DIR.name / PLUGIN_NAME)
     if p.exists()), None)
if PLUGIN_LIBRARY is None:
    raise ImportError(f"{PLUGIN_NAME} not found; build it with scripts/build.sh")

# Importing Triton imports compiler.py through the entry point, and compiler.py reads the names above, so Triton has
# to come after them.
import triton._C.libtriton as libtriton  # noqa: E402

libtriton.passes.plugin.extend_with(str(PLUGIN_LIBRARY))
libtriton.ir.extend_dialects_with(str(PLUGIN_LIBRARY))
