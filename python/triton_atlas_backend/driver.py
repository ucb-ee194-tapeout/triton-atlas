# Driver half of the Atlas backend for Triton (stub).
#
# Triton uses exactly one active driver (triton v3.9.0 python/triton/runtime/driver.py), so this one is active
# only when TRITON_ATLAS_TARGET is set; the NVIDIA and AMD drivers stay usable otherwise.
# TRITON_ATLAS_TARGET picks the output mode of IA-11 (what the backend delivers):
#   virtual  compile only, the deliverable is Atlas virtual IR (CompiledKernel.asm["atlas_virtual"])
#   npu      run on npu_model (phase 2 of the plan; not implemented yet)
# The hooks are DriverBase's (triton v3.9.0 python/triton/backends/driver.py:114-157).
import os
import warnings

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

TARGETS = ("virtual", "npu")


def _target():
    target = os.environ.get("TRITON_ATLAS_TARGET", "")
    if target and target not in TARGETS:
        raise ValueError(f"TRITON_ATLAS_TARGET={target!r}; expected one of {TARGETS}")
    return target


class AtlasUtils:

    @staticmethod
    def get_device_properties(device):
        return {"max_shared_mem": 0}

    @staticmethod
    def load_binary(name, kernel, shared, device):
        # (module, function, n_regs, n_spills, n_max_threads), as CompiledKernel expects
        return (None, name, 0, 0, 1)


class AtlasDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = AtlasUtils()

    @staticmethod
    def is_active():
        if _target():
            return True
        # Triton's error for no active driver doesn't name us, so say here what turns this one on. It imports us while
        # it fills `backends`, so that import has to wait until now.
        from triton.backends import backends
        if not any(backend.driver.is_active() for name, backend in backends.items() if name != "atlas"):
            warnings.warn(f"no Triton backend is active; to use Atlas, set TRITON_ATLAS_TARGET to one of {TARGETS}",
                          stacklevel=2)
        return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        raise NotImplementedError("the Atlas backend has no C launcher")

    def get_current_target(self):
        return GPUTarget("atlas", _target(), 1)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_benchmarker(self):
        raise NotImplementedError("benchmarking comes with npu_model's timed run (phase 5 of the plan)")

    def get_current_device(self):
        return 0

    def set_current_device(self, device):
        pass

    def get_current_stream(self, device=None):
        return 0

    def get_device_interface(self):
        import torch
        return torch.cpu

    @staticmethod
    def launcher_cls(src, metadata):

        def launch(gridX, gridY, gridZ, stream, function, *args):
            raise NotImplementedError(f"Atlas kernel {function!r}, grid ({gridX}, {gridY}, {gridZ}): "
                                      f"TRITON_ATLAS_TARGET={_target()} cannot launch yet "
                                      "(virtual mode compiles only; npu mode comes in phase 2)")

        return launch
