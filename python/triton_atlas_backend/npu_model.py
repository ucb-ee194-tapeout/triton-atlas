"""Execution-backed Atlas timing through the pinned UCB EE194 npu_model.

This module is deliberately separate from the driver: it accepts the emitted
program JSON, verifies the model can re-encode every instruction bit-for-bit,
then returns cycles from the execution-driven simulator.  A heuristic must not
pretend to be an autotuning measurement.
"""
import dataclasses
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_REVISION = "337e9f249f938a1dff007ddf8a763e5cd4b3aa2b"


def _tool(path, name):
    found = shutil.which(str(path)) if not os.path.sep in str(path) else path
    value = Path(found or path).expanduser().resolve()
    if not value.is_file() or not os.access(value, os.X_OK):
        raise RuntimeError(f"{name} is not executable: {value}")
    return value


def emit(virtual_ir, *, triton_atlas_opt=None, atlas_bin=None):
    """Lower one Atlas virtual module to the emitted program JSON.

    This is the exact production handoff: no interpreter or synthetic cost
    model is substituted between Triton's virtual stream and npu_model.
    ``ATLAS_OOT_BIN_DIR`` names atlas-opt/atlas-emit and
    ``TRITON_ATLAS_OPT`` names the plugin cleanup tool when arguments are
    omitted.
    """
    atlas_bin = Path(atlas_bin or os.environ.get("ATLAS_OOT_BIN_DIR", "")).expanduser()
    atlas_opt = _tool(atlas_bin / "atlas-opt", "atlas-opt")
    atlas_emit = _tool(atlas_bin / "atlas-emit", "atlas-emit")
    triton_opt = _tool(triton_atlas_opt or os.environ.get("TRITON_ATLAS_OPT", "triton-atlas-opt"),
                       "triton-atlas-opt")
    if isinstance(virtual_ir, bytes):
        virtual_ir = virtual_ir.decode()
    with tempfile.TemporaryDirectory(prefix="ttatlas-emit-") as directory:
        directory = Path(directory)
        virtual = directory / "kernel.virtual.mlir"
        machine = directory / "kernel.machine.mlir"
        emitted = directory / "kernel.emit.mlir"
        virtual.write_text(virtual_ir)
        commands = [
            [atlas_opt, virtual, "--allow-unregistered-dialect", "--atlas-compile", "-o", machine],
            [triton_opt, machine, "--ttatlas-finalize-machine-handoff", "-o", emitted],
            [atlas_emit, "--program-json", emitted],
        ]
        output = None
        for command in commands:
            completed = subprocess.run(list(map(str, command)), capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"{' '.join(map(str, command))}\n{completed.stderr.strip()}")
            output = completed.stdout
        return json.loads(output)


def _require_root(value, variable):
    if not value:
        raise RuntimeError(f"{variable} must name the matching source checkout")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"{variable}={root} is not a directory")
    return root


def _instruction(row, inventory, isa):
    """Create and independently verify one npu_model instruction."""
    from npu_model.isa import SBType, SType, UJType, UType, VIType

    variant = next(item for item in inventory["variants"] if item["id"] == row["mode_id"])
    name = row["mode_id"].removesuffix("_ANY") if row["mode_id"].startswith("DMA_") else variant["model_classes"][0]
    cls, word = getattr(isa, name), row["word_u32"]
    operands = {
        "rd": (word >> 7) & 31, "rs1": (word >> 15) & 31, "rs2": (word >> 20) & 31,
        "vd": (word >> 7) & 63, "vs1": (word >> 13) & 63, "vs2": (word >> 19) & 63,
        "ch": (word >> 12) & 7,
    }
    if issubclass(cls, SBType):
        operands["imm"] = row["fields"]["offset_bytes"]
    elif issubclass(cls, UJType):
        operands["imm"] = row["fields"]["offset"]
    elif issubclass(cls, SType):
        operands["imm"] = ((word >> 25) << 5) | ((word >> 7) & 31)
    elif issubclass(cls, VIType):
        operands["imm"] = word >> 16
    elif issubclass(cls, UType):
        operands["imm"] = word >> 12
    else:
        operands["imm"] = word >> 20
    instruction = cls(**{key: operands[key] for key in inspect.signature(cls).parameters})
    if instruction.to_bytecode() != word:
        raise RuntimeError(f"npu_model re-encoding differs at emitted word {row['word_index']}")
    return instruction


def run(program_json, initial, dram_base, *, max_cycles=500_000):
    """Execute an emitted Atlas program and return ``(cycles, dram_bytes)``.

    ``ATLAS_MODEL_ROOT`` must point at the documented model revision and
    ``ATLAS_MLIR_ROOT`` at the matching atlas-mlir checkout.  The caller owns
    boundary packing and output unpacking; this layer owns instruction fidelity
    and timing.
    """
    model_root = _require_root(os.environ.get("ATLAS_MODEL_ROOT", ""), "ATLAS_MODEL_ROOT")
    atlas_root = _require_root(os.environ.get("ATLAS_MLIR_ROOT", ""), "ATLAS_MLIR_ROOT")
    revision = subprocess.run(["git", "-C", str(model_root), "rev-parse", "HEAD"], capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != MODEL_REVISION:
        raise RuntimeError(f"ATLAS_MODEL_ROOT must be npu_model {MODEL_REVISION}")

    sys.path.insert(0, str(model_root))
    sys.path.insert(0, str(atlas_root))
    try:
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary

        emitted = json.loads(program_json) if isinstance(program_json, str) else program_json
        inventory = json.loads((atlas_root / "docs/selected-variant-inventory.json").read_text())
        binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in emitted["instructions"])
        decoded = decode_binary(inventory, binary)
        program = InstantiableProgram([_instruction(row, inventory, isa) for row in decoded["instructions"]])
        program.dram_base = dram_base
        program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
        config = DefaultHardwareConfig()
        config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics="rtl", randomize_init=True,
                                                       init_seed=187)
        with tempfile.TemporaryDirectory(prefix="ttatlas-autotune-") as directory:
            simulation = Simulation(config, LoggerConfig(filename=str(Path(directory) / "trace.json")), program,
                                    verbose=False)
            state = simulation.core.arch_state
            try:
                simulation.run(max_cycles=max_cycles)
                stats = simulation.get_stats()
                if not state.halted or stats.runtime_errors:
                    raise RuntimeError(f"npu_model did not complete cleanly: halted={state.halted}, errors={stats.runtime_errors}")
                return stats.cycles, bytes(state.read_dram(dram_base, len(initial)).tolist())
            finally:
                state.close()
    finally:
        sys.path.remove(str(atlas_root))
        sys.path.remove(str(model_root))
