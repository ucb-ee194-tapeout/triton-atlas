"""Execute an existing Atlas fixture; this does not test a Triton kernel."""
import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from reference import f32_values_from_bf16

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "third_party/atlas-mlir"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-bin", type=Path, default=ROOT / "build/atlas/bin")
    args = parser.parse_args()

    import torch
    from npu_model.configs import isa_definition as isa
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software.program import InstantiableProgram

    # Reuse the vendored adapter, including its bytecode identity check.
    sys.path.insert(0, str(ATLAS))
    sys.path.insert(0, str(ATLAS / "test"))
    from test_spill_model_execution import model_instruction
    from tools.check_program_decode import decode_binary

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="ttatlas-model-smoke-") as directory:
        directory = Path(directory)
        machine = directory / "existing.machine.mlir"
        subprocess.run([
            str(args.atlas_bin / "atlas-opt"),
            str(ATLAS / "test/examples/virtual_dma_mxu_bf16.mlir"), "--atlas-compile", "-o",
            str(machine)
        ], check=True)
        emitted = json.loads(
            subprocess.check_output([str(args.atlas_bin / "atlas-emit"), "--program-json",
                                     str(machine)], text=True))
        inventory = json.loads((ATLAS / "docs/selected-variant-inventory.json").read_text())
        binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in emitted["instructions"])
        decoded = decode_binary(inventory, binary)
        program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded["instructions"]])
        # These addresses belong to the existing fixed fixture, not a Triton manifest.
        program.dram_base = 0x90000000
        program.memory_regions = [(0, torch.full((2048, ), 0x38, dtype=torch.uint8))]
        config = DefaultHardwareConfig()
        config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics="rtl")
        simulation = Simulation(config, LoggerConfig(filename=str(directory / "trace.json")), program, verbose=False)
        state = simulation.core.arch_state
        try:
            simulation.run(max_cycles=100000)
            stats = simulation.get_stats()
            if not state.halted or stats.runtime_errors:
                raise RuntimeError(f"model halted={state.halted}, runtime_errors={stats.runtime_errors}")
            output = bytes(state.read_dram(0x90001000, 2048).tolist())
            values = f32_values_from_bf16(output)
            expected = bytes.fromhex("8042") * 1024  # two all-one 32-term contractions = 64
            mismatches = [
                dict(row=i // 32, column=i % 32, actual=value, expected=64.0)
                for i, value in enumerate(values)
                if output[2 * i:2 * i + 2] != expected[2 * i:2 * i + 2]
            ]
            report = dict(fixture="virtual_dma_mxu_bf16", numerics="rtl", cycles=stats.cycles,
                          runtime_errors=stats.runtime_errors, exact_matches=1024 - len(mismatches),
                          max_abs_error=max(abs(value - 64.0) for value in values), mismatches=mismatches)
            print(json.dumps(report, indent=2))
            return int(bool(mismatches))
        finally:
            state.close()


if __name__ == "__main__":
    raise SystemExit(main())
