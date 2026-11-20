"""npu_model subprocess adapter; expected numerical values never enter here."""
import argparse
import dataclasses
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "third_party/atlas-mlir"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()

    import npu_model
    import torch
    from npu_model.configs import isa_definition as isa
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software.program import InstantiableProgram

    sys.path.insert(0, str(ATLAS))
    sys.path.insert(0, str(ATLAS / "test"))
    try:  # atlas-mlir test-internal helpers, not a stable API
        from test_spill_model_execution import model_instruction
        from tools.check_program_decode import decode_binary
    except ImportError as error:
        raise ImportError("atlas-mlir's model_instruction or decode_binary moved; update run_model.py") from error

    torch.set_num_threads(1)
    emitted = json.loads(args.program.read_text())
    inventory = json.loads((ATLAS / "docs/selected-variant-inventory.json").read_text())
    if emitted["selected_rtl_revision"] != inventory["selected_sources"]["rtl_revision"]:
        raise ValueError("emitter and model adapter select different RTL revisions")
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in emitted["instructions"])
    decoded = decode_binary(inventory, binary)
    program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded["instructions"]])
    request = json.loads(args.request.read_text())
    initial = bytes.fromhex(request["initial_hex"])
    program.dram_base = request["base"]
    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
    config = DefaultHardwareConfig()
    config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics="rtl", randomize_init=True,
                                                   init_seed=187)
    simulation = Simulation(config, LoggerConfig(filename=str(args.stats.with_suffix(".trace.json"))), program,
                            verbose=False)
    state = simulation.core.arch_state
    try:
        simulation.run(max_cycles=100000)
        stats = simulation.get_stats()
        revision = subprocess.run(["git", "-C", str(Path(npu_model.__file__).parent), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip() or "unknown"
        report = dict(numerics="rtl", npu_model_revision=revision, halted=state.halted,
                      runtime_errors=stats.runtime_errors, cycles=stats.cycles)
        args.stats.write_text(json.dumps(report, indent=2) + "\n")
        if not state.halted or stats.runtime_errors:
            raise RuntimeError(f"model execution failed: {report}")
        args.out.write_bytes(bytes(state.read_dram(program.dram_base, len(initial)).tolist()))
    finally:
        state.close()


if __name__ == "__main__":
    main()
