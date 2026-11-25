"""Model-backed autotuning for Atlas Triton kernels.

The Atlas lowering pipeline already contains the legality, tiling, scheduling,
delay, slot-assignment, spilling, and machine-emission heuristics.  This
module deliberately keeps autotuning above those passes: each candidate is a
normal ``@triton.jit`` function, and the objective is measured by executing
the emitted program on the pinned cycle-accurate ``npu_model``.

Candidates are only comparable when the caller supplies an output validator.
An invalid or non-terminating candidate is recorded as rejected instead of
being allowed to win on a partial measurement.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import npu_model


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One legal Triton implementation to measure."""

    name: str
    fn: Callable[..., Any]
    signature: Mapping[str, str]
    constexprs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    options: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Measurement:
    """The result of compiling and executing one candidate."""

    name: str
    status: str
    cycles: int | None = None
    elapsed_ms: float | None = None
    error: str | None = None
    program: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TuneResult:
    """All measurements and the selected candidate."""

    selected: Measurement
    measurements: list[Measurement]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.as_dict(),
            "measurements": [measurement.as_dict() for measurement in self.measurements],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n")


class AtlasAutotuner:
    """Compile and rank Triton candidates using real Atlas model cycles.

    ``validator`` receives ``(candidate, program_json, memory_image)`` after
    execution.  It should raise ``ValueError`` (or another exception) when a
    candidate's output or memory guards are wrong.  The tuner then marks that
    candidate rejected and never selects it.  ``initial`` may be a callable
    when a candidate's manifest changes host packing (for example, a
    transposed input slot).
    """

    def __init__(
        self,
        candidates: Sequence[Candidate],
        *,
        atlas_bin: str | Path | None = None,
        triton_atlas_opt: str | Path | None = None,
        dram_base: int = 0x90000000,
        model_dram_base: int | None = None,
        max_cycles: int = 500_000,
    ) -> None:
        if not candidates:
            raise ValueError("AtlasAutotuner needs at least one candidate")
        if len({candidate.name for candidate in candidates}) != len(candidates):
            raise ValueError("candidate names must be unique")
        self.candidates = tuple(candidates)
        self.atlas_bin = atlas_bin
        self.triton_atlas_opt = triton_atlas_opt
        self.dram_base = dram_base
        # Leave a guard page before the first kernel slot, matching the
        # numerical harness.  The compiler's DRAM base is the first device
        # slot; the model base is where the returned memory image begins.
        self.model_dram_base = model_dram_base if model_dram_base is not None else dram_base - 1024
        self.max_cycles = max_cycles

    def _compile(self, candidate: Candidate, *, dram_base: int):
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import ASTSource

        options = dict(candidate.options)
        options.setdefault("dram_base", dram_base)
        source = ASTSource(
            fn=candidate.fn,
            signature=dict(candidate.signature),
            constexprs=dict(candidate.constexprs),
        )
        compiled = triton.compile(
            source,
            target=GPUTarget("atlas", "virtual", 1),
            options=options,
        )
        virtual = compiled.asm.get("atlas_virtual")
        if virtual is None:
            raise RuntimeError(f"candidate {candidate.name!r} did not produce atlas_virtual IR")
        program = npu_model.emit(
            virtual,
            atlas_bin=self.atlas_bin,
            triton_atlas_opt=self.triton_atlas_opt,
        )
        manifest = getattr(compiled.metadata, "ttatlas_kernel", None)
        if manifest is None:
            raise RuntimeError(f"candidate {candidate.name!r} did not export ttatlas_kernel metadata")
        # Keep the host-boundary manifest beside the emitted program.  The
        # npu_model ignores this private key; validators use it to pack inputs
        # and unpack outputs without re-running Triton's frontend.
        program["kernel_manifest"] = manifest
        return program

    def tune(
        self,
        initial: bytes | bytearray | Callable[[Candidate, Mapping[str, Any]], bytes],
        *,
        validator: Callable[[Candidate, Mapping[str, Any], bytes], None] | None = None,
        on_measurement: Callable[[Measurement], None] | None = None,
    ) -> TuneResult:
        """Measure all candidates and return the lowest-cycle valid one."""

        if validator is None:
            raise ValueError("an output validator is required before ranking Atlas candidates")
        measurements: list[Measurement] = []
        for candidate in self.candidates:
            started = time.perf_counter()
            measurement = Measurement(name=candidate.name, status="rejected")
            try:
                program = self._compile(candidate, dram_base=self.dram_base)
                model_initial = initial(candidate, program) if callable(initial) else bytes(initial)
                cycles, memory = npu_model.run(
                    program,
                    model_initial,
                    self.model_dram_base,
                    max_cycles=self.max_cycles,
                )
                validator(candidate, program, memory)
                measurement.status = "ok"
                measurement.cycles = cycles
                measurement.program = program
            except Exception as error:  # candidates are user-supplied search points
                measurement.error = f"{type(error).__name__}: {error}"
            measurement.elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            measurements.append(measurement)
            if on_measurement is not None:
                on_measurement(measurement)

        valid = [measurement for measurement in measurements if measurement.status == "ok"]
        if not valid:
            details = "; ".join(f"{item.name}: {item.error}" for item in measurements)
            raise RuntimeError(f"no valid Atlas autotune candidates ({details})")
        selected = min(valid, key=lambda item: item.cycles if item.cycles is not None else float("inf"))
        return TuneResult(selected=selected, measurements=measurements)
