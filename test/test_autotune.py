"""Portable tests for the candidate ranking contract."""

import unittest
from unittest import mock

from triton_atlas_backend.autotune import AtlasAutotuner, Candidate


class AutotuneTest(unittest.TestCase):

    def test_selects_lowest_cycle_candidate_after_validation(self):
        candidates = [
            Candidate("slow", object(), {"x": "*bf16"}),
            Candidate("fast", object(), {"x": "*bf16"}),
            Candidate("wrong", object(), {"x": "*bf16"}),
        ]
        tuner = AtlasAutotuner(candidates)
        programs = {candidate.name: {"candidate": candidate.name} for candidate in candidates}
        tuner._compile = lambda candidate, *, dram_base: programs[candidate.name]
        cycles = {"slow": 20, "fast": 10, "wrong": 1}

        def execute(program, initial, dram_base, *, max_cycles):
            return cycles[program["candidate"]], program["candidate"].encode()

        def validate(candidate, program, memory):
            if candidate.name == "wrong":
                raise ValueError("output mismatch")

        with mock.patch("triton_atlas_backend.autotune.npu_model.run", side_effect=execute):
            result = tuner.tune(
                lambda candidate, program: candidate.name.encode(),
                validator=validate,
            )
        self.assertEqual(result.selected.name, "fast")
        self.assertEqual([item.status for item in result.measurements], ["ok", "ok", "rejected"])
        self.assertIn("output mismatch", result.measurements[-1].error)

    def test_requires_validator(self):
        tuner = AtlasAutotuner([Candidate("one", object(), {})])
        with self.assertRaisesRegex(ValueError, "output validator"):
            tuner.tune(b"input")


if __name__ == "__main__":
    unittest.main()
