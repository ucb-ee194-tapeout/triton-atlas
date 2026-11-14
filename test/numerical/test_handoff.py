"""Compile-only regression using the real JIT kernel; no model dependency."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from compile_matmul_relu import DEFAULT_TRITON_ATLAS_OPT, ROOT, run


class HandoffTest(unittest.TestCase):

    def assert_preserved(self, baseline, finalized):
        header, body = baseline.split("\n", 1)
        self.assertEqual(body, finalized.split("\n", 1)[1])
        self.assertEqual(
            set(re.findall(r"ttatlas\.[a-z_]+", header)),
            {"ttatlas.dram_base", "ttatlas.fold_bias_into_dot", "ttatlas.layout", "ttatlas.precision_policy"})
        # Expected-text filtering only: the production transformation uses MLIR.
        # Compare every instruction, location and non-TTAtlas module attribute.
        for pattern in (r', ttatlas\.dram_base = \d+ : i64', r', ttatlas\.fold_bias_into_dot = (?:true|false)',
                        r', ttatlas\.layout = #ttatlas<layout row_major>',
                        r', ttatlas\.precision_policy = "fp8-bf16-v1"'):
            header, count = re.subn(pattern, "", header)
            self.assertEqual(count, 1)
        self.assertEqual(header + "\n" + body, finalized)
        self.assertNotIn("ttatlas.", finalized)

    def test_real_kernel_handoff_in_default_and_debug_modes(self):
        with tempfile.TemporaryDirectory(prefix="ttatlas-handoff-") as directory:
            atlas_bin = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/atlas/bin"))
            opt = DEFAULT_TRITON_ATLAS_OPT.absolute()
            # Exercise the CLI override through a non-default executable path.
            selected_opt = Path(directory) / "selected-triton-atlas-opt"
            selected_opt.symlink_to(opt)
            artifacts = []
            for debug in (False, True):
                with self.subTest(debug=debug):
                    output = Path(directory) / ("debug" if debug else "default")
                    printing = ["--mlir-print-debuginfo"] if debug else []
                    run([
                        sys.executable, ROOT / "test/numerical/compile_matmul_relu.py", "--out", output, "--atlas-bin",
                        atlas_bin, "--triton-atlas-opt", selected_opt, "--dram-base", "0x91000000",
                        *(["--debug"] if debug else [])
                    ])
                    report = json.loads((output / "compile-report.json").read_text())
                    self.assertEqual(report["debug"], debug)
                    self.assertTrue(report["instruction_emission"])
                    baseline = run([opt, output / "kernel.machine.mlir", *printing])
                    finalized = (output / "kernel.emit.mlir").read_text()
                    self.assert_preserved(baseline, finalized)
                    for source in (baseline, finalized):
                        if debug:
                            self.assertIn(" loc(", source)
                            self.assertRegex(source, r'kernels\.py":\d+:\d+')
                        else:
                            self.assertNotIn(" loc(", source)
                    self.assertIn("ttatlas.layout", (output / "kernel.virtual.mlir").read_text())
                    self.assertEqual(
                        run([opt, output / "kernel.emit.mlir", "--ttatlas-finalize-machine-handoff", *printing]),
                        finalized)
                    artifacts.append(
                        ((output / "kernel.json").read_text(), json.loads(
                            (output / "kernel.program.json").read_text())))

                    # Printing controls visibility, not the native pass's handling
                    # of locations. Reparse with debug printing to inspect explicit
                    # fused/named locations and emitter-supported debug attributes.
                    located = run([opt, output / "kernel.machine.mlir", "--mlir-print-debuginfo"])
                    annotated = located.replace('{atlas.virtual_cfg_block =',
                                                '{debug.note = "keep-me", atlas.virtual_cfg_block =', 1)
                    annotated = annotated.replace(
                        'loc(#loc)', 'loc(fused["handoff-debug.mlir":7:9, '
                        '"named"("handoff-debug.mlir":11:3)])', 1)
                    annotated_path = output / "annotated.mlir"
                    annotated_path.write_text(annotated)
                    before = run([opt, annotated_path, "--mlir-print-debuginfo"])
                    annotated_final = output / "annotated.emit.mlir"
                    run([
                        opt, annotated_path, "--ttatlas-finalize-machine-handoff", "--mlir-print-debuginfo", "-o",
                        annotated_final
                    ])
                    after = annotated_final.read_text()
                    self.assert_preserved(before, after)
                    self.assertIn('debug.note = "keep-me"', after)
                    self.assertIn("fused[", after)
                    self.assertIn('"named"(', after)
                    self.assertIn('handoff-debug.mlir":7:9', after)
                    emitted = json.loads(run([atlas_bin / "atlas-emit", "--program-json", annotated_final]))
                    self.assertTrue(any(row["fields"].get("debug.note") == "keep-me"
                                        for row in emitted["instructions"]))
                    self.assertEqual([row["word_u32"] for row in emitted["instructions"]],
                                     [row["word_u32"] for row in artifacts[-1][1]["instructions"]])

                    def rejected(source, diagnostic):
                        result = subprocess.run([str(opt), "--ttatlas-finalize-machine-handoff"], input=source,
                                                capture_output=True, text=True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(diagnostic, result.stderr)

                    rejected('module {}', "expected an Atlas virtual-to-machine artifact")
                    rejected(baseline.replace('atlas.timing_state = "timed"', 'atlas.timing_state = "untimed"'),
                             "timed")
                    rejected(baseline.replace("module attributes {", "module attributes {ttatlas.scale = 127 : i32, "),
                             "unexpected TTAtlas metadata")
                    rejected(
                        baseline.replace('"atlas.start"() :',
                                         '"atlas.start"() {ttatlas.layout = #ttatlas<layout row_major>} :'),
                        "unexpected TTAtlas metadata")
            # Debug visibility cannot affect the host manifest or emitted program.
            self.assertEqual(artifacts[0], artifacts[1])


if __name__ == "__main__":
    unittest.main()
