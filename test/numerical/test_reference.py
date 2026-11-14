"""Portable oracle tests; run independently of instruction execution."""
from fractions import Fraction
import unittest

from reference import bf16, deterministic_inputs, fp8, matmul_relu, round_bf16


class ReferenceTest(unittest.TestCase):

    def test_decoding(self):
        for code, value in ((0, 0), (0x80, 0), (0x30, Fraction(1, 2)), (0x38, 1), (0xBC, Fraction(-3, 2)), (0x7E, 448)):
            self.assertEqual(fp8(code), value)
        for code in (1, 0x7F, 0xFF):
            with self.assertRaises(ValueError):
                fp8(code)

    def test_rounding_ties_and_carry(self):
        self.assertEqual(round_bf16(1 + Fraction(1, 256)), 0x3F80)
        self.assertEqual(round_bf16(1 + Fraction(3, 256)), 0x3F82)
        self.assertEqual(round_bf16(-1 - Fraction(3, 256)), 0xBF82)
        self.assertEqual(round_bf16(2 - Fraction(1, 256)), 0x4000)
        self.assertEqual(round_bf16(0), 0)
        for code in (0x0080, 0x3F80, 0xBF81, 0x7F7F):
            self.assertEqual(round_bf16(bf16(code)), code)

    def test_k_order_rounding_is_not_one_final_cast(self):
        a, b = bytearray(1024), bytearray(1024)
        a[:3] = bytes((0x38, 0x18, 0x18))
        b[0], b[32], b[64] = 0x38, 0x18, 0x18
        result = matmul_relu(a, b)
        self.assertEqual(int.from_bytes(result[:2], "little"), 0x3F80)
        self.assertEqual(round_bf16(1 + Fraction(2, 256)), 0x3F81)
        self.assertEqual(result[2:], bytes(2046))

    def test_deterministic_inputs_and_relu(self):
        a, b = deterministic_inputs()
        self.assertEqual((a, b), deterministic_inputs())
        self.assertNotEqual(a, b)
        self.assertTrue(any(fp8(x) < 0 for x in a))
        # All-one A and negative-one B must be entirely zero after ReLU.
        self.assertEqual(matmul_relu(bytes([0x38]) * 1024, bytes([0xB8]) * 1024), bytes(2048))

    def test_row_major_orientation_with_identity(self):
        a, _ = deterministic_inputs()
        identity = bytearray(1024)
        for i in range(32):
            identity[i * 32 + i] = 0x38
        expected = b"".join(round_bf16(max(fp8(code), 0)).to_bytes(2, "little") for code in a)
        self.assertEqual(matmul_relu(a, identity), expected)


if __name__ == "__main__":
    unittest.main()
