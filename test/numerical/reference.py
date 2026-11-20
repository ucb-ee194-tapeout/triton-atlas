"""Independent exact-rational oracle for bounded, normal FP8 MXU0 inputs.

No Torch or npu_model imports. This is a finite-normal diagnostic, not an
implementation of Atlas NaN, infinity, subnormal or overflow behavior.
"""
from fractions import Fraction
import struct


def power2(exponent):
    return Fraction(2)**exponent


def fp8(code):
    exponent, mantissa = (code >> 3) & 15, code & 7
    if code & 127 == 0:
        return Fraction(0)
    if exponent == 0 or (exponent == 15 and mantissa == 7):
        raise ValueError("reference requires zero or finite normal E4M3FN")
    return (-1 if code & 128 else 1) * Fraction(8 + mantissa, 8) * power2(exponent - 7)


def bf16(code):
    if code & 32767 == 0:
        return Fraction(0)
    exponent = (code >> 7) & 255
    if exponent in (0, 255):
        raise ValueError("reference requires zero or finite normal BF16")
    return (-1 if code & 32768 else 1) * Fraction(128 + (code & 127), 128) * power2(exponent - 127)


def round_bf16(value):
    """Round an exact rational to normal BF16, nearest with ties to even."""
    value = Fraction(value)
    if not value:
        return 0
    sign = 32768 if value < 0 else 0
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power2(exponent):
        exponent -= 1
    if not -126 <= exponent <= 127:
        raise ValueError("reference outside finite normal BF16 range")
    scaled = value / power2(exponent - 7)
    significand, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = remainder * 2
    if twice > scaled.denominator or (twice == scaled.denominator and significand & 1):
        significand += 1
    if significand == 256:
        significand = 128
        exponent += 1
    if exponent > 127:
        raise ValueError("reference overflow")
    return sign | ((exponent + 127) << 7) | (significand - 128)


def deterministic_inputs():
    """Two distinct tiles, fixed LCG seed, zeros and signed normal [.5, 2]."""
    choices = (0, *range(0x30, 0x41), *range(0xB0, 0xC1))
    state = 0xA71A5
    tiles = []
    for _ in range(2):
        tile = bytearray()
        for _ in range(1024):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            tile.append(choices[(state >> 16) % len(choices)])
        tiles.append(bytes(tile))
    return tuple(tiles)


def matmul_relu(a, b):
    """A[M,K] @ B[K,N], K-ordered BF16 RNE after each exact product-add.

    The fixture stays far from exceptional values. Products of its normal
    E4M3 operands are exactly representable in BF16; there is no separately
    rounded multiply. ReLU follows the final BF16 accumulation.
    """
    if len(a) != 1024 or len(b) != 1024:
        raise ValueError("reference requires two 32x32 FP8 tiles")
    av, bv = list(map(fp8, a)), list(map(fp8, b))
    result = []
    for m in range(32):
        for n in range(32):
            accumulator = 0
            for k in range(32):
                accumulator = round_bf16(bf16(accumulator) + av[m * 32 + k] * bv[k * 32 + n])
            result.append(0 if accumulator & 32768 else accumulator)
    return b"".join(code.to_bytes(2, "little") for code in result)


def f32_values_from_bf16(tile):
    return [struct.unpack("<f", b"\0\0" + tile[i:i + 2])[0] for i in range(0, len(tile), 2)]
