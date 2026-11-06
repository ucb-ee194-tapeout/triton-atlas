# Boundaries + Stream lane (M1)

The lane now uses the existing pipeline's Pass 5, `ttatlas-assign-slots`, and
its frozen `#ttatlas.slot` attribute. The standalone `atlas-analyze-boundaries`
pass, dictionary attribute, registration, and tests have been retired after
migrating the pointer proof and its tests. No TableGen encoding or pipeline
order changed. The old dictionary schema tests are superseded by the existing
frozen-attribute verifier tests. Physical placement belongs to the documented
Pass 6 ABI, not pointer analysis.

## Pass 5

The native C++ analysis evaluates every pointer lane of the fixed 32x32 tile.
It handles function pointer arguments, splat, expand_dims, broadcast, chained
addptr, integer scalar/dense constants, make_range, and addi/subi/muli.
APInt preserves wrapping; poison-producing nsw/nuw overflow is rejected.
Every lane must equal `argument + offset + row*32 + column`. Hints such as
`tt.divisibility` are not accepted as proof. Offsets must be nonnegative,
row-aligned, and fit signed i64; tile_origin is the six-symbol affine map with
constant results `(offset / 32, 0)`. Other row strides/layouts need a richer
host-view contract and are rejected, even if each row separately is contiguous.

Pass 1's `ttatlas.host_dtype` is required and copied verbatim into SlotAttr.
Thus a legalized BF16 output can retain host dtype f32. The operation's current
pointer/value dtype selects virtual input/output types, not the host dtype.
SlotAttr's actual generated fields are ArrayRef<int64_t> and Type, rather than
a separately invented DenseI64Array/TypeAttr encoding.

Input indices are dense from zero in first-access order, with a separate dense
output space. Repeated inputs reuse an index only when argument, origin, role,
logical shape, and host dtype all match. The current dtype is fixed by the
source pointer argument; no pointer casts are accepted. Stores remain distinct
ordered output events. Loads after any store are rejected conservatively.
All attributes are published only after the complete module succeeds.

Role inference follows def-use paths through float casts and 2-D transposes:

- Dot operand A without transpose: activation_fp8.
- Dot operand B directly: weight_fp8_kn. With an odd number of [1,0]
  transposes: weight_fp8_nk.
- A BF16 input used by addf: bias_bf16; other BF16 inputs: constant_bf16.
- A BF16 load with a neg_inf pad tag: mask_bf16.
- Stores: output_bf16.

Ambiguous activation/weight use, unsupported activation orientation, and FP8
inputs with no provable dot role fail. This does not provide general softmax
mask inference. Explicit memory masks, volatile loads, nested control flow,
block pointers, dynamic offsets/strides, noncontiguous tiles, and FP8 stores
fail with the shared emitUnsupported diagnostic. Pad tags on loads/stores or
store producers are consumed; partial logical shapes fail pending Stage B
packing semantics. A 1-D bias is also explicitly deferred: a logical_shape
alone does not define the driver's broadcast/packing behavior.

## Pass 6 boundary conversion

`populateBoundaryPatterns` registers a tt.func conversion through the existing
Patterns.h hook. The pattern validates hand-tagged loads/stores, builds an
argument-free func.func returning !atlas.virtual_state, and emits one input per
unique slot in ascending input-index order. Repeated input indices must carry
identical slot attributes and element types. Outputs preserve source order and
must have unique, dense indices. A virtual_start/inputs/outputs/return chain
threads state linearly. The original typed SlotAttr is retained on virtual I/O
for KernelMetadata.cpp; no new attribute is introduced.

The pattern clones compute operations for the existing compute patterns to
convert. It does not implement their type conversions or mappings. Pointer
arguments may feed only the supported address grammar, which is removed
instead of being copied. Address values escaping into compute are rejected.
The shared conversion driver marks tt.func, tt.load, and tt.store illegal
(the boundary legality change was approved). Other TTIR remains subject to the
existing compute conversion and stage-6 contract checker.

## Metadata: implemented and gated

Jonathan's design agenda v15 (November 5) defines the v1 field set:
`version`, `name`, `dram_base: {input, output}`, `inputs`, `outputs`, `grid`,
and `legalized`. Each slot table entry contains `index`, `arg`, `role`,
`tile_origin`, `logical_shape`, `host_dtype`, and `layout`.

Implemented unambiguous ABI work:

- Input base from module ttatlas.dram_base, default 0x90000000.
- Output base = input base + number of input slots * 2048.
- Matching i64 atlas.input_dram_base / atlas.output_dram_base on func.func.
- Existing base attributes must agree. AtlasBoundaryABI checks alignment,
  selected DRAM, disjoint spans, and 32-bit address limits.
- Preserved SlotAttrs must agree with virtual I/O indices. Slot indices are
  dense and consistent. A supplied manifest's version, name, bases, slot
  counts/index/argument/role, and fixed [1,1,1] grid are checked.

Automatic manifest generation is deliberately gated: **without a supplied
manifest the conversion does not attach ttatlas.kernel**, and the stage-6
contract checker reports the missing manifest. No partial JSON is published.
Full slot-table serialization, layout serialization/defaulting, and copying
legalization records into a newly generated manifest await the confirmed schema.
Zero-input kernels also need a decision: Atlas's current ABI reserves at least
one input slot, whereas the documented placement formula counts actual inputs.
M1 currently requires nonempty input/output tables rather than inventing a
reservation rule.

The Python compiler uses native module.walk/get_str_attr to read the converted
func.func's manifest, checks version/name, and copies its exact JSON string
into cached metadata under `ttatlas_kernel`. Dotted metadata keys cannot be
used because Triton turns metadata into a Python namedtuple. The exporter
preserves Triton fields, sets the function name and shared=0, and fails clearly
if no finalized manifest exists. The driver is unchanged.

### Decisions for Jonathan's schema file and golden example

Smallest proposed choices, **not implemented/frozen by this lane**:

1. Serialize tile_origin as canonical MLIR affine-map assembly in a JSON
   string, including all six symbols. This preserves future symbolic origins
   without inventing a JSON expression language. Alternative: M1-only [row,col]
   constants, but that requires a later schema decision for symbolic maps.
2. Serialize host_dtype as its MLIR scalar type spelling (`bf16`, `f32`,
   `f8E4M3FN`), and layout as the existing enum spelling (`row_major` or
   `tile_major`). Confirm whether Triton dtype spellings are preferred instead.
3. Require an explicit module layout until the backend's default is agreed.
4. Sort input/output table entries by index; preserve the existing
   legalization record objects `{pass, what, loc}` unchanged, default [] only
   when ttatlas.legalized is absent. Use JSON integer addresses and grid values.
5. Confirm `ttatlas_kernel` as the driver-facing cached metadata key, containing
   the original JSON string, and the zero-input window rule above.

Once confirmed, check in the v1 JSON schema and a Stage A golden manifest with
input indices 0/1 (activation_fp8/weight_fp8_kn), output index 0 (output_bf16),
host output f32, origin (0,0), shape [32,32], grid [1,1,1], and bases
0x90000000/0x90001000. Then enable generation, record-copying, schema validation,
and the complete stage-6 contract test. Until then metadata tests distinguish
ABI checks and transport from a complete manifest producer.

## Integration limits

The checkout still has placeholder passes 1–4 and identity-only compute type
conversion. The raw canonical matmul+ReLU fixture needs Pass 1's dtype tags and
Pass 4's f32-to-BF16 legalization before Pass 5. The independent Pass 5 fixture
is prepared at that boundary without editing the shared raw fixture. Full
matmul conversion also awaits Devansh's dot/transpose/ReLU/type patterns.
Neither their files nor the driver was modified.
