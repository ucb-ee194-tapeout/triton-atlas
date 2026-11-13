# Atlas boundary conversion and metadata

## Pass 6: logical tile slots

Pass 6 assigns deterministic input and output slots after shape and dtype
legalization. It consumes `ttatlas.host_dtype` from Pass 1 and optional
`#ttatlas.pad<neutral = ..., logical_shape = [...]>` from Pass 4. The frozen
SlotAttr fields and enum values are unchanged.

Supported boundary forms are:

- Direct unmasked 32x32 loads and BF16 stores, tagged on the memory operation.
- Larger rank-two loads feeding only constant 32x32, unit-stride
  `tensor.extract_slice` operations, each tagged with its slot.
- Larger stores assembled by a disjoint, complete, tile-aligned chain of
  32x32 `tensor.insert_slice` operations into a fresh `tensor.empty`, with
  output slots on the inserts.
- Contiguous 32-element BF16 input vectors, represented as `bias_bf16` slots
  with logical shape `[1,32]` and origin `(0, element_offset)`. A padding tag
  `[1,N]` describes a shorter valid prefix. Rank-one outputs are unsupported.

The native pointer evaluator supports function pointer arguments,
splat/expand_dims/broadcast, chained addptr, make_range, scalar/dense integer
constants, and integer add/subtract/multiply. APInt preserves wrapping and
rejects poison-producing nsw/nuw overflow. Hints do not substitute for proof.
Each rank-two lane must prove
`argument + first_offset + row * parent_width + column`; tile origins include
both parent pointer and slice offsets. A pointer argument must use a consistent
host row width throughout the kernel.

Input slots follow memory/slice source order. Identical tiles reuse a slot only
when role, original host dtype, logical shape and padding neutral agree.
Outputs use a separate dense index space. Repeated writes to a logical output
tile and loads after stores fail. Attributes are published only after successful
module analysis.

FP8 role inference follows dot/matmul operands, float/tensor casts and
transposes: direct B is KN; a transposed B source is NK. Transposed activations
and ambiguous roles fail. BF16 addition inputs are classified as biases;
other BF16 inputs are constants, or masks when tagged with neg_inf padding.
Rank-one BF16 inputs are classified as bias rows without proving their later
broadcast direction.

## Partial tiles and host buffers

Pass 6 preserves direct tiles' unpadded logical rectangles. For slices it
intersects a parent's logical extent with the slice offsets. Output tags can
come from the stored value's producer. Invalid extents fail; full tiles retain
`[32,32]`. Physical IR tiles remain 32x32 even when their logical shapes are
smaller.

`python/triton_atlas_backend/boundary.py` provides independent host-buffer
primitives:

- `pack_input_tile` copies valid elements and fills invalid lanes with the
  tagged zero or negative-infinity neutral. It accepts already encoded,
  little-endian BF16 or E4M3FN bytes and rejects FP8 negative infinity.
- Rank-one BF16 bias packing repeats the padded row 32 times. This is the
  proposed row-broadcast convention for Devansh's broadcast-erasure patterns.
- `unpack_output_tile` writes only the logical BF16 rectangle, preserving
  bytes outside that rectangle in the destination buffer.

These helpers require the original host shape from their caller. They perform
no dtype conversion, allocation, device transfer or launch. Original host dtype
metadata, including f32, is retained faithfully, but helpers reject unsupported
encoded host dtypes. AtlasDriver's launcher remains a stub and does not call
these helpers. Physical buffer tests prove helper behavior, not device runtime
padding/cropping or kernel execution.

## Pass 7: boundary conversion

Boundary conversion creates an argument-free func.func returning
`!atlas.virtual_state`. Inputs are emitted by slot index; reused inputs share a
virtual value. Outputs follow original store/assembly order with a linear state
chain. SlotAttr remains on virtual I/O, and normalized PadAttr remains on virtual
inputs for metadata generation.

The conversion removes boundary loads/stores, supported slices/assemblies and
proven address scaffolding. Escaping addresses, unrelated slices, nested memory,
masked accesses and unsupported control flow fail. Linalg compute regions are
cloned for compute patterns. The shared driver marks TTIR boundary operations
illegal but still uses partial conversion; the Stage 7 contract and Atlas stream
verifier are necessary to establish completed conversion. This branch does not
implement Devansh's compute patterns.

## Kernel metadata

The packaged schema is
`python/triton_atlas_backend/schemas/ttatlas.kernel.v1.schema.json`; an example
manifest is `docs/examples/ttatlas.kernel.v1.json`.

Metadata includes version, name, DRAM bases, input/output tables, grid and
legalization history. Each slot records index, pointer argument, role, constant
nonnegative affine tile origin, logical shape, original host dtype and layout.
Logical dimensions are 1..32; input entries can additionally carry
`pad_neutral` (`zero` or `neg_inf`). The additive field and `[1,N]` convention
need consumer agreement before becoming an integrated runtime ABI.

Tables use dense sorted indices. Slots are 2048 bytes, default input base is
0x90000000, and output placement immediately follows inputs. Grid is `[1,1,1]`.
AtlasBoundaryABI checks alignment, disjoint windows and 32-bit address bounds.
Empty input/output tables fail. Existing manifests must match generated fields.

The Python exporter validates the schema and index/address relationships before
copying the manifest verbatim into Triton's `ttatlas_kernel` cache field. MLIR
uses `ttatlas.kernel`. Backend options and schema contents participate in cache
hashing. Placement options are normalized from the private
`atlas_backend.options` handoff to typed module attributes; explicit settings
must agree.

## Upstream integration limits

The review on November 12, 2026 fetched main at `694e1b2`, Devansh's
`devansh/pass7-compute` at `5f167b8`, Jonathan's `feature/legalize-dtypes` at
`21b067d`, and the newer `dev/legalize-dtypes` at `92111d0`. Main has not been
merged or rebased into this branch. Main and the newer dtype branch still have
a placeholder Pass 4; the feature/compute branches contain shape legalization.

Remaining integration work:

- The implemented Pass 4 paths tag masked loads, stored-value producers and
  tensor.pad operations. Masked memory and general tensor.pad/slice/crop graphs
  remain outside this collector's contract. A padding tag alone does not prove
  an arbitrary mask safe.
- Some rank-one Pass 4 paths construct one-dimensional padding shapes, while
  PadAttr's verifier requires two dimensions. Agree on normalization with
  Jonathan and Devansh without changing the shared attribute unilaterally.
- Prove supported bias broadcast directions and test an actual bias-add
  computation through compute conversion. The native bias test currently
  checks an unused vector's slot and metadata.
- Preserve Devansh's boundary-converter changes when integrating compute.
  Jonathan's feature branch has textual conflicts in the shared conversion
  driver and compiler.py; current main has no detected textual conflicts.
- Pointer proof assumes parent tensor width is the host row stride. Different
  strides and compact host layouts need an agreed interface; the metadata has
  no separate host shape/stride field.
- Wire helpers into a runtime with host dtype conversion, slot transfers and
  launch. Python plugin registration also lacks Tensor/Linalg dialects for
  mixed-IR parsing; main's newer Linalg linking change alone does not register
  those dialects.

No Pass 4 implementation or compute-pattern/type-conversion source is changed
by this boundary work. Shared verifier/driver integration is limited to the
boundary contract.

## Validation

The November 12 review of the current implementation produced:

- Plugin/tools build: passed with `make build.tools`.
- Relevant slot/boundary/metadata/compiler/contract suite: 52/52 lit tests
  passed, including 8 buffer/native-metadata Python cases, 7 slice cases and
  9 metadata-export cases inside lit wrappers.
- Full suite: 62 passed, 1 expected failure, 1 failure. Stage A still fails
  because Pass 1 does not attach host_dtype tags. This is an upstream pipeline
  blocker, not evidence of successful end-to-end execution.

Tests cover full tiles, partial extents and slice intersections, padding
neutrals, bias slots and row replication, output cropping, origins, roles,
dtypes, slot reuse and rejection of unsupported forms. Lit uses Python's fork
start method because the sandbox disallows the default forkserver socket.
