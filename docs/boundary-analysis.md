# Boundaries + Stream lane (M1)

## Pass 6: logical tile slots

The native C++ pointer evaluator from `b47dca6` remains the proof engine:
function pointer arguments, splat/expand_dims/broadcast, chained addptr,
make_range, scalar/dense integer constants, and add/subtract/multiply.
APInt preserves integer wrapping and rejects poison-producing nsw/nuw overflow.
Hints never substitute for an address proof.

The approved slice contract keeps the frozen SlotAttr fields unchanged:

- A direct 32x32 load/store carries its own `#ttatlas.slot`.
- A larger load is used only by constant 32x32, unit-stride
  `tensor.extract_slice` operations. Each slice carries its slot.
- A larger store consumes a disjoint, complete chain of constant, tile-aligned
  32x32 `tensor.insert_slice` operations into a fresh `tensor.empty`. Each
  insert carries its output slot. Shared or overlapping assemblies fail.

`BoundaryTiles.h` shares structural interpretation between assignment, boundary
lowering and the Stage 6 contract verifier. It does not number slots or prove
pointers. Jonathan's verifier and attribute comments were updated only for this
approved contract; no attribute fields or enum values changed.

Each slice is evaluated at its parent coordinates. Every lane must prove
`argument + first_offset + row * parent_width + column`; tile_origin is
`(first_offset / parent_width, first_offset % parent_width)`. Thus parent pointer
and slice offsets both contribute. A pointer argument must use a consistent
host row width throughout the kernel, preventing accidental reuse across
incompatible views. Other strides, dynamic accesses, masked memory, volatile
loads, nested memory/control flow, pointer casts and FP8 stores fail explicitly.

Inputs use deterministic memory/slice source order, independent of SSA use-list
order. Identical logical tiles reuse a slot only with matching role, original
host dtype and logical shape. Conflicting representations and repeated output
writes are rejected. Output indices form a separate dense space. No loads may
follow stores. Attributes are published only after successful module analysis.

Role inference follows TTIR dot and upstream `linalg.matmul` operands, float
casts, tensor casts, TTIR transposes and `linalg.transpose`. A direct B operand
is KN; a transposed B source is NK. Transposed activations and ambiguous roles
fail. BF16 addition/generic-add inputs are biases; other BF16 inputs are
constants, or masks when tagged with neg_inf padding.

The original host dtype is copied verbatim from the parent memory operation.
Full tiles inside a parent's unpadded logical extent retain shape [32,32].
Partial padded tiles and 1-D bias packing remain outside M1.

## Pass 7: boundary conversion

The existing boundary hook creates an argument-free func.func returning
`!atlas.virtual_state`. Inputs are emitted by slot index; slices of a reused
input map to the same virtual value. Outputs are emitted in original store and
assembly order, threading the state token linearly. The original SlotAttr stays
on virtual I/O for metadata generation.

Whole-tensor loads/stores, boundary slices, output assembly empties and proven
address scaffolding are removed. Escaping pointer/address values, unrelated
slices and nested boundary memory fail before rewriting. Linalg compute regions
are cloned for the dot and elementwise patterns. The conversion driver now
requires full conversion and an argument-free `func.func` returning
`!atlas.virtual_state`. Current boundary code accepts only complete 32x32 tiles;
partial tiles and 1-D bias broadcasts need Pass 6 packing support. The pass 7
fixtures and stream verifier have not been run on BWRC.

## Kernel metadata v1

The schema is packaged at
`python/triton_atlas_backend/schemas/ttatlas.kernel.v1.schema.json`; the Stage A
golden example is `docs/examples/ttatlas.kernel.v1.json`.

Fields are version, name, dram_base {input,output}, inputs, outputs, grid and
legalized. Slot entries include index, arg, role, tile_origin, logical_shape,
host_dtype and layout. Decisions for v1/M1:

- Origins use canonical MLIR affine-map strings with six symbols and constant,
  nonnegative results. Host dtypes use their original MLIR scalar spelling.
- Layout uses the frozen enum spelling; the default is row_major.
- Input/output tables are sorted by dense index. JSON key order is fixed by
  the producer. JSON integer addresses use 2048-byte slots; default input base
  is 0x90000000, output base immediately follows allocated input slots.
- Grid is [1,1,1]. Legalization records preserve {pass,what,loc} and source order;
  absent history becomes []. Existing manifests must match every generated
  field, including original dtypes, origins, layout and history.
- Empty input/output tables are rejected: atlas-mlir reserves at least one
  slot per window, so zero-input placement needs a separate ABI decision.

AtlasBoundaryABI validates aligned, disjoint selected-DRAM windows and 32-bit
address bounds. The Python exporter validates the packaged JSON Schema plus
index/address relationships before copying the exact JSON string into Triton's
cached metadata. The cache field is `ttatlas_kernel`, because Triton constructs
a namedtuple and cannot accept dotted metadata keys. MLIR retains `ttatlas.kernel`.
Existing name/shared and unrelated cache fields are preserved appropriately.

AtlasOptions includes precision_policy, layout and dram_base, all covered by
its cache hash. Since Triton's Python bindings expose no i64/custom-enum
attribute builder, placement is handed to our producer as the private string
attribute `atlas_backend.options`, then normalized to the frozen typed module
attributes. Explicit module settings must agree. The precision policy uses the
existing shared string tag. The backend hash also includes schema contents.

## Upstream integration limits

The feature branch contains latest fetched main `5ecf687`, plus preserved
history from the interrupted merge of `e909d2b`. Passes 1-5 and compute/type
conversion still contain upstream placeholders. The real Stage A TTIR fixture
first fails because Pass 1 supplies no host_dtype tags. No teammate's compute
implementation, shared driver, Merlin or model2MLIR was changed.

The standalone tool registers Tensor and Linalg, while the Python plugin's
`registerAtlasDialects` currently registers neither. Native Python mixed-IR
parsing therefore fails on `tensor.extract_slice`; this needs a coordinated
plugin registration/linking change alongside the upstream compute work. The
Python options/export test exercises direct TTIR tiles; the standalone tool
exercises the mixed Tensor boundary path. This lane leaves plugin.cpp intact.

## Validation (November 6, 2026)

- `make build.plugin build.tools`: passed, with UV_CACHE_DIR under /tmp.
- Wheel build: passed; the archive includes the version-1 JSON schema.
- Assign-slots: 29/29 lit tests, including five Python slice-contract tests.
- Boundary: 7/7 lit tests; kernel metadata: 6/6.
- Compiler export: 9/9 Python tests (one lit wrapper).
- Contract verifier: 8/8 lit tests, including Stages 6 and 7.
- Selected lane suite: 45/45 passed.
- Full repository lit suite: 61 passed, one expected failure, one failure.
  The sole failure is pipeline/stage-a.mlir, blocked by missing host_dtype
  tags from upstream Pass 1. The target Stage A test remains an expected failure.
- Ruff, clang-format verification, git diff --check, merge-marker scan and
  unmerged-index checks passed. Lit used Python's fork start method because
  the sandbox prohibits the forkserver socket.

The real matmul+ReLU TTIR fixture was separately attempted through the full
pipeline, Stage 7 contract and virtual stream verifier; it fails at the same
missing Pass 1 tags. The mixed two-tile copy succeeds through the standalone
native pipeline and final verifiers. No compute/shape/dtype implementation was
substituted to conceal the Stage A blocker. Logs, test JSON and wheel artifacts
are in /tmp/triton-atlas-resume-backup/.
