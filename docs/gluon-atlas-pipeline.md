# Gluon → Atlas integration contract

## Purpose and boundary

Gluon is Triton's experimental frontend for expressing layout-aware tensor
programs.  Its semantic operations are principally ordinary TTIR (`tt` and
`arith`) operations whose tensor types carry Gluon or TritonGPU layout
encodings; `gluon.set_auto_layout` is the explicit boundary where an
`#gluon.auto_encoding` value receives a concrete distributed layout.

Atlas must not run Triton's upstream `gluon_to_ttgir` pipeline.  That route
resolves Gluon layouts into GPU-specific TTGPU IR, which has thread/warp and
communication semantics Atlas does not implement.  Instead, Atlas accepts the
narrow, statically provable subset of GLIR below, removes frontend layout
metadata, and then runs the existing TTIR-to-Atlas implementation unchanged.

```text
Gluon AST
  → GLIR (`tt`, `arith`, `gluon.set_auto_layout`, encoded tensor types)
  → G0: ttatlas-validate-gluon
  → G1: ttatlas-lower-gluon-layouts-to-ttir
  → P1 validate TTIR → P2 normalize → P3 TTIR compute to Linalg
  → P5 dtype legalization → P4a–P4d packed tiling
  → P6 slot assignment → P7 Atlas virtual IR
```

This is deliberately different from the retired proposal that introduced a
G2 direct GLIR-to-Linalg bridge.  G1 emits layout-free TTIR, not Linalg.  That
keeps exactly one owner for TTIR normalization, compute lowering, dtype policy,
packing, slots, and Atlas instruction emission.

## G0 — validation

`ttatlas-validate-gluon` is non-mutating.  It establishes that a GLIR module
can be normalized without pretending that GPU layout semantics are Atlas
semantics.

For every `gluon.set_auto_layout`, G0 requires:

* ranked source and result tensors;
* static rank one or two shapes;
* identical logical shape and element type;
* an `#gluon.auto_encoding` source; and
* a concrete result encoding.

G0 rejects all `ttg.*` operations (the program has already entered TTGPU) and
any unknown `gluon.*` operation.  Diagnostics use the common Atlas unsupported
form and identify both the source operation and Gluon-specific reason.  It
does not reject `#ttg.*` *type attributes*: concrete Gluon layout anchors use
those attributes and G1 removes them.

The initial accepted subset remains intentionally narrow: static layout
anchors and TTIR operations that P1 subsequently accepts.  Shared memory,
barriers, async copies, GPU communication, descriptor operations, dynamic
control flow, and unsupported memory semantics must fail rather than being
silently reinterpreted.  P1 is currently the authoritative inventory for
ordinary TTIR operations; extending G0 with duplicate operation rules is only
appropriate when it improves a Gluon-specific diagnostic.

## G1 — layout erasure

`ttatlas-lower-gluon-layouts-to-ttir` is a dialect conversion pass with a
single type rule:

```text
tensor<SxT, encoding>  →  tensor<SxT>
```

It erases an encoding only; it never changes rank, dimensions, element type,
pointer type, operation attributes, or memory effects.  `gluon.set_auto_layout`
is replaced with its source because G0 proved it is logically type-preserving.
All region-free operations are rebuilt with converted operands and result
types.  Function argument/result types and entry-block arguments are converted
together, preventing encoded values from leaking across a function boundary.

G1's postcondition is mechanical and testable: no result or operand tensor has
an encoding and no `gluon.set_auto_layout` remains. It rewrites block arguments
inside regions in place, so reductions, loops, branches, and their terminators
retain their original structure and SSA value flow. The following pass (P1)
therefore receives ordinary TTIR and cannot accidentally consume a GPU layout.

## Shared passes P1–P7

Once G1 succeeds, Gluon has no special path.  P1 validates supported TTIR; P2
normalizes dots and loops; P3 creates Linalg; P5 selects Atlas compute dtypes;
P4a–P4d pack and tile values; P6 assigns DRAM slots; and P7 emits the ordered
Atlas virtual stream and schema-v2 manifest.  No G0/G1 pass may create Atlas
operations, choose a dtype, pack data, number a slot, or invent a host ABI.

This separation is critical: a mismatch after G1 is a shared-pipeline issue
and must be reproducible with equivalent Triton TTIR.

## Python backend integration

`AtlasBackend.add_stages` registers `Language.GLUON` as:

```text
glir (identity, preserved in artifacts) → atlas_virtual (G0 + G1 + P1–P7)
```

Gluon's AST creator calls `backend.get_target_name(options)` before it emits
GLIR.  Atlas supplies the stable offline target name `atlas:<arch>`; without
this hook compilation fails before G0.  The backend selects the dedicated
plugin pipeline only for `Language.GLUON`; normal Triton continues to use the
existing pipeline.

## Verification gates

1. `test/validate-gluon/anchors.mlir` checks a real AutoLayout anchor is
   accepted by G0 and erased by G1.
2. `test/validate-gluon/region-reduce.mlir` checks a region-bearing
   `tt.reduce`, its block arguments, and its terminator survive G1 with all
   tensor encodings removed.
3. `test/validate-gluon/reject-rank3-anchor.mlir` checks G0 rejects an
   out-of-scope rank-three anchor with an actionable diagnostic.
4. `test/from-jit/test_gluon_from_jit.py` creates GLIR through the actual
   Gluon Python frontend, includes `set_auto_layout`, and compiles it all the
   way to an Atlas schema-v2 manifest.
4. The complete repository suite remains the final integration gate.  A pass
   is not considered complete merely because its isolated fixture passes.

## Remaining supported-work checklist

* Add real frontend cases for rank-two elementwise work, reductions, masks,
  and matmul only as each reaches the shared P1–P7 contracts.
* Add G0 diagnostics for Gluon-specific unsupported effects where P1's generic
  TTIR diagnostic is insufficient.
* Preserve the front-door test and run the full lit suite before merge.
