// The Atlas hardware facts the passes depend on, in one place, so no pass
// spells a hardware number itself. Changing a fact here is the whole change.
// Logic that depends on the hardware in a way a constant can't hold (which
// MXU to prefer, a data layout, a LUT range) carries an `atlas-hw:` comment
// with its source instead; `git grep -n "atlas-hw:\|target::"` lists every
// place the hardware reaches into the passes.
//
// Sources: atlas-mlir at triton-atlas's pin (third_party/atlas-mlir,
// 8ee175c, its default branch). Where atlas-mlir's own headers already define a
// fact (taken there from the RTL), the constant here is that definition, so the
// two can't drift; static_asserts below flag a change our slot ABI or tags
// depend on. For hardware facts the taped-out RTL wins over any model or doc.
#ifndef TRITON_ATLAS_TTATLAS_ATLASTARGET_H
#define TRITON_ATLAS_TTATLAS_ATLASTARGET_H

#include "Atlas/AtlasMXUOwnership.h"
#include "Atlas/AtlasTarget.h"

#include <cstdint>

namespace mlir::ttatlas::target {

// Not defined as constants in atlas-mlir (only in its docs and code), so they
// live here: the tile size, scale codes, IMEM size and DRAM window.
//
// Every virtual-IR value is one 32x32 tile; a 1-D value (a bias row, a row
// reduction's result) is 32 long (docs/compiler-pipeline.md:109,
// docs/dialect-reference.md:14).
inline constexpr int64_t kTileSize = 32;

// One input or output slot in DRAM, the size of a BF16 tile: address = base +
// index * kSlotBytes (docs/compiler-pipeline.md:34; atlas::kAtlasBF16TileBytes,
// include/Atlas/AtlasTarget.h:22).
inline constexpr int64_t kSlotBytes = mlir::atlas::kAtlasBF16TileBytes;

// Two matmul units, MXU0 and MXU1, each holding at most 2 weights and 2 live
// accumulators per block (atlas::MXUOwnership::kUnits and kSlots,
// include/Atlas/AtlasMXUOwnership.h:18; docs/dialect-reference.md:80,
// docs/hardware-status.md:40).
inline constexpr int kMxuUnits = mlir::atlas::MXUOwnership::kUnits;
inline constexpr int kMxuWeightsPerUnit = mlir::atlas::MXUOwnership::kSlots;
inline constexpr int kMxuAccumulatorsPerUnit =
    mlir::atlas::MXUOwnership::kSlots;

// pack_fp8 scale codes (docs/compiler-pipeline.md:154).
inline constexpr int kScaleCodeMin = 122;
inline constexpr int kScaleCodeMax = 128;

// Instruction memory (docs/host-launch.md:29).
inline constexpr int64_t kImemBytes = 128 * 1024;

// DMA buffers must lie in [kDramWindowBegin, kDramWindowEnd), aligned to one
// 1 KiB tensor register (lib/AtlasBoundaryABI.cpp:36-38, using
// atlas::kAtlasRegisterBytes, include/Atlas/AtlasTarget.h:20);
// kDefaultDramBase is where atlas-mlir's examples place them
// (docs/host-launch.md:34).
inline constexpr int64_t kDramWindowBegin = 0x8000'0000;
inline constexpr int64_t kDramWindowEnd = int64_t(1) << 32;
inline constexpr int64_t kDramAlignment = mlir::atlas::kAtlasRegisterBytes;
inline constexpr int64_t kDefaultDramBase = 0x9000'0000;

static_assert(
    kSlotBytes == 2048 && kDramAlignment == 1024,
    "atlas-mlir's tile or register size changed: recheck the slot ABI");
static_assert(kMxuUnits == 2,
              "atlas-mlir's MXU count changed: recheck ttatlas.mxu and pass 4");

} // namespace mlir::ttatlas::target

#endif // TRITON_ATLAS_TTATLAS_ATLASTARGET_H
