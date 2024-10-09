#ifndef MLIR_DIALECT_ROCK_UTILITY_MEMORYUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_MEMORYUTILS_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace rock {

struct LDSInfo {
  llvm::SmallDenseMap<GpuAllocOp, llvm::SetVector<GpuAllocOp>>
      interferenceGraph;
  SmallVector<GpuAllocOp> allocs;
  SmallVector<GpuDeallocOp> deallocs;
  llvm::SmallDenseMap<GpuAllocOp, llvm::SetVector<GpuAllocOp>> deallocBefore;
};

/// Utility function to get workgroup memory size
std::optional<int64_t> getWorkgroupMemorySize(MemRefType type);

/// Utility function to check if there is enough LDS on the target architecture
LogicalResult checkLDSSize(Operation *op, int64_t ldsBytes);

/// This is a greedy graph coloring algorithm.
/// There are some changes to make it work for LDS, the main one:
/// each alloc can be assigned more than one color, this is because
/// in graph coloring all vertex are assumed to be the same size
/// (for example, register allocation).
/// Example: A=GpuAllocOp(1kB), B=GpuAllocOp(1kB), C=GpuAllocOp(2kB)
///           A <--> B     C (A and B have an edge, C disjoint)
/// In this case, we can assign colors: A -> {0}, B -> {1}, and C -> {0, 1}.
/// Colors 0 and 1 are 1kB each.
/// Note: If an alloc has more than one color assigned, they have to be
///       consecutive.
std::tuple<llvm::MapVector<int64_t, int64_t>,
           SmallVector<std::tuple<GpuAllocOp, int64_t, int64_t, bool>>>
graphColoring(LDSInfo &ldsInfo);

/// Utility function to create an interference graph of GPUAllocs and
/// GPUDeallocs
FailureOr<LDSInfo> createInterferenceGraph(func::FuncOp &func);

/// Utility function to compute allocated LDS after LDS reuse pass.
FailureOr<int64_t> getAllocatedLDSAfterReuse(func::FuncOp &func);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_MEMORYUTILS_H
