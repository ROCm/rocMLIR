#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/Tuning/SqliteDb.h"

#define DEBUG_TYPE "miopen-tuning-parameter"

template <class Solver>
std::tuple<llvm::StringMap<int64_t>, LogicalResult>
find_solution(Solver solver, const ConvolutionContext &ctx) {
  using PerformanceConfig = decltype(solver.GetPerformanceConfig(ctx));
  PerformanceConfig config{};
  LogicalResult valid = failure();

#if __MLIR_ENABLE_SQLITE__
  SQLitePerfDb perfDb = getDb(ctx.arch, ctx.num_cu);
  // auto idx = name.find_last_of(':');
  // auto solverId = name.substr(idx + 1);
  std::string solverId = solver.getId();
  LLVM_DEBUG(llvm::dbgs() << "solverId=" << solverId << "\n");
  bool loadRes = perfDb.load(ctx, solverId, config);
  if (loadRes) {
    if (succeeded(solver.IsValidPerformanceConfig(ctx, config))) {
      LLVM_DEBUG(llvm::dbgs() << "DB load succeed,"
                              << " M/block: " << config.GemmMPerBlock
                              << " N/block: " << config.GemmNPerBlock
                              << " K/block: " << config.GemmKPerBlock << "\n");
      return std::make_tuple(solver.GetSolution(ctx, config), success());
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "DB load failed, falling back to backup path.\n");
#endif // MLIR_ENABLE_SQLITE

  config = solver.GetPerformanceConfig(ctx);
  llvm::StringMap<int64_t> results = solver.GetSolution(ctx, config);
  valid = solver.IsValidPerformanceConfig(ctx, config);
  return std::tie(results, valid);
}

template <class F, class... Ts> void each_args(F f, Ts &&... xs) {
  (void)std::initializer_list<int>{(f(std::forward<Ts>(xs)), 0)...};
}

template <class... Solvers>
std::tuple<llvm::StringMap<int64_t>, LogicalResult>
SolverContainer<Solvers...>::SearchForConfigParameters(
    const ConvolutionContext &ctx) {
  llvm::StringMap<int64_t> parameters;
  LogicalResult valid = failure();
  each_args(
      [&](auto solver) {
        if (succeeded(solver.IsApplicable(ctx))) {
          std::tie(parameters, valid) = find_solution(solver, ctx);
          return;
        }
      },
      Solvers{}...);
  return std::make_tuple(parameters, valid);
}

std::tuple<llvm::StringMap<int64_t>, LogicalResult>
GetConfigParameters(const ConvolutionContext &ctx) {

  SolverContainer<
      ConvHipImplicitGemmV4R4Fwd, ConvHipImplicitGemmBwdDataV1R1,
      ConvHipImplicitGemmV4R4WrW, ConvHipImplicitGemmForwardV4R4Xdlops,
      ConvHipImplicitGemmBwdDataV4R1Xdlops, ConvHipImplicitGemmWrwV4R4Xdlops>
      dummyObj;

  return dummyObj.SearchForConfigParameters(ctx);
}
