# Enable errors for any global constructors.
add_flag_if_supported("-Werror=global-constructors" WERROR_GLOBAL_CONSTRUCTOR)

add_subdirectory(Analysis)
#add_subdirectory(Bindings)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
#add_subdirectory(EDSC)
add_subdirectory(ExecutionEngine)

# Now that both `Dialect/Rock/IR` (which declares MLIRRockOps) and
# `ExecutionEngine` (which optionally declares the `mlir_rocm_arch_runtime`
# target) have been processed, wire up the build-time dependency. The runtime
# is loaded at run time via dlopen by AmdArchDb.cpp, but we want it present
# next to the consumer in the build tree.
if (TARGET MLIRRockOps AND TARGET mlir_rocm_arch_runtime)
  add_dependencies(MLIRRockOps mlir_rocm_arch_runtime)
endif()
#add_subdirectory(IR)
#add_subdirectory(Interfaces)
#add_subdirectory(Parser)
#add_subdirectory(Pass)
#add_subdirectory(Reducer)
#add_subdirectory(Rewrite)
#add_subdirectory(Support)
#add_subdirectory(TableGen)
add_subdirectory(Target)
#add_subdirectory(Transforms)
add_subdirectory(Translation)
