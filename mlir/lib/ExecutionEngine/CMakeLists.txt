set(disable_install "")
if (BUILD_FAT_LIBROCKCOMPILER)
  set(disable_install "DISABLE_INSTALL")
endif()
add_mlir_library(conv-validation-wrappers SHARED
  EXCLUDE_FROM_LIBMLIR
  PARTIAL_SOURCES_INTENDED
  ${disable_install}
  conv-validation-wrappers.cpp
)
set_target_properties(conv-validation-wrappers PROPERTIES EXCLUDE_FROM_ALL ${BUILD_FAT_LIBROCKCOMPILER})

target_link_libraries(conv-validation-wrappers
  PUBLIC
  LLVMSupport
)
if (NOT MSVC)
  # Suppress some compiler warnings when building with clang.  The first two
  # suppressions regard a GNU extension, the third an incomplete-type warning
  # that apparently works out okay.
  set_source_files_properties(conv-validation-wrappers.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-gnu-anonymous-struct;-Wno-nested-anon-types;-Wno-return-type-c-linkage;-Wno-c++98-compat-extra-semi")
endif()

# `mlir_rocm_arch_runtime` is the only target that link-depends on the AMD HIP
# / HSA runtimes. MLIRRockOps loads it lazily via a private `dlmopen` /
# `LoadLibraryW` call (see `mlir/lib/Dialect/Rock/IR/AmdArchDb.cpp`) so that
# binaries embedding rocMLIR's LLVM never link `libamdhip64`, which would
# transitively pull in ROCm's `libLLVM.so` and clobber `cl::opt` global state.
#
# The runtime is platform-specific: on Linux it links HIP+HSA, on Windows it
# links HIP only (HSA is unsupported on Windows ROCm). It is skipped entirely
# when no HIP package is found, in which case MLIRRockOps will simply report
# 0 native devices at run time.
if (NOT DEFINED ROCM_PATH)
  if (NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
  endif()
endif()

find_package(hip QUIET PATHS ${ROCM_PATH})
if (NOT WIN32)
  find_package(hsa-runtime64 1.0 QUIET)
endif()

# Build the runtime when HIP is found and (on Linux) HSA is also found.
# Windows ROCm has no HSA, so HIP alone is enough there.
if (hip_FOUND AND (WIN32 OR hsa-runtime64_FOUND))
  # Use a plain `add_library` (not `add_mlir_library`) so we do not pull in
  # `LLVMSupport`. This library MUST stay free of any rocMLIR-side LLVM
  # symbol so that loading it from a process that already embeds rocMLIR's
  # LLVM is harmless.
  add_library(mlir_rocm_arch_runtime SHARED RocmArchRuntime.cpp)
  set_target_properties(mlir_rocm_arch_runtime PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    # Place the library next to the other rocMLIR shared libraries so the
    # loader's `<exe-dir>/../lib/` search rule (see AmdArchDb.cpp) finds it
    # in both the build tree and the install tree.
    LIBRARY_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
    RUNTIME_OUTPUT_DIRECTORY ${LLVM_RUNTIME_OUTPUT_INTDIR})

  target_include_directories(mlir_rocm_arch_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${PROJECT_BINARY_DIR}/mlir/include)
  target_compile_definitions(mlir_rocm_arch_runtime PRIVATE __HIP_PLATFORM_AMD__)
  target_include_directories(mlir_rocm_arch_runtime SYSTEM PRIVATE
    "${ROCM_PATH}/include")
  target_link_libraries(mlir_rocm_arch_runtime PRIVATE
    hip::host hip::amdhip64)
  if (NOT WIN32)
    target_link_libraries(mlir_rocm_arch_runtime PRIVATE
      hsa-runtime64::hsa-runtime64)
  endif()

  if (NOT MSVC)
    # Suppress HIP header warnings (same set used elsewhere in the tree for
    # HIP consumers; see mlir/tools/rocmlir-tuning-driver/CMakeLists.txt).
    set_source_files_properties(RocmArchRuntime.cpp PROPERTIES
      COMPILE_OPTIONS "-Wno-gnu-anonymous-struct;-Wno-nested-anon-types;-Wno-return-type-c-linkage;-Wno-c++98-compat-extra-semi")
  endif()

  # Install the runtime in both fat and shared builds: downstream consumers
  # of `librockCompiler.a` (MIOpen, MIGraphX) need this `.so`/`.dll` next to
  # their binary if they invoke `--arch native:N`. Bundling it into the
  # static archive would re-introduce the libamdhip64 / libamd_comgr / ROCm
  # libLLVM.so transitive dependency that the split exists to avoid, so we
  # deliberately ignore `${disable_install}` here.
  install(TARGETS mlir_rocm_arch_runtime
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    COMPONENT mlir_rocm_arch_runtime)
endif()

