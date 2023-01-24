add_mlir_library(conv-validation-wrappers SHARED
  conv-validation-wrappers.cpp
)

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