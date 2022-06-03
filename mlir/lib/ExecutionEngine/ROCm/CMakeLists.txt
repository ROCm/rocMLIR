set(LLVM_OPTIONAL_SOURCES
  IsaNameParser.cpp
)

set(LLVM_LINK_COMPONENTS Support)

add_llvm_library(LLVMROCmBackendUtils
  IsaNameParser.cpp
)
