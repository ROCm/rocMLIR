# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

add_subdirectory(ROCm)
