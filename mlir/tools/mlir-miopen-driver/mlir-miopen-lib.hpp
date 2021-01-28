#include <cstddef>

namespace mlir {

typedef void *MlirHandle;

extern "C" MlirHandle CreateMlirHandle(const char *options);

extern "C" void MlirLowerCpp(MlirHandle handle);

extern "C" const char *MlirGenIgemmSource(MlirHandle handle);

extern "C" const char *MlirGenIgemmHeader(MlirHandle handle);

extern "C" const char *MlirGenIgemmCflags(MlirHandle handle);

extern "C" void MlirLowerBin(MlirHandle handle);

extern "C" void MlirGenIgemmBin(MlirHandle mlirHandle, char **buffer,
                                size_t *size);

extern "C" void DestroyMlirHandle(MlirHandle handle);

} // namespace mlir
