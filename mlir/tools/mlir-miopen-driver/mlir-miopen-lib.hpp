#include <cstddef>

namespace mlir {

/*! @brief The MLIR handle used for lowering and code generation
 */
typedef void *MlirHandle;

/*! @brief Create the MLIR handle according to options string
 *  @param options A comma seperated string
 *  @return        MLIR handle
 */
extern "C" MlirHandle CreateMlirHandle(const char *options);

/*! @brief Lower the MLIR module to c++ code * @param handle MLIR handle
 */
extern "C" void MlirLowerCpp(MlirHandle handle);

/*! @brief Populate Conv2d implicitgemm host code for MIOpen
 *  @param handle   MLIR handle
 *  @return         Source string
 */
extern "C" const char *MlirGenIgemmSource(MlirHandle handle);

/*! @brief Populate Conv2d implicitgemm header code for MIOpen
 *  @param handle   MLIR handle
 *  @return         Header string
 */
extern "C" const char *MlirGenIgemmHeader(MlirHandle handle);

/*! @brief Populate Conv2d implicitgemm compilation flags for MIOpen
 *  @param handle   MLIR handle
 *  @return         Compilation flags string
 */
extern "C" const char *MlirGenIgemmCflags(MlirHandle handle);

/*! @brief Lower the MLIR module to binary code * @param handle MLIR handle
 */
extern "C" void MlirLowerBin(MlirHandle handle);

/*! @brief Populate Conv2d implicitgemm hsaco code object
 *  @param handle MLIR handle
 *  @param buffer Binary buffer holds hsaco code
 *  @param size Size of the binary buffer
 */
extern "C" void MlirGenIgemmBin(MlirHandle mlirHandle, char **buffer,
                                size_t *size);

/*! @brief Destroy MLIR handle * @param handle MLIR handle
 */
extern "C" void DestroyMlirHandle(MlirHandle handle);

} // namespace mlir
