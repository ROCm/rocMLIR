#include <cstddef>

#define MLIRMIOPEN_VERSION_FLAT 0

enum {
  EMlirSuccess = 0,
  EMlirInvalidParam = -1,
  EMlirInvalidModule = -2,
  EMlirBuildFailure = -3
};

/*! @brief The MLIR handle used for lowering and code generation
 */
typedef void *MlirHandle;

/*! @brief Create the MLIR handle according to options string
 *  @param options Command-line options as a string
 *  @return        MLIR handle
 */
extern "C" MlirHandle CreateMlirHandle(const char *options);

/*! @brief Lower the MLIR module to c++ code
 *  @param handle   MLIR handle
 */
extern "C" int MlirLowerCpp(MlirHandle handle);

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

/*! @brief Lower the MLIR module to binary code
 *  @param handle MLIR handle
 */
extern "C" int MlirLowerBin(MlirHandle handle);

/*! @brief Populate Conv2d implicitgemm hsaco code object
 *  @param handle MLIR handle
 *  @param buffer Binary buffer holds hsaco code
 *  @param size Size of the binary buffer
 */
extern "C" int MlirGenIgemmBin(MlirHandle handle, char **buffer, size_t *size);

/*! @brief Get the global and local size for Dispatch
 *  @param handle MLIR handle
 *  @param global_size Pointer to global size storage
 *  @param local_size Pointer to local size storage
 */
extern "C" int MlirGetExecutionDims(MlirHandle handle, size_t *global_size,
                                    size_t *local_size);

/*! @brief Destroy MLIR handle
 *  @param handle MLIR handle
 */
extern "C" int DestroyMlirHandle(MlirHandle handle);
