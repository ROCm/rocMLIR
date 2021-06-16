//===-------------- Miir.h - C API ---------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_MIIR_H
#define MLIR_C_MIIR_H

#include <stddef.h>
#include <stdint.h>

#define MIIR_VERSION_FLAT 5

enum MiirStatus {
  MIIR_SUCCESS = 0,
  MIIR_INVALID_PARAM,
  MIIR_INVALID_MODULE,
  MIIR_BUILD_FAILURE
};
typedef enum MiirStatus MiirStatus;

/*! @brief The MLIR handle used for lowering and code generation
 */
typedef void *MiirHandle;

// A convolution includes three arguments of StridedMemRef to
// represent filter, input and output tensors

/*! @brief Device interface argument type for 2D convolution
 * There is an additional group dimension before channel dimension
 */
struct StridedMemRef5D {
  void *basePtr;
  void *data;
  int32_t offset;
  int32_t sizes[5];
  int32_t strides[5];
};

/*! @brief Create the MLIR handle according to options string
 *  @param options Command-line options as a string
 *  @return        MLIR handle
 */
extern "C" MiirHandle miirCreateHandle(const char *options);

/*! @brief Return the number of kernels required for operation
 *  @param handle  MLIR handle
 *  @return        Kernel count
 */
extern "C" int miirGetKernelCount(MiirHandle handle);

/*! @brief Lower the MLIR module to c++ code
 *  @param handle   MLIR handle
 */
extern "C" MiirStatus miirLowerCpp(MiirHandle handle);

/*! @brief Populate Conv2d implicitgemm host code for MIOpen
 *  @param handle   MLIR handle
 *  @return         Source string
 */
extern "C" const char *miirGenIgemmSource(MiirHandle handle);

/*! @brief Populate Conv2d implicitgemm header code for MIOpen
 *  @param handle   MLIR handle
 *  @return         Header string
 */
extern "C" const char *miirGenIgemmHeader(MiirHandle handle);

/*! @brief Populate Conv2d implicitgemm compilation flags for MIOpen
 *  @param handle   MLIR handle
 *  @return         Compilation flags string
 */
extern "C" const char *miirGenIgemmCflags(MiirHandle handle);

/*! @brief Lower the MLIR module to be able to obtain tuning parameters
 *  @param handle MLIR handle
 */
extern "C" MiirStatus miirLowerTuningParams(MiirHandle mlirHandle);

/*! @brief Lower the MLIR module to binary code
 *  @param handle MLIR handle
 */
extern "C" MiirStatus miirLowerBin(MiirHandle handle);

/*! @brief Populate Conv2d implicitgemm hsaco code object
 *         Client is responsible for the buffer allocation
 *         * First call: client invoke the API with buffer param set to nullptr
 *           and this API set the size param only
 *         * Second call: Client passes in the allocated buffer and this API
 *           copies the hsaco into the client allocated buffer
 *  @param handle MLIR handle
 *  @param buffer Binary buffer holds hsaco code
 *  @param size Size of the binary buffer
 */
extern "C" MiirStatus miirBufferGet(MiirHandle handle, char *buffer,
                                    size_t *size);

/*! @brief Get the global and local size for Dispatch
 *  @param handle MLIR handle
 *  @param global_size Pointer to global size storage (1 dimension)
 *  @param local_size Pointer to local size storage (1 dimension)
 */
extern "C" MiirStatus miirGetExecutionDims(MiirHandle handle,
                                           size_t *global_size,
                                           size_t *local_size);

/*! @brief Destroy MLIR handle
 *  @param handle MLIR handle
 */
extern "C" MiirStatus miirDestroyHandle(MiirHandle handle);

#endif // MLIR_C_MIIR_H
