//===- Transforms.h - ArmNeon Transformation Entrypoints --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMNEON_TRANSFORMS_H
#define MLIR_DIALECT_ARMNEON_TRANSFORMS_H

namespace mlir {
class RewritePatternSet;

namespace arm_neon {
<<<<<<< HEAD
void populateLowerContractionToNeonI8MMPatternPatterns(
    RewritePatternSet &patterns);
=======
void populateLowerContractionToNeonI8MMPatterns(RewritePatternSet &patterns);
void populateLowerContractionToNeonBFMMLAPatterns(RewritePatternSet &patterns);
>>>>>>> 9860325438b8f8620553a524caa547ae9733f02a
} // namespace arm_neon

} // namespace mlir

#endif // MLIR_DIALECT_ARMNEON_TRANSFORMS_H
