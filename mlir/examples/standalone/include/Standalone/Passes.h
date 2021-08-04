
#ifndef STANDALONE_DIALECT_PASSES_H
#define STANDALONE_DIALECT_PASSES_H

#include <memory>

namespace mlir {
class Pass;
namespace standalone {
std::unique_ptr<Pass> createShapeInferencePass();

std::unique_ptr<mlir::Pass> createLowerToAffinePass();

// std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace standalone
} // namespace mlir

#endif // STANDALONE_DIALECT_PASSES_H
