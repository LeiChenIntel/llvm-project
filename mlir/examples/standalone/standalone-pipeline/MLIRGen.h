
#ifndef MLIR_TUTORIAL_TOY_MLIRGEN_H_
#define MLIR_TUTORIAL_TOY_MLIRGEN_H_

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace standalone {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
} // namespace standalone

#endif // MLIR_TUTORIAL_TOY_MLIRGEN_H_
