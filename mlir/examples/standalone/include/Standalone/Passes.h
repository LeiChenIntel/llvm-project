//
// Created by leichen1 on 2021/7/19.
//

#ifndef STANDALONE_DIALECT_PASSES_H
#define STANDALONE_DIALECT_PASSES_H

#include <memory>

namespace mlir {
class Pass;
namespace standalone {
    std::unique_ptr<Pass> createShapeInferencePass();

    std::unique_ptr<mlir::Pass> createLowerToAffinePass();

    //std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
}
}

#endif //STANDALONE_DIALECT_PASSES_H
