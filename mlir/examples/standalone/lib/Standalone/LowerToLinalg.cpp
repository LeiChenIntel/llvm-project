
#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"


#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

struct MatmulOpLowering : public OpRewritePattern<standalone::MatmulOp> {
  using OpRewritePattern<standalone::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(standalone::MatmulOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto lhs = op.leftInput();
    auto rhs = op.rightInput();

    auto resultTensorType = op.getType().cast<ShapedType>();
    auto resultElementType = resultTensorType.getElementType();
    auto resultShape = resultTensorType.getShape();

    auto resultMemRefType = MemRefType::get(resultShape, resultElementType);

    auto alloc = rewriter.create<AllocOp>(loc, resultMemRefType);

    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());

    auto newOp = rewriter.create<linalg::MatmulOp>(loc, ValueRange{lhs, rhs},
                                                   ValueRange{alloc});
    rewriter.replaceOp(op, newOp.getOperand(2));

    return success();
  }
};

struct StandaloneToLinalgLoweringPass
        : public PassWrapper<StandaloneToLinalgLoweringPass, FunctionPass> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<standalone::StandaloneDialect, linalg::LinalgDialect>();
    }
    void runOnFunction() final;
};

void StandaloneToLinalgLoweringPass::runOnFunction() {
    auto function = getFunction();

    if (function.getName() != "main")
        return;

    if (function.getNumArguments() || function.getType().getNumResults()) {
        function.emitError("expected 'main' to have 0 inputs and 0 results");
        return signalPassFailure();
    }

    OwningRewritePatternList patterns;
    patterns.insert<MatmulOpLowering>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::standalone::createLowerToLinalgPass() {
    return std::make_unique<StandaloneToLinalgLoweringPass>();
}