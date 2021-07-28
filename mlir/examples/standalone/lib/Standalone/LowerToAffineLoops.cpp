//
// Created by leichen1 on 2021/7/27.
//

#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
    assert(type.hasRank() && "expected only ranked shapes");
    return MemRefType::get(type.getShape(), type.getElementType());
}

// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
    auto alloc = rewriter.create<memref::AllocOp>(loc, type); // Need "mlir/Dialect/MemRef/IR/MemRef.h"

    // Make sure to allocate at the beginning of the block.
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    // Make sure to deallocate this alloc at the end of the block. This is fine
    // as toy functions have no control flow.
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc); // "mlir/Dialect/MemRef/IR/MemRef.h"
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}


/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
        OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Create a nest of affine loops, with one loop per dimension of the shape.
    // The buildAffineLoopNest function takes a callback that is used to construct
    // the body of the innermost loop given a builder, a location and a range of
    // loop induction variables.
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
    buildAffineLoopNest(
            rewriter, loc, lowerBounds, tensorType.getShape(), steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
                // Call the processing function with the rewriter, the memref operands,
                // and the loop induction variables. This function will return the value
                // to store at the current index.
                Value valueToStore = processIteration(nestedBuilder, operands, ivs);
                nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
            });

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
}

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
struct StandaloneToAffineLoweringPass
        : public PassWrapper<StandaloneToAffineLoweringPass, FunctionPass> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect>(); // TAG: Register dialect and operation here
    }
    void runOnFunction() final;
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
    BinaryOpLowering(MLIRContext *ctx)
            : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(
                op, operands, rewriter,
                [loc](OpBuilder &builder, ValueRange memRefOperands,
                      ValueRange loopIvs) {
                    // Generate an adaptor for the remapped operands of the BinaryOp. This
                    // allows for using the nice named accessors that are generated by the
                    // ODS.
                    typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                    // Generate loads for the element of 'lhs' and 'rhs' at the inner
                    // loop.
                    auto loadedLhs =
                            builder.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
                    auto loadedRhs =
                            builder.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

                    // Create the binary operation performed on the loaded values.
                    return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
                });
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<standalone::ConstantOp> {
    using OpRewritePattern<standalone::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(standalone::ConstantOp op,
                                  PatternRewriter &rewriter) const final {
        DenseElementsAttr constantValue = op.value();
        Location loc = op.getLoc();

        // When lowering the constant operation, we allocate and assign the constant
        // values to a corresponding memref allocation.
        auto tensorType = op.getType().cast<TensorType>();
        auto memRefType = convertTensorToMemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // We will be generating constant indices up-to the largest dimension.
        // Create these constants up-front to avoid large amounts of redundant
        // operations.
        auto valueShape = memRefType.getShape();
        SmallVector<Value, 8> constantIndices;

        if (!valueShape.empty()) {
            for (auto i : llvm::seq<int64_t>(
                    0, *std::max_element(valueShape.begin(), valueShape.end())))
                constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
        } else {
            // This is the case of a tensor of rank 0.
            constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
        }

        // The constant operation represents a multi-dimensional constant, so we
        // will need to generate a store for each of the elements. The following
        // functor recursively walks the dimensions of the constant shape,
        // generating a store when the recursion hits the base case.
        SmallVector<Value, 2> indices;
        auto valueIt = constantValue.getValues<FloatAttr>().begin();
        std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
            // The last dimension is the base case of the recursion, at this point
            // we store the element at the given index.
            if (dimension == valueShape.size()) {
                rewriter.create<AffineStoreOp>(
                        loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc, // The ConstantOp is a mlir ConstantOp
                        llvm::makeArrayRef(indices));
                return;
            }

            // Otherwise, iterate over the current dimension and add the indices to
            // the list.
            for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
                indices.push_back(constantIndices[i]);
                storeElements(dimension + 1);
                indices.pop_back();
            }
        };

        // Start the element storing recursion from the first dimension.
        storeElements(/*dimension=*/0);

        // Replace this operation with the generated alloc.
        rewriter.replaceOp(op, alloc);
        return success();
    }
};


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//
struct ReturnOpLowering : public OpRewritePattern<standalone::ReturnOp> {
    using OpRewritePattern<standalone::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(standalone::ReturnOp op,
                                  PatternRewriter &rewriter) const final {
        // During this lowering, we expect that all function calls have been
        // inlined.
        if (op.hasOperand())
            return failure();

        // We lower "toy.return" directly to "std.return".
        rewriter.replaceOpWithNewOp<ReturnOp>(op); // The ReturnOp is a mlir ReturnOp
        return success();
    }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
    TransposeOpLowering(MLIRContext *ctx)
            : ConversionPattern(standalone::TransposeOp::getOperationName(), 1, ctx) {}

    LogicalResult
    matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](OpBuilder &builder, ValueRange memRefOperands,
                             ValueRange loopIvs) {
                           // Generate an adaptor for the remapped operands of the
                           // TransposeOp. This allows for using the nice named
                           // accessors that are generated by the ODS.
                           standalone::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                           Value input = transposeAdaptor.input();

                           // Transpose the elements by generating a load from the
                           // reverse indices.
                           SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                           return builder.create<AffineLoadOp>(loc, input,
                                                               reverseIvs);
                       });
        return success();
    }
};

void StandaloneToAffineLoweringPass::runOnFunction() {
    auto function = getFunction();

    // We only lower the main function as we expect that all other functions have
    // been inlined.
    if (function.getName() != "main")
        return;

    // Verify that the given main has no inputs and results.
    if (function.getNumArguments() || function.getType().getNumResults()) {
        function.emitError("expected 'main' to have 0 inputs and 0 results");
        return signalPassFailure();
    }

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine` and `Standard` dialects.
    // TAG: Register dialect and operation here
    target.addLegalDialect<AffineDialect, memref::MemRefDialect, StandardOpsDialect>();

    // We also define the Toy dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Toy operations that don't want
    // to lower, `toy.print`, as `legal`.
    target.addIllegalDialect<standalone::StandaloneDialect>();
    target.addLegalOp<standalone::PrintOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.insert<BinaryOpLowering<standalone::AddOp, AddFOp>, ConstantOpLowering, BinaryOpLowering<standalone::MulOp, MulFOp>,
            ReturnOpLowering, TransposeOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::standalone::createLowerToAffinePass() {
    return std::make_unique<StandaloneToAffineLoweringPass>();
}