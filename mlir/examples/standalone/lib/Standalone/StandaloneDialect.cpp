//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::standalone;

//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
struct StandaloneInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// All call operations within toy can be inlined.
    bool isLegalToInline(Operation *call, Operation *callable,
                         bool wouldBeCloned) const final {
        return true;
    }

    /// All operations within toy can be inlined.
    bool isLegalToInline(Operation *, Region *, bool,
                         BlockAndValueMapping &) const final {
        return true;
    }

    //===--------------------------------------------------------------------===//
    // Transformation Hooks
    //===--------------------------------------------------------------------===//

    /// Handle the given inlined terminator(toy.return) by replacing it with a new
    /// operation as necessary.
    void handleTerminator(Operation *op,
                          ArrayRef<Value> valuesToRepl) const final {
        // Only "toy.return" needs to be handled here.
        auto returnOp = cast<ReturnOp>(op);

        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }

    /// Attempts to materialize a conversion for a type mismatch between a call
    /// from this dialect, and a callable region. This method should generate an
    /// operation that takes 'input' as the only operand, and produces a single
    /// result of 'resultType'. If a conversion can not be generated, nullptr
    /// should be returned.

    Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                         Type resultType,
                                         Location conversionLoc) const final {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/StandaloneOps.cpp.inc"
      >();
  addInterfaces<StandaloneInlinerInterface>();
}
