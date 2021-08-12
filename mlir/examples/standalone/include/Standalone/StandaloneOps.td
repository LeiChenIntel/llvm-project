//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "StandaloneDialect.td"
include "ShapeInferenceInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Standalone_FooOp : Standalone_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def ConstantOp : Standalone_Op<"constant", [NoSideEffect]> {
    // Provide a summary and description for this operation. This can be used to
    // auto-generate documentation of the operations within our dialect.
    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        ```mlir
        %0 = standalone.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
        : tensor<2x3xf64>
        ```
    }];

    // The constant operation takes an attribute as the only input.
    let arguments = (ins F64ElementsAttr:$value);

    // The constant operation returns a single value of TensorType.
    let results = (outs F64Tensor);

    // Specify a parser and printer method.
    let parser = [{ return ::parseConstantOp(parser, result); }];
    let printer = [{ return ::print(p, *this); }];

    // Add custom build methods for the constant operation. These method populates
    // the `state` that MLIR uses to create operations, i.e. these are used when
    // using `builder.create<ConstantOp>(...)`.
    let builders = [
        // Build a constant with a given constant tensor value.
        OpBuilderDAG<(ins "DenseElementsAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,

        // Build a constant with a given constant floating-point value.
        OpBuilderDAG<(ins "double":$value)>
    ];

    // Invoke a static verify method to verify this constant operation.
    let verifier = [{ return ::verify(*this); }];
}

def AddOp : Standalone_Op<"add", [
NoSideEffect,
DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two tensors.
        The shapes of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    // Specify a parser and printer method.
    let parser = [{ return ::parseBinaryOp(parser, result); }];
    let printer = [{ return ::printBinaryOp(p, *this); }];

    // Allow building an AddOp with from the two input operands.
    let builders = [
        OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def CastOp : Standalone_Op<"cast", [
DeclareOpInterfaceMethods<CastOpInterface>,
DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
NoSideEffect,
SameOperandsAndResultShape
]> {
    let summary = "shape cast operation";
    let description = [{
        The "cast" operation converts a tensor from one type to an equivalent type
                without changing any data elements. The source and destination types must
                both be tensor types with the same element type. If both are ranked, then
        shape is required to match. The operation is invalid if converting to a
                mismatching constant dimension.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$output);

    let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def GenericCallOp : Standalone_Op<"generic_call", [
DeclareOpInterfaceMethods<CallOpInterface>
]> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that needs to
        be specialized for the shape of its arguments. The callee name is attached
                as a symbol reference via an attribute. The arguments list must match the
        arguments expected by the callee. For example:

        ```mlir
        %4 = standalone.generic_call @my_func(%1, %3)
        : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
        ```

        This is only valid if a function named "my_func" exists and takes two
                arguments.
    }];

    // The generic call operation takes a symbol reference attribute as the
    // callee, and inputs for the call.
    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

    // The generic call operation returns a single value of TensorType.
    let results = (outs F64Tensor);

    // Specialize assembly printing and parsing using a declarative format.
    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    // Add custom build methods for the generic call operation.
    let builders = [
        OpBuilderDAG<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
    ];
}

def MulOp : Standalone_Op<"mul", [
NoSideEffect,
DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
    let summary = "element-wise multiplication operation";
    let description = [{
        The "mul" operation performs element-wise multiplication between two
        tensors. The shapes of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    // Specify a parser and printer method.
    let parser = [{ return ::parseBinaryOp(parser, result); }];
    let printer = [{ return ::printBinaryOp(p, *this); }];

    // Allow building a MulOp with from the two input operands.
    let builders = [
        OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def PrintOp : Standalone_Op<"print"> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
                no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input); // TAG: Add F64MemRef input

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReshapeOp : Standalone_Op<"reshape"> {
    let summary = "tensor reshape operation";
    let description = [{
        Reshape operation is transforming its input tensor into a new tensor with
        the same number of elements but different shapes. For example:

        ```mlir
        %0 = standalone.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
        ```
    }];

    let arguments = (ins F64Tensor:$input);

    // We expect that the reshape operation returns a statically shaped tensor.
    let results = (outs StaticShapeTensorOf<[F64]>);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    // Enable registering canonicalization patterns with this operation.
    let hasCanonicalizer = 1;
}

def ReturnOp : Standalone_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
        Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function.
        The operation takes an optional tensor operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation. For example:

        ```mlir
                func @foo() -> tensor<2xf64> {
        ...
        standalone.return %0 : tensor<2xf64>
        }
        ```
    }];

    // The return operation takes an optional input operand to return. This
    // value must match the return type of the enclosing function.
    let arguments = (ins Variadic<F64Tensor>:$input);

    // The return operation only emits the input in the format if it is present.
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

    // Allow building a ReturnOp with no return operand.
    let builders = [
        OpBuilderDAG<(ins), [{ build($_builder, $_state, llvm::None); }]>
    ];

    // Provide extra utility definitions on the c++ operation class definition.
    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    // Invoke a static verify method to verify this return operation.
    let verifier = [{ return ::verify(*this); }];
}

def TransposeOp : Standalone_Op<"transpose", [
NoSideEffect,
DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
    let summary = "transpose operation";

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let hasCanonicalizer = 1;

    // Allow building a TransposeOp with from the input operand.
    let builders = [
        OpBuilderDAG<(ins "Value":$input)>
    ];

    // Invoke a static verify method to verify this transpose operation.
    let verifier = [{ return ::verify(*this); }];
}

def MatmulOp : Standalone_Op<"matmul", [
NoSideEffect,
DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
]> {
    let arguments = (ins F64Tensor:$leftInput, F64Tensor:$rightInput);
    let results = (outs F64Tensor);

    let builders = [
        OpBuilderDAG<(ins "Value":$leftInput, "Value":$rightInput)>
    ];
}

#endif // STANDALONE_OPS
