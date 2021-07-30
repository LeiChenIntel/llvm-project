//
// Created by leichen1 on 2021/7/15.
//
#include<iostream>
#include "standalone/MLIRGen.h"
#include "standalone/Parser.h"
#include "Standalone/Passes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h" // for registerLLVMDialectTranslation
#include "mlir/Target/LLVMIR/Export.h" // for translateModuleToLLVMIR
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h" // for makeOptimizingTransformer

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h" // llvm::LLVMContext
#include "llvm/Support/TargetSelect.h" // llvm::InitializeNativeTarget

#include "Standalone/StandaloneDialect.h"

using namespace standalone;

std::unique_ptr<standalone::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    return parser.parseModule();
}

static char* inputFilename;

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
    // TAG: 1. Translate from .standalone to .ast and then to .mlir.
    if(!llvm::StringRef(inputFilename).endswith(".mlir")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST) {
            return 6;
        }
        module = mlirGen(context, *moduleAST);
        if (!module) {
            return 1;
        }
        module->dump();
        std::cout << "loadMLIR dump end" << std::endl;
        return 0;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningModuleRef &module) {
    if (int error = loadMLIR(context, module)) {
        return error;
    }


    // TAG: 2. Add pass.
    mlir::PassManager pm(&context);
    applyPassManagerCLOptions(pm);

    // Pre-processing
    pm.addPass(mlir::createInlinerPass());

    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::standalone::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // Lower to Linalg
    /*
    mlir::OpPassManager &optPM2 = pm.nest<mlir::FuncOp>();
    optPM2.addPass(mlir::standalone::createLowerToLinalgPass());
    optPM2.addPass(mlir::createCanonicalizerPass());
    optPM2.addPass(mlir::createCSEPass());
     */

    // Lower to Affine
    optPM.addPass(mlir::standalone::createLowerToAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createAffineScalarReplacementPass());

    // Lower to LLVM
    pm.addPass(mlir::standalone::createLowerToLLVMPass());

    if (mlir::failed(pm.run(*module)))
        return 4;

    module->dump();
    std::cout << "loadAndProcessMLIR dump end" << std::endl;
    return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
    mlir::registerLLVMDialectTranslation(*module->getContext());
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }
    llvm::errs() << *llvmModule << "\n";
    return 0;
}

int runJit(mlir::ModuleOp module) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    mlir::registerLLVMDialectTranslation(*(module->getContext())); // Translate from "LLVM dialect" to "LLVM IR"

    auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
    auto maybeEngine = mlir::ExecutionEngine::create(module, /*llvmModuleBuilder=*/nullptr, optPipeline);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }
    return 0;
}

int main(int argc, char** argv) {
    std::cout << "Standalone pipeline" << std::endl;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::standalone::StandaloneDialect>();

    std::cout << argv[1] << std::endl;
    inputFilename = argv[1];
    /*
    if(!llvm::StringRef(argv[1]).endswith(".mlir")) {
        std::cout << "" << std::endl;
        auto moduleAST = parseInputFile(argv[1]);
        if (!moduleAST)
            return 6;
        mlir::OwningModuleRef module = mlirGen(context, *moduleAST);
        if (!module)
            return 1;

        module->dump();
        return 0;
    }
    */

    mlir::OwningModuleRef module;
    if (int error = loadAndProcessMLIR(context, module)) {
        return error;
    }

    bool isRunJit = false;

    if (isRunJit) {
        return runJit(*module);
    }
    else {
        return dumpLLVMIR(*module);
    }
}
