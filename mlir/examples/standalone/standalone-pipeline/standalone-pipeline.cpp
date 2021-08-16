//
// Created by leichen1 on 2021/8/2.
//
#include "MLIRGen.h"
#include "Parser.h"
#include <iostream>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h" // for makeOptimizingTransformer
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h" // llvm::InitializeNativeTarget
#include "llvm/Support/raw_ostream.h"

#include "Standalone/Passes.h"
#include "Standalone/StandaloneDialect.h"

using namespace standalone;

std::unique_ptr<standalone::ModuleAST>
parseInputFile(llvm::StringRef filename) {
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

static char *inputFilename;

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  // TAG: 1. Translate from .standalone to .ast and then to .mlir.
  if (!llvm::StringRef(inputFilename).endswith(".mlir")) {
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

  // TAG: Add inliner pass.
  pm.addPass(mlir::createInlinerPass());

  mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
  // TAG: Add shape inference pass.
  optPM.addPass(mlir::standalone::createShapeInferencePass());
  // TAG: Add canonicalizer pass.
  optPM.addPass(mlir::createCanonicalizerPass());
  // TAG: Add CSE pass.
  optPM.addPass(mlir::createCSEPass());

  // TAG: Lower to Affine.
  optPM.addPass(mlir::standalone::createLowerToAffinePass());
  //optPM.addPass(mlir::createCanonicalizerPass());
  //optPM.addPass(mlir::createCSEPass());
  //optPM.addPass(mlir::createLoopFusionPass());
  //optPM.addPass(mlir::createMemRefDataFlowOptPass());

  // TAG: Lower to LLVM.
  pm.addPass(mlir::standalone::createLowerToLLVMPass());

  if (mlir::failed(pm.run(*module)))
    return 4;

  module->dump();
  std::cout << "loadAndProcessMLIR dump end" << std::endl;
  return 0;
}

int runJit(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // optLevel = 3 makes matmul nan. Set to 0 for debug.
  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }
  return 0;
}

int main(int argc, char **argv) {
  std::cout << "Standalone pipeline" << std::endl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::standalone::StandaloneDialect>();

  inputFilename = argv[1];
  mlir::OwningModuleRef module;
  if (int error = loadAndProcessMLIR(context, module)) {
    return error;
  }

  if (argc == 3) {
    std::cout << "Standalone results:" << std::endl;
    return runJit(*module);
  }
  return 0;
}