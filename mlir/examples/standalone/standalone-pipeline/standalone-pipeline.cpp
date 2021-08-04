//
// Created by leichen1 on 2021/8/2.
//
#include "MLIRGen.h"
#include "Parser.h"
#include <iostream>

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

  if (mlir::failed(pm.run(*module)))
    return 4;

  module->dump();
  std::cout << "loadAndProcessMLIR dump end" << std::endl;
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

  return 0;
}