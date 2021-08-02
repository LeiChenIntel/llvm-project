//
// Created by leichen1 on 2021/8/2.
//
#include <iostream>
#include "MLIRGen.h"
#include "Parser.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

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

int main(int argc, char **argv) {
  std::cout << "Standalone pipeline" << std::endl;
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::standalone::StandaloneDialect>();

  if(!llvm::StringRef(argv[1]).endswith(".mlir")) {
    auto moduleAST = parseInputFile(argv[1]);
    if (!moduleAST)
      return 6;
    mlir::OwningModuleRef module = mlirGen(context, *moduleAST);
    if (!module)
      return 1;

    module->dump();
    return 0;
  }

  return 0;
}