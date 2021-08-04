
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_llvm_executable(standalone-pipeline
        standalone-pipeline.cpp
        MLIRGen.cpp
        AST.cpp)
target_link_libraries(standalone-pipeline
        PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRTransforms
        MLIRTargetLLVMIR
        MLIRExecutionEngine
        MLIRStandalone)
