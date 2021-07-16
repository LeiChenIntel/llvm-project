set(LLVM_LINK_COMPONENTS
    Support
    Core
    nativecodegen
    OrcJIT
    )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

include_directories(include/)
#include_directories(/home/leichen1/test/llvm-project/mlir/examples/standalone/build/include/Standalone)
#message("${CMAKE_CURRENT_BINARY_DIR}/..")
add_llvm_executable(standalone-pipeline
    standalone-pipeline.cpp
    parser/AST.cpp
    mlir/MLIRGen.cpp
    )
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
        MLIRStandalone
        )
