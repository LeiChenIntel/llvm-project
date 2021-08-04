add_mlir_dialect(StandaloneOps standalone)
add_mlir_doc(StandaloneDialect -gen-dialect-doc StandaloneDialect Standalone/)
add_mlir_doc(StandaloneOps -gen-op-doc StandaloneOps Standalone/)

# Most dialects should use add_mlir_interfaces().
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(StandaloneShapeInferenceInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS StandaloneCombine.td)
mlir_tablegen(StandaloneCombine.inc -gen-rewriters)
add_public_tablegen_target(StandaloneCombineIncGen)
