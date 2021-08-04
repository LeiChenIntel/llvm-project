add_mlir_dialect_library(MLIRStandalone
		StandaloneDialect.cpp
		StandaloneOps.cpp
		ShapeInferencePass.cpp

		ADDITIONAL_HEADER_DIRS
		${PROJECT_SOURCE_DIR}/include/Standalone

		DEPENDS
		MLIRStandaloneOpsIncGen
		StandaloneShapeInferenceInterfaceIncGen

		LINK_LIBS PUBLIC
		MLIRIR
		)
