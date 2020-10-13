add_subdirectory(tablegen)

add_mlir_dialect_library(ArrowDialect
    ArrowOps.cpp
    ArrowDialect.cpp
    ArrowTypes.cpp
    passes/LowerToAffinePass.cpp
    passes/SplitColumnarOpsPass.cpp
    ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/src/dialects/arrow

    DEPENDS
        MLIRArrowOpsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
    )