add_subdirectory(tablegen)

add_mlir_dialect_library(RelalgDialect
    RelalgDialect.cpp
    RelalgOps.cpp
    RelalgTypes.cpp
    ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/src/dialects/relalg

    DEPENDS
        MLIRArrowOpsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
    )
