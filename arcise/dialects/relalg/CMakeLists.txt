add_subdirectory(tablegen)
add_subdirectory(frontends)

add_mlir_dialect_library(RelalgDialect
    RelalgOps.cpp
    RelalgDialect.cpp
    RelalgTypes.cpp
    passes/LowerToArrowPass.cpp
    
    ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/arcise/dialects/relalg

    DEPENDS
        MLIRRelalgOpsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        ArrowDialect
    )

target_compile_options(RelalgDialect PUBLIC -fno-rtti)
