add_subdirectory(tablegen)

add_mlir_dialect_library(RelalgDialect
    RelalgOps.cpp
    RelalgDialect.cpp
    RelalgTypes.cpp
    
    ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/arcise/dialects/relalg

    DEPENDS
        MLIRRelalgOpsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
    )

target_compile_options(RelalgDialect PUBLIC -fno-rtti)
