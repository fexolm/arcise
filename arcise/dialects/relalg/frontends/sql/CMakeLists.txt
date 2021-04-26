get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIR
        MLIRTransforms
        
        RelalgDialect
        )

add_library(RelalgSQLFrontend AST.cpp Parser.cpp SQLToMLIRConverter.cpp)

add_executable(main main.cpp)


target_link_libraries(RelalgSQLFrontend PRIVATE ${LIBS})
target_compile_options(RelalgSQLFrontend PUBLIC -fno-rtti)

target_link_libraries(main PRIVATE RelalgSQLFrontend ${LIBS})
target_compile_options(main PUBLIC -fno-rtti)


mlir_check_all_link_libraries(RelalgSQLFrontend)
mlir_check_all_link_libraries(main)