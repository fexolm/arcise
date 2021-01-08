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
        
        ArrowDialect
        )

add_llvm_executable(arcise-printer main.cpp)
llvm_update_compile_flags(arcise-printer)
target_link_libraries(arcise-printer PRIVATE ${LIBS})

mlir_check_all_link_libraries(arcise-printer)