#ifndef TABLEGEN_RELALG_OPS
#define TABLEGEN_RELALG_OPS

include "RelalgDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AnyNumericType : AnyTypeOf<[I8, I16, I32, I64, F32, F64]>;

// TODO

#endif // TABLEGEN_RELALG_OPS