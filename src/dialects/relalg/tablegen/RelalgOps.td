#ifndef TABLEGEN_RELALG_OPS
#define TABLEGEN_RELALG_OPS

include "RelalgDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AnyNumericType : AnyTypeOf<[I8, I16, I32, I64, F32, F64]>;

def Relalg_ProjectionOp : Relalg_Op<"projection", [NoSideEffect]> {
    let summary = [{
        ...
    }];
    let description = [{
        ...
    }];

    // TODO
    // let arguments = (ins Relalg_RelationType:$lhs, Relalg_PredicateType:$rhs);
    let arguments = (ins Relalg_RelationType:$lhs, Relalg_RelationType:$rhs);
    let results = (outs Relalg_RelationType:$res);
}

#endif // TABLEGEN_RELALG_OPS
