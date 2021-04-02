#ifndef TABLEGEN_RELALG_COMPARE_OPS
#define TABLEGEN_RELALG_COMPARE_OPS

include "RelalgDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AnyLitType : AnyTypeOf<[I8, I16, I32, I64, F32, F64]>;

class Relalg_CompareOp<string name> : Relalg_Op<name, [NoSideEffect]> {
    let arguments = (ins Relalg_RelationType: $reference_table, StrAttr: $col_name, AnyLitType: $val);
    let results = (outs Relalg_RelationType: $filter);
}

def Relalg_LTOp : Relalg_CompareOp<"lt">;
def Relalg_GTOp : Relalg_CompareOp<"gt">;
def Relalg_LEOp : Relalg_CompareOp<"le">;
def Relalg_GEOp : Relalg_CompareOp<"ge">;
def Relalg_EQOp : Relalg_CompareOp<"eq">;
def Relalg_NEQOp : Relalg_CompareOp<"neq">;

class Relalg_PredicateOp<string name> : Relalg_Op<name, [NoSideEffect]> {
    let arguments = (ins Relalg_RelationType: $lhs, Relalg_RelationType: $rhs);
    let results = (outs Relalg_RelationType: $filter);
}

def Relalg_AndOp : Relalg_PredicateOp<"and">;
def Relalg_OrOp : Relalg_PredicateOp<"or">;

#endif // TABLEGEN_RELALG_COMPARE_OPS