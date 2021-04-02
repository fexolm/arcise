#ifndef TABLEGEN_RELALG_BASE_OPS
#define TABLEGEN_RELALG_BASE_OPS

include "RelalgDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Relalg_ScanOp : Relalg_Op<"scan", [NoSideEffect]> {
    let arguments = (ins StrAttr: $table_name);
    let results = (outs Relalg_RelationType: $res);
}

def Relalg_ProjectOp : Relalg_Op<"project", [NoSideEffect]> {
    let arguments = (ins Relalg_RelationType: $rel, StrArrayAttr: $columns);
    let results = (outs Relalg_RelationType: $res);
}

def Relalg_FilterOp : Relalg_Op<"filter", [NoSideEffect]> {
    let arguments = (ins Relalg_RelationType: $rel, Relalg_RelationType: $filter);
    let results = (outs Relalg_RelationType: $res);
}

def Relalg_JoinOp : Relalg_Op<"join", [NoSideEffect]> {
    let arguments = (ins Relalg_RelationType: $lhs, Relalg_RelationType: $rhs, StrAttr: $lhs_col, StrAttr: $rhs_col);
    let results = (outs Relalg_RelationType: $res);
}

#endif // TABLEGEN_RELALG_BASE_OPS