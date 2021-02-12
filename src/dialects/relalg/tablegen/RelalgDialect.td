#ifndef TABLEGEN_RELALG_DIALECT
#define TABLEGEN_RELALG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Relalg_Dialect : Dialect {
    let name = "relalg";
    let cppNamespace = "arcise::dialects::relalg";
}

// TODO

#endif // TABLEGEN_RELALG_DIALECT
