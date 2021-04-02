#ifndef TABLEGEN_RELALG_DIALECT
#define TABLEGEN_RELALG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Relalg_Dialect : Dialect {
    let name = "relalg";
    let cppNamespace = "arcise::dialects::relalg";
}

class Relalg_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Relalg_Dialect, mnemonic, traits>;

def Relalg_RelationType : DialectType<Relalg_Dialect, CPred<"$_self.isa<arcise::dialects::relalg::RelationType>()">, "Relation">;

#endif // TABLEGEN_RELALG_DIALECT