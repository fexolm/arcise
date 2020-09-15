#ifndef TABLEGEN_ARROW_DIALECT
#define TABLEGEN_ARROW_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Arrow_Dialect : Dialect {
    let name = "arrow";
    let cppNamespace = "arcise::dialects";
}

class Arrow_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Arrow_Dialect, mnemonic, traits>;

def Arrow_IsArrayType : CPred<"$_self.isa<arcise::dialects::ArrayType>()">;

class Arrow_ArrayType<Type type> : DialectType<Arrow_Dialect, Arrow_IsArrayType, "Array type">;

#endif // TABLEGEN_ARROW_DIALECT