#ifndef TABLEGEN_ARROW_DIALECT
#define TABLEGEN_ARROW_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Arrow_Dialect : Dialect {
    let name = "arrow";
    let cppNamespace = "arcise::dialects::arrow";
}

class Arrow_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Arrow_Dialect, mnemonic, traits>;

def Arrow_IsChunkedArrayType : CPred<"$_self.isa<arcise::dialects::arrow::ChunkedArrayType>()">;

class Arrow_ChunkedArrayType<Type type> : DialectType<Arrow_Dialect, Arrow_IsChunkedArrayType, "Concrete array type">;
def Arrow_AnyChunkedArrayType : DialectType<Arrow_Dialect, Arrow_IsChunkedArrayType, "Any array type">;

#endif // TABLEGEN_ARROW_DIALECT