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

def Arrow_IsArrayType : CPred<"$_self.isa<arcise::dialects::arrow::ArrayType>()">;
class Arrow_ArrayType<Type type> : DialectType<Arrow_Dialect, Arrow_IsArrayType, "Concrete array type">;
def Arrow_AnyArrayType : DialectType<Arrow_Dialect, Arrow_IsArrayType, "Any array type">;

def Arrow_AnyArrayOrScalarType : DialectType<Arrow_Dialect,
        Or<[Arrow_IsArrayType, AnyInteger.predicate, AnyFloat.predicate]>,
        "Any array or scalar type">;

def Arrow_IsRecordBatchType : CPred<"$_self.isa<arcise::dialects::arrow::RecordBatchType>()">;
def Arrow_RecordBatchType : DialectType<Arrow_Dialect, Arrow_IsRecordBatchType, "Record batch type">;

#endif // TABLEGEN_ARROW_DIALECT