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

def Arrow_IsColumnType : CPred<"$_self.isa<arcise::dialects::arrow::ColumnType>()">;
class Arrow_ColumnType<Type type> : DialectType<Arrow_Dialect, Arrow_IsColumnType, "Concrete array type">;
def Arrow_AnyColumnType : DialectType<Arrow_Dialect, Arrow_IsColumnType, "Any array type">;

def Arrow_IsTableType : CPred<"$_self.isa<arcise::dialects::arrow::TableType>()">;
def Arrow_TableType : DialectType<Arrow_Dialect, Arrow_IsTableType, "Table type">;

class Arrow_ArrayOrColumnType<Type type> : DialectType<Arrow_Dialect, Or<[Arrow_IsColumnType, Arrow_IsArrayType]>, "Array or column">;
def Arrow_AnyArrayOrColumnType: DialectType<Arrow_Dialect, Or<[Arrow_IsColumnType, Arrow_IsArrayType]>, "Array or column">;

def Arrow_AnyArrayOrMemref: DialectType<Arrow_Dialect, Or<[IsMemRefTypePred, Arrow_IsArrayType]>, "Array or memref">;

#endif // TABLEGEN_ARROW_DIALECT