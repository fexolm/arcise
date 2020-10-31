#ifndef TABLEGEN_ARROW_OPS
#define TABLEGEN_ARROW_OPS

include "ArrowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AnyNumericType : AnyTypeOf<[I8, I16, I32, I64, F32, F64]>;

class Arrow_ConstOp<string mnemonic, list<OpTrait> traits = []> : Arrow_Op<mnemonic, traits> {
    let verifier = [{ return verifySameArrayParamAndConstType(*this); }];
}

class Arrow_SameArrayOp<string mnemonic, list<OpTrait> traits = []> : Arrow_Op<mnemonic, traits> {
    let verifier = [{ return verifyParamTypesAreSame(*this); }];
} 

// ==== Binary FUNCTIONS ====
// TODO: add trait to check if arr and rhsue types are same
def Arrow_ConstEqOp : Arrow_ConstOp<"const_eq", [NoSideEffect]> {
    let summary = [{ 
        a[i] == const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_ConstNeqOp : Arrow_ConstOp<"const_neq", [NoSideEffect]> {
    let summary = [{ 
        a[i] != const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_ConstGeOp : Arrow_ConstOp<"const_ge", [NoSideEffect]> {
    let summary = [{ 
        a[i] >= const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_ConstLeOp : Arrow_ConstOp<"const_le", [NoSideEffect]> {
    let summary = [{ 
        a[i] <= const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_ConstGtOp : Arrow_ConstOp<"const_gt", [NoSideEffect]> {
    let summary = [{ 
        a[i] > const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_ConstLtOp : Arrow_ConstOp<"const_lt", [NoSideEffect]> {
    let summary = [{ 
        a[i] < const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_EqOp : Arrow_SameArrayOp<"eq", [NoSideEffect]> {
    let summary = [{ 
        a[i] == b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}


def Arrow_NeqOp : Arrow_SameArrayOp<"neq", [NoSideEffect]> {
    let summary = [{ 
        a[i] != b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_GeOp : Arrow_SameArrayOp<"ge", [NoSideEffect]> {
    let summary = [{ 
        a[i] >= b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_LeOp : Arrow_SameArrayOp<"le", [NoSideEffect]> {
    let summary = [{ 
        a[i] <= b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_GtOp : Arrow_SameArrayOp<"gt", [NoSideEffect]> {
    let summary = [{ 
        a[i] > b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_LtOp : Arrow_SameArrayOp<"lt", [NoSideEffect]> {
    let summary = [{ 
        a[i] < b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

// ==== ARITHMETIC FUNCTIONS ====
def Arrow_ConstSumOp : Arrow_ConstOp<"const_sum", [NoSideEffect]> {
    let summary = [{
        a[i] + const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

def Arrow_ConstSubOp : Arrow_ConstOp<"const_sub", [NoSideEffect]> {
    let summary = [{
        a[i] - const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

def Arrow_ConstDivOp : Arrow_ConstOp<"const_div", [NoSideEffect]> {
    let summary = [{
        a[i] / const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

def Arrow_ConstMulOp : Arrow_ConstOp<"const_mul", [NoSideEffect]> {
    let summary = [{
        a[i] * const
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, AnyNumericType:$rhs);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

def Arrow_SumOp : Arrow_SameArrayOp<"sum", [NoSideEffect]> {
    let summary = [{
        a[i] + b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

def Arrow_SubOp : Arrow_SameArrayOp<"sub", [NoSideEffect]> {
    let summary = [{
        a[i] - b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

def Arrow_DivOp : Arrow_SameArrayOp<"div", [NoSideEffect]> {
    let summary = [{
        a[i] / b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

def Arrow_MulOp : Arrow_SameArrayOp<"mul", [NoSideEffect]> {
    let summary = [{
        a[i] * b[i]
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$lhs, Arrow_AnyArrayOrColumnType:$rhs);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

// ==== BOOLEAN FUNCTIONS ==== 

def Arrow_AndOp : Arrow_SameArrayOp<"and", [NoSideEffect]> {
    let summary = [{
        a[i] && b[i]
    }];
    let arguments = (ins Arrow_ArrayOrColumnType<I1>:$lhs, Arrow_ArrayOrColumnType<I1>:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

def Arrow_OrOp : Arrow_SameArrayOp<"or", [NoSideEffect]> {
    let summary = [{
        a[i] || b[i]
    }];
    let arguments = (ins Arrow_ArrayOrColumnType<I1>:$lhs, Arrow_ArrayOrColumnType<I1>:$rhs);
    let results = (outs Arrow_ArrayOrColumnType<I1>:$res);
}

// ==== QUERY OPERATIONS ====

def Arrow_FilterOp : Arrow_Op<"filter", [NoSideEffect]> {
    let summary = [{
        returns only rows that have 1 in bitmap
    }];
    let arguments = (ins Arrow_AnyArrayOrColumnType:$arr, Arrow_ArrayOrColumnType<I1>:$bitmap);
    let results = (outs Arrow_AnyArrayOrColumnType:$res);
}

// ==== OTHER OPERATIONS ====

def Arrow_CastToMemrefOp : Arrow_Op<"array_to_memref_cast", [NoSideEffect]> {
    let summary = [{
        cast array to memref
    }];
    let arguments = (ins Arrow_AnyArrayType:$arr);
    let results = (outs AnyMemRef: $res);
}

def Arrow_CastMemrefToArrayOp : Arrow_Op<"memref_to_array_cast", [NoSideEffect]> {
    let summary = [{
        cast array to memref
    }];
    let arguments = (ins AnyMemRef:$mem);
    let results = (outs Arrow_AnyArrayType: $res);
}

def Arrow_ConstArrayOp : Arrow_Op<"const_array", [NoSideEffect]> {
    let summary = [{
        creates const array from memrefs
    }];
    let arguments = (ins AnyMemRef:$null_bitmap, AnyMemRef: $value_buffer);
    let results = (outs Arrow_AnyArrayType: $res);
}

def Arrow_GetColumnOp : Arrow_Op<"get_column", [NoSideEffect]> {
    let summary = [{
        return column by index from table 
    }];
    let arguments = (ins StrAttr: $name);
    let results = (outs Arrow_AnyColumnType: $res);
}

def Arrow_GetArrayOp : Arrow_Op<"get_array", [NoSideEffect]> {
    let summary = [{
        returns array from table
    }];
    let arguments = (ins StrAttr: $col, I64Attr: $idx);
    let results = (outs Arrow_AnyArrayType: $res);
}

def Arrow_GetChunkOp : Arrow_Op<"get_chunk", [NoSideEffect]> {
    let summary = [{
        return chunk from column
    }];
    let arguments = (ins Arrow_AnyColumnType: $col, I64Attr: $idx);
    let results = (outs Arrow_AnyArrayType: $res);
}

def Arrow_SetChunkOp : Arrow_Op<"set_chunk", []> {
    let summary = [{
        set i'th chunk
    }];
    let arguments = (ins Arrow_AnyColumnType: $col, I64Attr: $idx, Arrow_AnyArrayType: $arr);
    let results = (outs);
}

def Arrow_CreateEmptyColumnOp : Arrow_Op<"create_empty_column", [NoSideEffect]> {
    let summary = [{
        allocates column
    }];
    let arguments = (outs);
    let results = (outs Arrow_AnyColumnType: $res);
}

def Arrow_ReturnColumnOp : Arrow_Op<"return_column", []> {
    let summary = [{
        return resulting column
    }];
    let arguments = (ins Arrow_AnyColumnType: $col, StrAttr: $name);
    let results = (outs);
}

def Arrow_ReturnArrayOp : Arrow_Op<"return_array", []> {
    let summary = [{
        return resulting array
    }];
    let arguments = (ins Arrow_AnyArrayOrMemref: $arr, StrAttr: $col, I64Attr: $idx);
    let results = (outs);
}


#endif // TABLEGEN_ARROW_OPS