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

def Arrow_GetColumnOp : Arrow_Op<"get_column", [NoSideEffect]> {
    let summary = [{
        return column by index from table 
    }];
    let arguments = (ins Arrow_TableType: $table, StrAttr: $name);
    let results = (outs Arrow_AnyColumnType: $res);
}

def Arrow_UnwrapArrayOp : Arrow_Op<"unwrap_array", [NoSideEffect]> {
    let summary = [{
        return array's buffers
    }];
    let arguments = (ins Arrow_AnyArrayType: $arr);
    let results = (outs AnyMemRef: $null_bitmap,
                        AnyMemRef: $data_buffer);
    let hasCanonicalizer = 1;
}

def Arrow_UnwrapColumnOp : Arrow_Op<"unwrap_column", [NoSideEffect]> {
    let summary = [{
        return column's chunks
    }];
    let arguments = (ins Arrow_AnyColumnType: $col);
    let results = (outs Variadic<Arrow_AnyArrayType>: $res);
    let hasCanonicalizer = 1;
}

def Arrow_MakeArrayOp : Arrow_Op<"make_array", [NoSideEffect]> {
    let summary = [{
        construct array from buffers
    }];
    let arguments = (ins I1MemRef: $null_bitmap, AnyMemRef: $data_buffer);
    let results = (outs Arrow_AnyArrayType: $arr);
}

def Arrow_MakeColumnOp : Arrow_Op<"make_column", [NoSideEffect]> {
    let summary = [{
        make column from arrays
    }];
    let arguments = (ins Variadic<Arrow_AnyArrayType>: $chunks);
    let results = (outs Arrow_AnyColumnType: $res);
}

def Arrow_MakeTableOp : Arrow_Op<"make_table", [NoSideEffect]> {
    let summary = [{
        make table from columns
    }];
    let arguments = (ins Variadic<Arrow_AnyColumnType>: $columns);
    let results = (outs Arrow_TableType: $res);
}


#endif // TABLEGEN_ARROW_OPS