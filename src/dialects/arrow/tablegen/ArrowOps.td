#ifndef TABLEGEN_ARROW_OPS
#define TABLEGEN_ARROW_OPS

include "ArrowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AnyNumericType : AnyTypeOf<[I8, I16, I32, I64, F32, F64]>;

// ==== Binary FUNCTIONS ====
// TODO: add trait to check if arr and rhsue types are same
def Arrow_EqOp : Arrow_Op<"eq", [NoSideEffect]> {
    let summary = [{ 
        a[i] == b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_ArrayType<I1>:$res);
}


def Arrow_NeqOp : Arrow_Op<"neq", [NoSideEffect]> {
    let summary = [{ 
        a[i] != b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_ArrayType<I1>:$res);
}

def Arrow_GeOp : Arrow_Op<"ge", [NoSideEffect]> {
    let summary = [{ 
        a[i] >= b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_ArrayType<I1>:$res);
}

def Arrow_LeOp : Arrow_Op<"le", [NoSideEffect]> {
    let summary = [{ 
        a[i] <= b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_ArrayType<I1>:$res);
}

def Arrow_GtOp : Arrow_Op<"gt", [NoSideEffect]> {
    let summary = [{ 
        a[i] > b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_ArrayType<I1>:$res);
}

def Arrow_LtOp : Arrow_Op<"lt", [NoSideEffect]> {
    let summary = [{ 
        a[i] < b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_ArrayType<I1>:$res);
}

// ==== ARITHMETIC FUNCTIONS ====

def Arrow_SumOp : Arrow_Op<"sum", [NoSideEffect]> {
    let summary = [{
        a[i] + b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_AnyArrayType:$res);
}

def Arrow_SubOp : Arrow_Op<"sub", [NoSideEffect]> {
    let summary = [{
        a[i] - b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_AnyArrayType:$res);
}

def Arrow_DivOp : Arrow_Op<"div", [NoSideEffect]> {
    let summary = [{
        a[i] / b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_AnyArrayType:$res);
}

def Arrow_MulOp : Arrow_Op<"mul", [NoSideEffect]> {
    let summary = [{
        a[i] * b[i]
    }];
    let arguments = (ins Arrow_AnyArrayType:$lhs, Arrow_AnyArrayOrScalarType:$rhs);
    let results = (outs Arrow_AnyArrayType:$res);
}

// ==== BOOLEAN FUNCTIONS ==== 

def Arrow_AndOp : Arrow_Op<"and", [NoSideEffect]> {
    let summary = [{
        a[i] && b[i]
    }];
    let arguments = (ins Arrow_ArrayType<I1>:$lhs, Arrow_ArrayType<I1>:$rhs);
    let results = (outs Arrow_ArrayType<I1>:$res);
}

def Arrow_OrOp : Arrow_Op<"or", [NoSideEffect]> {
    let summary = [{
        a[i] || b[i]
    }];
    let arguments = (ins Arrow_ArrayType<I1>:$lhs, Arrow_ArrayType<I1>:$rhs);
    let results = (outs Arrow_ArrayType<I1>:$res);
}

// ==== OTHER OPERATIONS ====

def Arrow_GetColumnOp : Arrow_Op<"get_column", [NoSideEffect]> {
    let summary = [{
        return column by name from record batch 
    }];
    let arguments = (ins Arrow_RecordBatchType: $recordBatch, StrAttr: $columnName);
    let results = (outs Arrow_AnyArrayType: $res);
}

def Arrow_GetNullBitmapOp : Arrow_Op<"get_null_bitmap", [NoSideEffect]> {
    let summary = [{
        return null bitmap buffer from array 
    }];
    let arguments = (ins Arrow_AnyArrayType: $array);
    let results = (outs AnyMemRef: $res);
    let hasCanonicalizer = 1;
}

def Arrow_GetDataBufferOp : Arrow_Op<"get_data_buffer", [NoSideEffect]> {
    let summary = [{
        return data buffer from array 
    }];
    let arguments = (ins Arrow_AnyArrayType: $array);
    let results = (outs AnyMemRef: $res);
    let hasCanonicalizer = 1;
}

def Arrow_GetLengthOp : Arrow_Op<"get_length", [NoSideEffect]> {
    let summary = [{
        return length of array 
    }];
    let arguments = (ins Arrow_AnyArrayType: $array);
    let results = (outs Index: $res);
    let hasCanonicalizer = 1;
}

def Arrow_GetRowsCountOp : Arrow_Op<"get_rows_count", [NoSideEffect]> {
    let summary = [{
        return rows count of record batch 
    }];
    let arguments = (ins Arrow_RecordBatchType: $recordBatch);
    let results = (outs Index: $res);
}

def Arrow_MakeArrayOp : Arrow_Op<"make_array", [NoSideEffect]> {
    let summary = [{
        construct array from buffers
    }];
    let arguments = (ins I1MemRef: $null_bitmap, AnyMemRef: $data_buffer);
    let results = (outs Arrow_AnyArrayType: $res);
}

def Arrow_MakeRecordBatchOp : Arrow_Op<"make_recordbatch", [NoSideEffect]> {
    let summary = [{
        make recordbatch from arrays
    }];
    let arguments = (ins Variadic<Arrow_AnyArrayType>: $columns);
    let results = (outs Arrow_RecordBatchType: $res);
}

#endif // TABLEGEN_ARROW_OPS