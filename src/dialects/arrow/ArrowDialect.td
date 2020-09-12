#ifndef TABLEGEN_ARROW_DIALECT
#define TABLEGEN_ARROW_DIALECT

include "mlir/IR/OpBase.td"

def Arrow_Dialect : Dialect {
    let name = "arrow";
    let cppNamespace = "arcise::dialects";
}

class Arrow_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Arrow_Dialect, mnemonic, traits>;

#endif // TABLEGEN_ARROW_DIALECT