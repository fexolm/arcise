#pragma once

#include "arcise/dialects/relalg/frontends/sql/AST.h"
#include "arcise/interfaces/Schema.h"

#include <mlir/IR/BuiltinOps.h>

namespace arcise::dialects::relalg::frontends::sql {

mlir::ModuleOp translate_to_mlir(mlir::MLIRContext *ctx,
                                 const SchemaProvider &schema_provider,
                                 std::shared_ptr<SqlNode> ast);
}
