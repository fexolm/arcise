#pragma once

#include "mlir/IR/Dialect.h"

namespace arcise::dialects {
using namespace mlir;
#include "dialects/arrow/tablegen/ArrowOpsDialect.h.inc"
} // namespace arcise::dialects