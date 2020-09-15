#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace arcise::dialects {
#define GET_OP_CLASSES
#include "dialects/arrow/tablegen/ArrowOps.cpp.inc"
} // namespace arcise::dialects
