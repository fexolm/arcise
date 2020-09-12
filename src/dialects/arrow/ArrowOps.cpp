#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowDialect.h"
#include "mlir/IR/OpImplementation.h"

namespace arcise::dialects {
#define GET_OP_CLASSES
#include "dialects/arrow/ArrowOps.cpp.inc"
} // namespace arcise::dialiects
