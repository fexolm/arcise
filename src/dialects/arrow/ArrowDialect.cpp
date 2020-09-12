#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"

namespace arcise::dialects {
void ArrowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialects/arrow/ArrowOps.cpp.inc"
      >();
}
} // namespace arcise::dialects
