#include "dialects/relalg/RelalgDialect.h"
#include "dialects/relalg/RelalgOps.h"
#include "dialects/relalg/RelalgTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

void arcise::dialects::relalg::RelalgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialects/relalg/tablegen/RelalgOps.cpp.inc"
#undef GET_OP_LIST

      >();
  addTypes<RelationType>();
}
namespace arcise::dialects::relalg {
    // TODO
} // namespace arcise::dialects::relalg
