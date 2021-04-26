#include "arcise/dialects/relalg/RelalgDialect.h"
#include "arcise/dialects/relalg/RelalgOps.h"
#include "arcise/dialects/relalg/RelalgTypes.h"
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

void arcise::dialects::relalg::RelalgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "arcise/dialects/relalg/tablegen/RelalgOps.cpp.inc"
#undef GET_OP_LIST

      >();
  addTypes<RelationType>();
}

mlir::Type arcise::dialects::relalg::RelalgDialect::parseType(
    mlir::DialectAsmParser &parser) const {
  assert(0);
}

/// Print a type registered to this dialect.
void arcise::dialects::relalg::RelalgDialect::printType(
    ::mlir::Type type, ::mlir::DialectAsmPrinter &os) const {
  if (auto rel = type.dyn_cast<RelationType>()) {
    os.getStream() << "Relation<";
    if (rel.getColumnNames().size() > 0) {
      os.getStream() << rel.getColumnNames()[0];
      for (auto field : rel.getColumnNames().drop_front()) {
        os.getStream() << "," << field;
      }
    }
    os.getStream() << ">";
  }
}
