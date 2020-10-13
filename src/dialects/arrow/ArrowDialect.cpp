#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include <mlir/IR/DialectImplementation.h>

namespace arcise::dialects::arrow {
void ArrowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialects/arrow/tablegen/ArrowOps.cpp.inc"
#undef GET_OP_LIST
      >();
  addTypes<ArrayType, ColumnType>();
}

mlir::Type ArrowDialect::parseType(mlir::DialectAsmParser &parser) const {}
void ArrowDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &printer) const {
  if (type.isa<ArrayType>()) {
    auto arrayType = type.cast<ArrayType>();
    printer.getStream() << "array<";
    printer.printType(arrayType.getElementType());
    printer.getStream() << ", length=" << arrayType.getLength();
    printer.getStream() << ">";
  }

  if (type.isa<ColumnType>()) {
    auto columnType = type.cast<ColumnType>();
    printer.getStream() << "column<";
    printer.printType(columnType.getElementType());
    printer.getStream() << ">";
  }
}
} // namespace arcise::dialects::arrow
