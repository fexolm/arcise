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
  addTypes<ChunkedArrayType>();
}

mlir::Type ArrowDialect::parseType(mlir::DialectAsmParser &parser) const {}
void ArrowDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &printer) const {
  if (type.isa<ChunkedArrayType>()) {
    auto arrayType = type.cast<ChunkedArrayType>();
    printer.getStream() << "array<";
    printer.printType(arrayType.elementType());
    printer.getStream() << ">";
  }
}
} // namespace arcise::dialects::arrow
