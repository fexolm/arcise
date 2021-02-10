#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

void arcise::dialects::arrow::ArrowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialects/arrow/tablegen/ArrowOps.cpp.inc"
#undef GET_OP_LIST

      >();
  addTypes<ArrayType, RecordBatchType>();
}

namespace arcise::dialects::arrow {
mlir::Type ArrowDialect::parseType(mlir::DialectAsmParser &parser) const {
  assert(false);
  return nullptr;
}

void ArrowDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &printer) const {
  if (type.isa<ArrayType>()) {
    auto arrayType = type.cast<ArrayType>();
    printer.getStream() << "array<";
    printer.printType(arrayType.getElementType());
    printer.getStream() << ">";
  }

  if (type.isa<RecordBatchType>()) {
    printer.getStream() << "record_batch";
  }
}
} // namespace arcise::dialects::arrow
