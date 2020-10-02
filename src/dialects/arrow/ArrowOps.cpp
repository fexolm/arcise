#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace arcise::dialects::arrow {

mlir::LogicalResult verifySameArrayParamAndConstType(mlir::Operation *op) {
  if (op->getNumOperands() != 2) {
    return op->emitOpError() << "Expected exactly 2 operands";
  }
  auto op0 = op->getOperand(0);
  auto op1 = op->getOperand(1);
  if (!op0.getType().isa<ArrayType>()) {
    return op->emitOpError() << "Expected first operand to be ArrayType";
  }
  if (op1.getType() != op0.getType().cast<ArrayType>().elementType()) {
    return op->emitOpError()
           << "Expected constant to be the same type as array elements";
  }
  return mlir::success();
}

mlir::LogicalResult verifyParamTypesAreSame(mlir::Operation *op) {
  if (op->getNumOperands() != 2) {
    return op->emitOpError() << "Expected exactly 2 operands";
  }
  auto op0 = op->getOperand(0);
  auto op1 = op->getOperand(1);
  if (!op0.getType().isa<ArrayType>() && op1.getType() == op0.getType()) {
    return op->emitOpError() << "Expected operands to be the same ArrayType";
  }

  return mlir::success();
}

#define GET_OP_CLASSES
#include "dialects/arrow/tablegen/ArrowOps.cpp.inc"
} // namespace arcise::dialects::arrow
