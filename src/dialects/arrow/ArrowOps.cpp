#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
namespace arcise::dialects::arrow {

mlir::LogicalResult verifySameArrayParamAndConstType(mlir::Operation *op) {
  if (op->getNumOperands() != 2) {
    return op->emitOpError() << "Expected exactly 2 operands";
  }
  auto op0 = op->getOperand(0).getType();
  auto op1 = op->getOperand(1).getType();

  if (auto lhs = op0.dyn_cast_or_null<ArrayType>()) {
    if (op1 != lhs.getElementType()) {
      return op->emitOpError()
             << "Expected constant to be the same type as array elements";
    }
  } else if (auto lhs = op0.dyn_cast_or_null<ColumnType>()) {
    if (op1 != lhs.getElementType()) {
      return op->emitOpError()
             << "Expected constant to be the same type as column elements";
    }
  } else {
    return op->emitOpError()
           << "Expected first operand to be ArrayType or ColumnType";
  }

  return mlir::success();
}

mlir::LogicalResult verifyParamTypesAreSame(mlir::Operation *op) {
  if (op->getNumOperands() != 2) {
    return op->emitOpError() << "Expected exactly 2 operands";
  }
  auto op0 = op->getOperand(0).getType();
  auto op1 = op->getOperand(1).getType();

  if (auto lhs = op0.dyn_cast_or_null<ArrayType>()) {
    if (auto rhs = op1.dyn_cast_or_null<ArrayType>()) {
      if (lhs.getElementType() != rhs.getElementType()) {
        return op->emitOpError()
               << "Expected both operands to have the same elements type";
      }
      if (lhs.getLength() != rhs.getLength()) {
        return op->emitOpError()
               << "Expected both operands to have the same length";
      }
    } else {
      return op->emitOpError() << "Expected both operands to be array types";
    }
  } else if (auto lhs = op0.dyn_cast_or_null<ColumnType>()) {
    if (auto rhs = op1.dyn_cast_or_null<ColumnType>()) {
      if (lhs.getElementType() != rhs.getElementType()) {
        return op->emitOpError()
               << "Expected both operands to have the same elements type";
      }
      if (lhs.getChunksCount() != rhs.getChunksCount()) {
        return op->emitOpError()
               << "Expected both operands to have the same chunks count";
      }
    } else {
      return op->emitOpError() << "Expected both operands to be column types";
    }
  } else {
    return op->emitOpError() << "Expected operands to be Column or Array Types";
  }

  return mlir::success();
}

#define GET_OP_CLASSES
#include "dialects/arrow/tablegen/ArrowOps.cpp.inc"
} // namespace arcise::dialects::arrow
