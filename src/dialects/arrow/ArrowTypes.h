#pragma once

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace arcise::dialects::arrow {
namespace detail {
class ArrayTypeStorage;
class RecordBatchTypeStorage;
} // namespace detail

struct ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                               detail::ArrayTypeStorage> {
  using Base::Base;
  using ImplType = detail::ArrayTypeStorage;

  static ArrayType get(mlir::MLIRContext *ctx, Type elementType);

  mlir::Type getElementType() const;
};

struct RecordBatchType
    : public mlir::Type::TypeBase<RecordBatchType, mlir::Type,
                                  detail::RecordBatchTypeStorage> {
  using Base::Base;
  using ImplType = detail::RecordBatchTypeStorage;

  static RecordBatchType get(mlir::MLIRContext *ctx,
                             mlir::ArrayRef<std::string> names,
                             mlir::ArrayRef<mlir::Type> types);

  ArrayType getColumnType(const std::string &name) const;

  mlir::ArrayRef<std::string> getColumnNames() const;
  mlir::ArrayRef<mlir::Type> getColumnTypes() const;
};

} // namespace arcise::dialects::arrow