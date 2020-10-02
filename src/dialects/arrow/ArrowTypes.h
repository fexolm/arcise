#pragma once

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace arcise::dialects::arrow {
namespace detail {
class ArrayTypeStorage;
} // namespace detail

struct ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                               detail::ArrayTypeStorage> {
  using Base::Base;
  using ImplType = detail::ArrayTypeStorage;

  static ArrayType get(mlir::MLIRContext *ctx, Type elementType, size_t length);

  mlir::Type elementType() const;
  size_t length() const;
};
} // namespace arcise::dialects::arrow