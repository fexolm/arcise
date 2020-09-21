#pragma once

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace arcise::dialects::arrow {
namespace detail {
class ChunkedArrayTypeStorage;
} // namespace detail

struct ChunkedArrayType
    : public mlir::Type::TypeBase<ChunkedArrayType, mlir::Type,
                                  detail::ChunkedArrayTypeStorage> {
  using Base::Base;
  using ImplType = detail::ChunkedArrayTypeStorage;

  static ChunkedArrayType get(mlir::MLIRContext *ctx, Type elementType);

  mlir::Type elementType() const;
};
} // namespace arcise::dialects::arrow