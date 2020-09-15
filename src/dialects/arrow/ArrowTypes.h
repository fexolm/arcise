#pragma once

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace arcise::dialects {
namespace detail {
class ArrayTypeStorage;
} // namespace detail

class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                              detail::ArrayTypeStorage> {
public:
  using Base::Base;

  static ArrayType get(mlir::MLIRContext *ctx, Type elementType);

  mlir::Type elementType() const;
};

} // namespace arcise::dialects