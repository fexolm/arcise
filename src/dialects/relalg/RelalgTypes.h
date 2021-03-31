#pragma once

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace arcise::dialects::relalg {
namespace detail {
class RelationTypeStorage;
} // namespace detail

struct PredicateType;

struct RelationType : public mlir::Type::TypeBase<RelationType, mlir::Type,
                                               detail::RelationTypeStorage> {
  using Base::Base;
  using ImplType = detail::RelationTypeStorage;

  static RelationType get(mlir::MLIRContext *ctx, mlir::ArrayRef<mlir::Type>, mlir::ArrayRef<std::string>);

  mlir::ArrayRef<mlir::Type> getColTypes(void) const;
  mlir::ArrayRef<std::string> getColNames(void) const;
};
} // namespace arcise::dialects::relalg
