#pragma once

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

namespace arcise::dialects::relalg {
namespace detail {
struct RelationTypeStorage;
}

struct RelationType : public mlir::Type::TypeBase<RelationType, mlir::Type,
                                                  detail::RelationTypeStorage> {
  using Base::Base;
  using ImplType = detail::RelationTypeStorage;

  static RelationType get(mlir::MLIRContext *ctx,
                          mlir::ArrayRef<std::string> names,
                          mlir::ArrayRef<mlir::Type> types);

  mlir::ArrayRef<std::string> getColumnNames() const;
  mlir::ArrayRef<mlir::Type> getColumnTypes() const;
};
} // namespace arcise::dialects::relalg