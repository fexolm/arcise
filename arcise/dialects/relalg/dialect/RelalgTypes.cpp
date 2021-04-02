#include "arcise/dialects/relalg/dialect/RelalgTypes.h"
#include <mlir/IR/Types.h>
#include <tuple>
#include <utility>

namespace arcise::dialects::relalg {
namespace detail {
struct RelationTypeStorage : public mlir::TypeStorage {
  using KeyTy =
      std::tuple<mlir::ArrayRef<std::string>, mlir::ArrayRef<mlir::Type>>;

  static RelationTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<RelationTypeStorage>())
        RelationTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const { return key == KeyTy(names, types); }

  RelationTypeStorage(const KeyTy &key)
      : names(std::get<0>(key)), types(std::get<1>(key)) {}

  std::vector<std::string> names;
  std::vector<mlir::Type> types;
};
} // namespace detail

RelationType RelationType::get(mlir::MLIRContext *ctx,
                               mlir::ArrayRef<std::string> names,
                               mlir::ArrayRef<mlir::Type> types) {
  return Base::get(ctx, names, types);
}

mlir::ArrayRef<std::string> RelationType::getColumnNames() const {
  return getImpl()->names;
}

mlir::ArrayRef<mlir::Type> RelationType::getColumnTypes() const {
  return getImpl()->types;
}
} // namespace arcise::dialects::relalg
