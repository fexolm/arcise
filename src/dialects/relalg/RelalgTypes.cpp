#include "dialects/relalg/RelalgTypes.h"
#include "mlir/IR/Types.h"
#include <tuple>
#include <utility>

namespace arcise::dialects::relalg {
namespace detail {
struct RelationTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type>;

  static RelationTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<RelationTypeStorage>()) RelationTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }

  RelationTypeStorage(const KeyTy &key) : elementType(std::get<0>(key)) {}

  mlir::Type elementType;
};
} // namespace detail

RelationType RelationType::get(mlir::MLIRContext *ctx, mlir::ArrayRef<mlir::Type>, mlir::ArrayRef<std::string>) {
  return Base::get(ctx, elementType);
}

mlir::ArrayRef<mlir::Type> RelationType::getColTypes(void) const { return; }
mlir::ArrayRef<std::string> RelationType::getColNames(void) const { return; }

} // namespace arcise::dialects::relalg
