#include "dialects/arrow/ArrowTypes.h"
#include "mlir/IR/Types.h"
#include <tuple>
#include <utility>

namespace arcise::dialects {

namespace detail {
struct ArrayTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type>;

  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }

  ArrayTypeStorage(const KeyTy &key) : elementType(std::get<0>(key)) {}

  mlir::Type elementType;
};
} // namespace detail
ArrayType ArrayType::get(mlir::MLIRContext *ctx, Type elementType) {
  return Base::get(ctx, elementType);
}

mlir::Type ArrayType::elementType() const { return getImpl()->elementType; }

} // namespace arcise::dialects
