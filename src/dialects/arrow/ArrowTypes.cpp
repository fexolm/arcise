#include "dialects/arrow/ArrowTypes.h"
#include "mlir/IR/Types.h"
#include <tuple>
#include <utility>

namespace arcise::dialects::arrow {

namespace detail {
struct ArrayTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type, size_t>;

  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>()) ArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, length);
  }

  ArrayTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), length(std::get<1>(key)) {}

  mlir::Type elementType;
  size_t length;
};
} // namespace detail
ArrayType ArrayType::get(mlir::MLIRContext *ctx, Type elementType,
                         size_t length) {
  return Base::get(ctx, elementType, length);
}

mlir::Type ArrayType::elementType() const { return getImpl()->elementType; }

size_t ArrayType::length() const { return getImpl()->length; }

} // namespace arcise::dialects::arrow
