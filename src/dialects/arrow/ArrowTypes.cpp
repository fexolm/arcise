#include "dialects/arrow/ArrowTypes.h"
#include "mlir/IR/Types.h"
#include <tuple>
#include <utility>

namespace arcise::dialects::arrow {

namespace detail {
struct ChunkedArrayTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type>;

  static ChunkedArrayTypeStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<ChunkedArrayTypeStorage>())
        ChunkedArrayTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }

  ChunkedArrayTypeStorage(const KeyTy &key) : elementType(std::get<0>(key)) {}

  mlir::Type elementType;
};
} // namespace detail
ChunkedArrayType ChunkedArrayType::get(mlir::MLIRContext *ctx,
                                       Type elementType) {
  return Base::get(ctx, elementType);
}

mlir::Type ChunkedArrayType::elementType() const {
  return getImpl()->elementType;
}

} // namespace arcise::dialects::arrow
