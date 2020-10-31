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

struct ColumnTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type, mlir::ArrayRef<ArrayType>>;

  static ColumnTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<ColumnTypeStorage>()) ColumnTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, chunks);
  }

  ColumnTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), chunks(std::get<1>(key)) {}

  mlir::Type elementType;
  std::vector<ArrayType> chunks;
};
} // namespace detail
ArrayType ArrayType::get(mlir::MLIRContext *ctx, Type elementType,
                         size_t length) {
  return Base::get(ctx, elementType, length);
}

mlir::Type ArrayType::getElementType() const { return getImpl()->elementType; }

size_t ArrayType::getLength() const { return getImpl()->length; }

ColumnType ColumnType::get(mlir::MLIRContext *ctx, Type elementType,
                           mlir::ArrayRef<ArrayType> chunks) {
  return Base::get(ctx, elementType, chunks);
}
mlir::Type ColumnType::getElementType() const { return getImpl()->elementType; }

mlir::ArrayRef<ArrayType> ColumnType::getChunks() const {
  return getImpl()->chunks;
}

} // namespace arcise::dialects::arrow
