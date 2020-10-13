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
  using KeyTy = std::tuple<mlir::Type, size_t, mlir::ArrayRef<size_t>>;

  static ColumnTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<ColumnTypeStorage>()) ColumnTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, chunksCount, chunkLengths);
  }

  ColumnTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), chunksCount(std::get<1>(key)),
        chunkLengths(std::get<2>(key)) {}

  mlir::Type elementType;
  size_t chunksCount;
  std::vector<size_t> chunkLengths;
};
} // namespace detail
ArrayType ArrayType::get(mlir::MLIRContext *ctx, Type elementType,
                         size_t length) {
  return Base::get(ctx, elementType, length);
}

mlir::Type ArrayType::getElementType() const { return getImpl()->elementType; }

size_t ArrayType::getLength() const { return getImpl()->length; }

ColumnType ColumnType::get(mlir::MLIRContext *ctx, Type elementType,
                           size_t chunksCount,
                           mlir::ArrayRef<size_t> chunkLengths) {
  return Base::get(ctx, elementType, chunksCount, chunkLengths);
}
mlir::Type ColumnType::getElementType() const { return getImpl()->elementType; }

size_t ColumnType::getChunksCount() const { return getImpl()->chunksCount; }

mlir::ArrayRef<size_t> ColumnType::getChunkLengths() const {
  return getImpl()->chunkLengths;
}

} // namespace arcise::dialects::arrow
