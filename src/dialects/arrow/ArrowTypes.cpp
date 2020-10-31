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
  using KeyTy = std::tuple<mlir::Type, mlir::ArrayRef<mlir::Type>>;

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
  std::vector<mlir::Type> chunks;
};

struct TableTypeStorage : public mlir::TypeStorage {
  using KeyTy =
      std::tuple<mlir::ArrayRef<std::string>, mlir::ArrayRef<mlir::Type>>;

  static TableTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<TableTypeStorage>()) TableTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const { return key == KeyTy(names, types); }

  TableTypeStorage(const KeyTy &key)
      : names(std::get<0>(key)), types(std::get<1>(key)) {}

  std::vector<std::string> names;
  std::vector<mlir::Type> types;
};
} // namespace detail

ArrayType ArrayType::get(mlir::MLIRContext *ctx, Type elementType,
                         size_t length) {
  return Base::get(ctx, elementType, length);
}

mlir::Type ArrayType::getElementType() const { return getImpl()->elementType; }

size_t ArrayType::getLength() const { return getImpl()->length; }

ColumnType ColumnType::get(mlir::MLIRContext *ctx, Type elementType,
                           mlir::ArrayRef<mlir::Type> chunks) {
  return Base::get(ctx, elementType, chunks);
}

mlir::ArrayRef<mlir::Type> ColumnType::getChunks() const {
  return getImpl()->chunks;
}

ArrayType ColumnType::getChunk(size_t idx) const {
  return getImpl()->chunks[idx].cast<ArrayType>();
}

size_t ColumnType::getChunksCount() const { return getImpl()->chunks.size(); }

TableType TableType::get(mlir::MLIRContext *ctx,
                         mlir::ArrayRef<std::string> names,
                         mlir::ArrayRef<mlir::Type> types) {
  return Base::get(ctx, names, types);
}

ColumnType TableType::getColumnType(const std::string &name) const {
  auto &names = getImpl()->names;
  auto idx =
      std::distance(names.begin(), std::find(names.begin(), names.end(), name));

  return getImpl()->types[idx].cast<ColumnType>();
}

mlir::ArrayRef<std::string> TableType::getColumnNames() const {
  return getImpl()->names;
}

mlir::ArrayRef<mlir::Type> TableType::getColumnTypes() const {
  return getImpl()->types;
}

} // namespace arcise::dialects::arrow
