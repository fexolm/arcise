#include "dialects/arrow/ArrowTypes.h"
#include "mlir/IR/Types.h"
#include <tuple>
#include <utility>

namespace arcise::dialects::arrow {

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

struct RecordBatchTypeStorage : public mlir::TypeStorage {
  using KeyTy =
      std::tuple<mlir::ArrayRef<std::string>, mlir::ArrayRef<mlir::Type>>;

  static RecordBatchTypeStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RecordBatchTypeStorage>())
        RecordBatchTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const { return key == KeyTy(names, types); }

  RecordBatchTypeStorage(const KeyTy &key)
      : names(std::get<0>(key)), types(std::get<1>(key)) {}

  std::vector<std::string> names;
  std::vector<mlir::Type> types;
};
} // namespace detail

ArrayType ArrayType::get(mlir::MLIRContext *ctx, Type elementType) {
  return Base::get(ctx, elementType);
}

mlir::Type ArrayType::getElementType() const { return getImpl()->elementType; }

RecordBatchType RecordBatchType::get(mlir::MLIRContext *ctx,
                                     mlir::ArrayRef<std::string> names,
                                     mlir::ArrayRef<mlir::Type> types) {
  return Base::get(ctx, names, types);
}

ArrayType RecordBatchType::getColumnType(const std::string &name) const {
  auto &names = getImpl()->names;
  auto idx =
      std::distance(names.begin(), std::find(names.begin(), names.end(), name));

  return getImpl()->types[idx].cast<ArrayType>();
}

mlir::ArrayRef<std::string> RecordBatchType::getColumnNames() const {
  return getImpl()->names;
}

mlir::ArrayRef<mlir::Type> RecordBatchType::getColumnTypes() const {
  return getImpl()->types;
}

} // namespace arcise::dialects::arrow
