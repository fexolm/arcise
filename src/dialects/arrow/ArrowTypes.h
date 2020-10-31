#pragma once

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace arcise::dialects::arrow {
namespace detail {
class ArrayTypeStorage;
class ColumnTypeStorage;
class TableTypeStorage;
} // namespace detail

struct ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                               detail::ArrayTypeStorage> {
  using Base::Base;
  using ImplType = detail::ArrayTypeStorage;

  static ArrayType get(mlir::MLIRContext *ctx, Type elementType, size_t length);

  mlir::Type getElementType() const;
  size_t getLength() const;
};

struct ColumnType : public mlir::Type::TypeBase<ColumnType, mlir::Type,
                                                detail::ColumnTypeStorage> {
  using Base::Base;
  using ImplType = detail::ColumnTypeStorage;

  static ColumnType get(mlir::MLIRContext *ctx, mlir::Type elementType,
                        mlir::ArrayRef<mlir::Type> chunks);

  mlir::ArrayRef<mlir::Type> getChunks() const;

  ArrayType getChunk(size_t idx) const;

  size_t getChunksCount() const;
};

struct TableType : public mlir::Type::TypeBase<TableType, mlir::Type,
                                               detail::TableTypeStorage> {
  using Base::Base;
  using ImplType = detail::TableTypeStorage;

  static TableType get(mlir::MLIRContext *ctx,
                       mlir::ArrayRef<std::string> names,
                       mlir::ArrayRef<mlir::Type> types);

  ColumnType getColumnType(const std::string &name) const;

  mlir::ArrayRef<std::string> getColumnNames() const;
  mlir::ArrayRef<mlir::Type> getColumnTypes() const;
};

} // namespace arcise::dialects::arrow