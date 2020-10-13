#pragma once

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace arcise::dialects::arrow {
namespace detail {
class ArrayTypeStorage;
class ColumnTypeStorage;
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
                        size_t chunksCount,
                        mlir::ArrayRef<size_t> chunkLengths);

  mlir::Type getElementType() const;
  size_t getChunksCount() const;
  mlir::ArrayRef<size_t> getChunkLengths() const;
};
} // namespace arcise::dialects::arrow