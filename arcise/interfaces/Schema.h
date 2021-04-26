#pragma once
#include <string>
#include <vector>
namespace arcise {
enum class ColumnType { I64, F64 };

struct Schema {
  std::vector<std::pair<std::string, ColumnType>> fields;
};

class SchemaProvider {
public:
  virtual const Schema &
  get_schema_for_table(const std::string &table_name) const = 0;
  virtual ~SchemaProvider() = default;
};
} // namespace arcise