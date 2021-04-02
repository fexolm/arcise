#include "arcise/dialects/relalg/dialect/RelalgDialect.h"
#include "arcise/dialects/relalg/frontends/sql/AST.h"
#include "arcise/dialects/relalg/frontends/sql/Parser.h"
#include "arcise/dialects/relalg/frontends/sql/SQLToMLIRConverter.h"
#include "arcise/interfaces/Schema.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>

#include <iostream>

using namespace arcise::dialects::relalg;
using namespace arcise::dialects::relalg::frontends;

class Catalog : public arcise::SchemaProvider {
public:
  const arcise::Schema &
  get_schema_for_table(const std::string &table_name) const override {
    if (table_name == "table1") {
      static auto schema = arcise::Schema{
          {{"t1a", arcise::ColumnType::I64}, {"t1b", arcise::ColumnType::I64}}};
      return schema;
    }
    if (table_name == "table2") {
      static auto schema = arcise::Schema{
          {{"t2a", arcise::ColumnType::I64}, {"t2b", arcise::ColumnType::I64}}};
      return schema;
    }
    assert(0);
  };
};

int main() {
  mlir::MLIRContext ctx;

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<RelalgDialect>();
  registry.loadAll(&ctx);

  auto ast = sql::parse_sql(
      R"(
select "t1a", "t2b" from (select "t1a", "t1b" from "table1" where "t1a" < 4) 
    where 
        "t1a" > 3 and 
        "t1b" < 4 
    join "table2" 
        on "t1b" = "t2b";
)");

  Catalog catalog;
  auto module = sql::translate_to_mlir(&ctx, catalog, ast);

  module.dump();
}