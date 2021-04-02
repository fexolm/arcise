#include "arcise/dialects/relalg/frontends/sql/SQLToMLIRConverter.h"

#include "arcise/dialects/relalg/dialect/RelalgDialect.h"
#include "arcise/dialects/relalg/dialect/RelalgOps.h"
#include "arcise/dialects/relalg/dialect/RelalgTypes.h"
#include "arcise/dialects/relalg/frontends/sql/AST.h"
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <iostream>
#include <stack>

namespace arcise::dialects::relalg::frontends::sql {

struct MLIRVisitor : SqlVisitor {
  mlir::MLIRContext *ctx;
  const SchemaProvider *schema_provider;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::ModuleOp module;
  mlir::FuncOp funcOp;
  std::stack<mlir::Value> values;
  std::stack<mlir::Attribute> attributes;

  MLIRVisitor(mlir::MLIRContext *ctx, const SchemaProvider *schema_provider)
      : ctx(ctx), schema_provider(schema_provider), builder(ctx),
        loc(builder.getUnknownLoc()) {
    module = mlir::ModuleOp::create(loc);
    auto func = builder.getFunctionType({}, {});
    funcOp = builder.create<mlir::FuncOp>(loc, "f", func);
    auto block = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(block);
  }

  void visit(const sql::IntLit &i) override {
    auto type = builder.getI64Type();
    values.push(builder.create<mlir::ConstantOp>(
        loc, type, builder.getI64IntegerAttr(i.value)));
  }

  void visit(const sql::RealLit &r) override {
    auto type = builder.getF64Type();
    values.push(builder.create<mlir::ConstantOp>(
        loc, type, builder.getF64FloatAttr(r.value)));
  }
  void visit(const sql::PredicateExpr &p) override {
    p.lhs->visit(*this);
    auto lhs = values.top();
    values.pop();
    p.rhs->visit(*this);
    auto rhs = values.top();
    values.pop();
    mlir::Type resType =
        RelationType::get(ctx, {"__builtin_filter__"}, {builder.getI1Type()});

    switch (p.op) {
    case sql::PredicateExpr::Op::AND:
      values.push(builder.create<AndOp>(loc, resType, lhs, rhs));
      return;
    case sql::PredicateExpr::Op::OR:
      values.push(builder.create<OrOp>(loc, resType, lhs, rhs));
      return;
    }
  }
  void visit(const sql::ColumnLit &comp) override {
    attributes.push(builder.getStringAttr(comp.name));
  }

  void visit(const sql::CompareExpr &comp) override {
    comp.lhs->visit(*this);
    auto lhs = attributes.top().cast<mlir::StringAttr>();
    attributes.pop();
    comp.rhs->visit(*this);
    auto rhs = values.top();
    values.pop();
    mlir::Type resType =
        RelationType::get(ctx, {"__builtin_filter__"}, {builder.getI1Type()});

    switch (comp.op) {
    case CompareExpr::Op::GT:
      values.push(builder.create<GTOp>(loc, resType, values.top(), lhs, rhs));
      return;
    case CompareExpr::Op::LT:
      values.push(builder.create<LTOp>(loc, resType, values.top(), lhs, rhs));
      return;
    case CompareExpr::Op::GE:
      values.push(builder.create<GEOp>(loc, resType, values.top(), lhs, rhs));
      return;
    case CompareExpr::Op::LE:
      values.push(builder.create<LEOp>(loc, resType, values.top(), lhs, rhs));
      return;
    case CompareExpr::Op::EQ:
      values.push(builder.create<EQOp>(loc, resType, values.top(), lhs, rhs));
      return;
    case CompareExpr::Op::NEQ:
      values.push(builder.create<NEQOp>(loc, resType, values.top(), lhs, rhs));
      return;
    }
  }
  void visit(const sql::ProjectionNode &proj) override {
    std::vector<std::string> names;
    std::vector<mlir::Type> types;

    auto scan = values.top();
    auto rel_type = scan.getType().cast<RelationType>();
    values.pop();

    auto column_names = rel_type.getColumnNames();
    auto filter_col = std::find(column_names.begin(), column_names.end(),
                                "__builtin_filter__");
    if (filter_col != column_names.end()) {
      auto idx = std::distance(column_names.begin(), filter_col);
      names.push_back(rel_type.getColumnNames()[idx]);
      types.push_back(rel_type.getColumnTypes()[idx]);
    }

    std::vector<mlir::Attribute> name_attrs;
    for (auto &expr : proj.exprs) {
      // TODO: this would work only for columnlit
      expr->visit(*this);
      auto attr = attributes.top();
      attributes.pop();
      name_attrs.push_back(attr);

      auto col = std::find(column_names.begin(), column_names.end(),
                           attr.cast<mlir::StringAttr>().getValue().str());
      assert(col != column_names.end());
      auto idx = std::distance(column_names.begin(), col);
      names.push_back(column_names[idx]);
      types.push_back(rel_type.getColumnTypes()[idx]);
    }

    mlir::Type resType = RelationType::get(ctx, names, types);

    values.push(builder.create<ProjectOp>(loc, resType, scan,
                                          builder.getArrayAttr(name_attrs)));
  }
  void visit(const sql::FilterNode &filter) override {
    filter.expr->visit(*this);
    auto filter_expr = values.top();
    values.pop();
    auto rel = values.top();
    values.pop();

    auto rel_type = rel.getType().cast<RelationType>();
    std::vector<std::string> names;
    std::vector<mlir::Type> types;

    names.insert(names.end(), rel_type.getColumnNames().begin(),
                 rel_type.getColumnNames().end());
    types.insert(types.end(), rel_type.getColumnTypes().begin(),
                 rel_type.getColumnTypes().end());

    auto filter_col =
        std::find(names.begin(), names.end(), "__builtin_filter__");
    if (filter_col == names.end()) {
      names.push_back("__builtin_filter__");
      types.push_back(builder.getI1Type());
    }

    auto res_type = RelationType::get(ctx, names, types);

    values.push(builder.create<FilterOp>(loc, res_type, rel, filter_expr));
  }
  void visit(const sql::JoinNode &join) override {
    std::vector<std::string> names;
    std::vector<mlir::Type> types;

    auto lhs = values.top();
    auto lhs_type = lhs.getType().cast<RelationType>();
    values.pop();
    join.scan->visit(*this);
    auto rhs = values.top();
    auto rhs_type = rhs.getType().cast<RelationType>();
    values.pop();

    names.insert(names.end(), lhs_type.getColumnNames().begin(),
                 lhs_type.getColumnNames().end());
    names.insert(names.end(), rhs_type.getColumnNames().begin(),
                 rhs_type.getColumnNames().end());

    types.insert(types.end(), lhs_type.getColumnTypes().begin(),
                 lhs_type.getColumnTypes().end());
    types.insert(types.end(), rhs_type.getColumnTypes().begin(),
                 rhs_type.getColumnTypes().end());

    mlir::Type resType = RelationType::get(ctx, names, types);
    values.push(builder.create<JoinOp>(loc, resType, lhs, rhs, join.lhs->name,
                                       join.rhs->name));
  }

  void visit(const sql::SelectNode &select) override {
    select.scan->visit(*this);
    if (select.filter) {
      select.filter->visit(*this);
    }
    if (select.join) {
      select.join->visit(*this);
    }
    select.projection->visit(*this);
  }
  void visit(const sql::TableScanNode &table_scan) override {
    std::vector<std::string> names;
    std::vector<mlir::Type> types;

    for (auto &field :
         schema_provider->get_schema_for_table(table_scan.table_name).fields) {
      names.push_back(field.first);
      switch (field.second) {
      case ColumnType::I64:
        types.push_back(builder.getI64Type());
        break;
      case ColumnType::F64:
        types.push_back(builder.getF64Type());
        break;
      }
    }
    mlir::Type resType = RelationType::get(ctx, names, types);
    values.push(builder.create<ScanOp>(loc, resType, table_scan.table_name));
  }

  mlir::ModuleOp finish() {
    module.push_back(funcOp);
    return module;
  }
};

mlir::ModuleOp translate_to_mlir(mlir::MLIRContext *ctx,
                                 const SchemaProvider &schema_provider,
                                 std::shared_ptr<SqlNode> ast) {
  MLIRVisitor visitor(ctx, &schema_provider);
  ast->visit(visitor);
  return visitor.finish();
}
} // namespace arcise::dialects::relalg::frontends::sql