#include "sql_parser.h"
#include <iostream>

struct DebugVisitor : frontends::sql::SqlVisitor {
  void visit(const frontends::sql::ColumnLit &col) override {
    std::cout << "Column: " << col.name << std::endl;
  }
  void visit(const frontends::sql::IntLit &i) override {
    std::cout << "Int: " << i.value << std::endl;
  }
  void visit(const frontends::sql::RealLit &r) override {
    std::cout << "Real: " << r.value << std::endl;
  }
  void visit(const frontends::sql::StrLit &str) override {
    std::cout << "Str: " << str.value << std::endl;
  }
  void visit(const frontends::sql::BoolLit &b) override {
    std::cout << "Bool :" << b.value << std::endl;
  }
  void visit(const frontends::sql::PredicateExpr &p) override {
    std::cout << "Predicate: " << std::endl;
    p.lhs->visit(*this);
    p.rhs->visit(*this);
  }
  void visit(const frontends::sql::CompareExpr &comp) override {
    std::cout << "Compare: " << std::endl;
    comp.lhs->visit(*this);
    comp.rhs->visit(*this);
  }
  void visit(const frontends::sql::ArithmeticOp &arith) override {
    std::cout << "Arith: " << std::endl;
    arith.lhs->visit(*this);
    arith.rhs->visit(*this);
  }
  void visit(const frontends::sql::ProjectionNode &proj) override {
    std::cout << "Proj: " << std::endl;
    for (auto &e : proj.exprs) {
      e->visit(*this);
    }
  }
  void visit(const frontends::sql::FilterNode &filter) override {
    std::cout << "Filter: " << std::endl;
    filter.expr->visit(*this);
  }
  void visit(const frontends::sql::GroupByNode &gb) override {
    std::cout << "Group by: " << std::endl;
    for (auto &f : gb.fields) {
      f->visit(*this);
    }
  }

  void visit(const frontends::sql::SelectNode &select) override {
    std::cout << "Select: " << select.table_name << std::endl;
    if (select.projection) {
      select.projection->visit(*this);
    }
    if (select.filter) {
      select.filter->visit(*this);
    }
    if (select.groub_by) {
      select.groub_by->visit(*this);
    }
  }
};

int main() {
  auto ast = frontends::sql::parse_sql(
      R"(select "b" + 3 from "kek" where "b" > 3 and "c" < 4;)");
  DebugVisitor visitor;
  ast->visit(visitor);
}