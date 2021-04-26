#include "arcise/dialects/relalg/frontends/sql/Parser.h"
#include "arcise/dialects/relalg/frontends/sql/AST.h"
#include <cctype>
#include <stdexcept>

#include <boost/phoenix.hpp>
#include <boost/phoenix/bind/bind_member_variable.hpp>
#include <boost/spirit/include/qi.hpp>

namespace arcise::dialects::relalg::frontends::sql {

namespace qi = boost::spirit::qi;

template <typename Parser, typename... Args>
void parse_or_die(const std::string &input, const Parser &p, Args &&...args) {
  std::string::const_iterator begin = input.begin(), end = input.end();
  bool ok = qi::parse(begin, end, p, std::forward<Args>(args)...);
  if (!ok || begin != end) {
    throw std::runtime_error("Parse error");
  }
}

template <typename T> struct make_shared_f {
  template <typename... A> struct result { typedef std::shared_ptr<T> type; };

  template <typename... A>
  typename result<A...>::type operator()(A &&...a) const {
    return std::make_shared<T>(std::forward<A>(a)...);
  }
};

template <typename T>
using make_shared_ = boost::phoenix::function<make_shared_f<T>>;

template <typename Iter>
struct SqlGrammar
    : qi::grammar<Iter, std::shared_ptr<SelectNode>(), qi::ascii::space_type> {
  qi::rule<Iter, std::shared_ptr<SelectNode>(), qi::ascii::space_type> start;
  qi::rule<Iter, std::shared_ptr<ProjectionNode>(), qi::ascii::space_type>
      projection;
  qi::rule<Iter, std::shared_ptr<FilterNode>(), qi::ascii::space_type> filter;
  qi::rule<Iter, std::shared_ptr<JoinNode>(), qi::ascii::space_type> join;
  qi::rule<Iter, std::shared_ptr<ScanNode>(), qi::ascii::space_type> scan;
  qi::rule<Iter, std::shared_ptr<ColumnLit>(), qi::ascii::space_type>
      column_name;
  qi::rule<Iter, std::shared_ptr<ArithmeticExpr>(), qi::ascii::space_type>
      basic_arith_expr;
  qi::rule<Iter, std::shared_ptr<ArithmeticOp>(), qi::ascii::space_type>
      binary_arith_expr;
  qi::rule<Iter, std::shared_ptr<ArithmeticExpr>(), qi::ascii::space_type>
      arithmetic_expr;
  qi::rule<Iter, ArithmeticOp::Op(), qi::ascii::space_type> arithmetic_op;
  qi::rule<Iter, std::string(), qi::ascii::space_type> quoted_string;
  qi::rule<Iter, std::shared_ptr<BooleanExpr>(), qi::ascii::space_type>
      boolean_expr;
  qi::rule<Iter, std::shared_ptr<BoolLit>(), qi::ascii::space_type> bool_lit;
  qi::rule<Iter, std::shared_ptr<PredicateExpr>(), qi::ascii::space_type>
      binary_predicate;
  qi::rule<Iter, PredicateExpr::Op(), qi::ascii::space_type> predicate_op;

  qi::rule<Iter, std::shared_ptr<CompareExpr>(), qi::ascii::space_type>
      compare_expr;

  qi::rule<Iter, std::shared_ptr<SelectNode>(), qi::ascii::space_type> select;

  qi::rule<Iter, CompareExpr::Op(), qi::ascii::space_type> compare_op;

  SqlGrammar(bool debug) : SqlGrammar::base_type(start) {
    using qi::_1;
    using qi::_val;

    using boost::phoenix::bind;
    namespace phoenix = boost::phoenix;
    // clang-format off
    start %= select >> ";";

    select = "select" >> qi::eps [_val = make_shared_<SelectNode>()()]
        >> projection [bind(&SelectNode::projection, _val) =  _1] 
        >> "from"
        >> scan [bind(&SelectNode::scan, _val) = _1] 
        >> -filter [bind(&SelectNode::filter, _val) = _1] 
        >> -join [bind(&SelectNode::join, _val) = _1];

    projection =
        qi::eps [_val = make_shared_<ProjectionNode>()()] >>
            arithmetic_expr [phoenix::push_back(bind(&ProjectionNode::exprs, _val), _1)] % ',';

    quoted_string %= qi::lexeme['"' >> +(qi::char_ - '"') >> '"'];

    column_name = quoted_string [_val = make_shared_<ColumnLit>()(_1)];

    scan = quoted_string [_val = make_shared_<TableScanNode>()(_1)]
        || ("(" >> select [_val = _1] >> ")");

    basic_arith_expr = 
        (qi::int_ [_val = make_shared_<IntLit>()(_1)])
        ||  (qi::double_ [_val = make_shared_<RealLit>()(_1)])
        ||  (column_name [_val = _1]);

    arithmetic_op = 
        ('+' >> qi::eps [_val = ArithmeticOp::Op::Sum])
        ||  ('-' >> qi::eps [_val = ArithmeticOp::Op::Sub])
        ||  ('*' >> qi::eps [_val = ArithmeticOp::Op::Mul])
        ||  ('/' >> qi::eps [_val = ArithmeticOp::Op::Div]);

    binary_arith_expr =
        qi::eps [_val = make_shared_<ArithmeticOp>()()] 
        >> basic_arith_expr [bind(&ArithmeticOp::lhs, _val) = _1]
        >> arithmetic_op [bind(&ArithmeticOp::op, _val) = _1]
        >> arithmetic_expr [bind(&ArithmeticOp::rhs, _val) = _1];

    arithmetic_expr = binary_arith_expr [_val = _1] || basic_arith_expr [_val = _1];

    filter = qi::eps [_val = make_shared_<FilterNode>()()] 
                >> "where"
                >> boolean_expr [bind(&FilterNode::expr, _val) = _1]; 

    bool_lit = ("true" >> qi::eps [_val = make_shared_<BoolLit>()(true)]) 
                  || ("false" >> qi::eps[_val = make_shared_<BoolLit>()(false)]);

    compare_expr = qi::eps [_val = make_shared_<CompareExpr>()()] 
        >> basic_arith_expr [bind(&CompareExpr::lhs, _val) = _1] 
        >> compare_op [bind(&CompareExpr::op, _val) = _1] 
        >> basic_arith_expr [bind(&CompareExpr::rhs, _val) = _1];
    
    compare_op = (">" >> qi::eps [_val = CompareExpr::Op::GT])
                    || ("<" >> qi::eps [_val = CompareExpr::Op::LT])
                    || (">=" >> qi::eps [_val = CompareExpr::Op::GE])
                    || ("<=" >> qi::eps [_val = CompareExpr::Op::LE])
                    || ("==" >> qi::eps [_val = CompareExpr::Op::EQ])
                    || ("!=" >> qi::eps [_val = CompareExpr::Op::NEQ]);

    predicate_op = ("and" >> qi::eps [_val = PredicateExpr::Op::AND])
                      || ("or" >> qi::eps [_val = PredicateExpr::Op::OR]);

    binary_predicate = qi::eps [_val = make_shared_<PredicateExpr>()()]
      >> ((compare_expr [bind(&PredicateExpr::lhs, _val) = _1])
            || (bool_lit [bind(&PredicateExpr::lhs, _val) = _1])) 
      >> predicate_op [bind(&PredicateExpr::op, _val) = _1] 
      >> boolean_expr [bind(&PredicateExpr::rhs, _val) = _1]; 
    
    boolean_expr = (binary_predicate [_val = _1]) || (compare_expr [_val = _1]);

    join = "join" >> qi::eps[_val = make_shared_<JoinNode>()()] 
      >> scan [bind(&JoinNode::scan, _val) = _1]
      >> "on" 
      >> column_name [bind(&JoinNode::lhs, _val) = _1] 
        >> "=" 
      >> column_name [bind(&JoinNode::rhs, _val) = _1];

    // clang-format on
    if (debug) {

      start.name("start");
      projection.name("projection");
      quoted_string.name("quoted_string");
      column_name.name("column_name");
      basic_arith_expr.name("basic_arith_expr");
      arithmetic_expr.name("arithmetic_expr");
      binary_arith_expr.name("binary_arith_expr");
      filter.name("filter");
      bool_lit.name("bool_lit");
      compare_expr.name("compare_expr");
      binary_predicate.name("binary_predicate");
      boolean_expr.name("boolean_expr");

      qi::debug(start);
      qi::debug(projection);
      qi::debug(quoted_string);
      qi::debug(column_name);
      qi::debug(basic_arith_expr);
      qi::debug(binary_arith_expr);
      qi::debug(arithmetic_expr);
      qi::debug(filter);
      qi::debug(bool_lit);
      qi::debug(compare_expr);
      qi::debug(binary_predicate);
      qi::debug(boolean_expr);
    }
  }
};

std::shared_ptr<SqlNode> parse_sql(std::string_view sql, bool debug) {
  SqlGrammar<std::string_view::iterator> grammar(debug);
  std::shared_ptr<SqlNode> res;
  auto iter = sql.begin();
  if (!qi::phrase_parse(iter, sql.end(), grammar, qi::ascii::space, res)) {
    throw std::runtime_error("Unable to parse sql");
  }
  return res;
}
} // namespace arcise::dialects::relalg::frontends::sql