//===------- Serializable.h - MLIR serializable targets from perfdb--------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR serializable items
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_SERIALIZABLE_H
#define MLIR_DIALECT_MIOPEN_SERIALIZABLE_H

#include <numeric>
#include <vector>

template <class T> struct Parse {
  static bool apply(const std::string &s, T &result) {
    std::stringstream ss;
    ss.str(s);
    ss >> result;
    return true;
  }
};

template <class Derived, char Seperator = ','> struct Serializable {
  struct SerializeField {
    template <class T>
    void operator()(std::ostream &stream, char &sep, const T &x) const {
      if (sep != 0)
        stream << sep;
      stream << x;
      sep = Seperator;
    }
  };

  struct DeserializeField {
    template <class T>
    void operator()(bool &ok, std::istream &stream, char sep, T &x) const {
      if (not ok)
        return;
      std::string part;

      if (!std::getline(stream, part, sep)) {
        ok = false;
        return;
      }

      ok = Parse<T>::apply(part, x);
    }
  };
  void serialize(std::ostream &stream) const {
    char sep = 0;
    Derived::visit(static_cast<const Derived &>(*this),
                   std::bind(SerializeField{}, std::ref(stream), std::ref(sep),
                             std::placeholders::_1));
  }

  bool deserialize(const std::string &s) {
    auto out = static_cast<const Derived &>(*this);
    bool ok = true;
    std::istringstream ss(s);
    Derived::visit(out,
                   std::bind(DeserializeField{}, std::ref(ok), std::ref(ss),
                             Seperator, std::placeholders::_1));

    if (!ok)
      return false;

    static_cast<Derived &>(*this) = out;
    return true;
  }

  friend std::ostream &operator<<(std::ostream &os, const Derived &c) {
    c.serialize(os);
    return os;
  }
};

template <class Strings>
inline std::string joinStrings(Strings strings, std::string delim) {
  auto it = strings.begin();
  if (it == strings.end())
    return "";

  auto nit = std::next(it);
  return std::accumulate(
      nit, strings.end(), *it,
      [&](std::string x, std::string y) { return x + delim + y; });
}

template <class Derived> struct SQLiteSerializable {
  std::vector<std::string> fieldNames() const {
    std::vector<std::string> names;
    Derived::visit(static_cast<const Derived &>(*this),
                   [&](const std::string &value, const std::string &name) {
                     std::ignore = value;
                     names.push_back(name);
                   });
    return names;
  }
  std::string queryClause() const {
    std::vector<std::string> clauses;
    Derived::visit(static_cast<const Derived &>(*this),
                   [&](const std::string &value, const std::string &name) {
                     clauses.push_back("(" + name + " = " + value + " )");
                   });
    std::string clause = joinStrings(clauses, " AND ");
    return clause;
  }
};

#endif // MLIR_DIALECT_MIOPEN_SERIALIZABLE_H
