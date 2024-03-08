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

#ifndef MLIR_DIALECT_ROCK_SERIALIZABLE_H
#define MLIR_DIALECT_ROCK_SERIALIZABLE_H

#include <algorithm>
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

    const auto numCommas =
        std::count_if(s.begin(), s.end(), [](char c) { return c == ','; });
    if (numCommas != 8) {
      // string is supposed to contain 9 integers separated by ','.
      // Thus, one should expect to see 8 commas
      return false;
    }

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

#endif // MLIR_DIALECT_ROCK_SERIALIZABLE_H
