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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

template <class T>
struct Parse {
  static bool apply(const std::string &s, T &result) {
    std::stringstream ss;
    ss.str(s);
    ss >> result;
    return true;
  }
};

template <class Derived, char Seperator = ','>
struct Serializable {
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
    stream << "v" << static_cast<int32_t>(version) << ":";
    char sep = 0;
    Derived::visit(static_cast<const Derived &>(*this),
                   std::bind(SerializeField{}, std::ref(stream), std::ref(sep),
                             std::placeholders::_1));
  }

  bool checkVersionFormat(const std::string &s) {
    const int32_t maxNumTokens = version == Version::V1 ? 8 : 9;
    const int32_t maxNumSeperators = maxNumTokens - 1;
    const int32_t minNumSeperators = maxNumSeperators - 2;
    const auto numFoundSeperators = std::count_if(
        s.begin(), s.end(), [](char c) { return c == Seperator; });
    return numFoundSeperators >= minNumSeperators &&
           numFoundSeperators <= maxNumSeperators;
  }

  bool deserialize(std::string s) {
    llvm::Regex versionExpr("^v([0-9]+):");
    llvm::SmallVector<llvm::StringRef, 2> matches;
    if (versionExpr.match(s, &matches)) {
      assert(matches.size() == 2 &&
             "a match of the version regex expected 2 items");
      int32_t value = std::stoi(matches[1].str());
      if (value >= static_cast<int32_t>(Version::V1) &&
          value < static_cast<int32_t>(Version::Count)) {
        version = static_cast<Version>(value);
        s = std::string(s.begin() + matches[0].size(), s.end());
      } else {
        // unknown perf config version
        return false;
      }
    } else {
      version = Version::V1;
    }

    if (!checkVersionFormat(s)) {
      // incorrect perf config format
      return false;
    }

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

  enum class Version : int32_t { V1 = 1, V2, Count };
  Version getVersion() { return version; }

protected:
  Version version{Version::V2};
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
