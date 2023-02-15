//===------- SqliteDb.h - MLIR sqlite database client ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR sqlite db client
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_SQLITEDB_H
#define MLIR_DIALECT_ROCK_SQLITEDB_H

#if __MLIR_ENABLE_SQLITE__

#include "sqlite3.h"
#include "llvm/ADT/Any.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>
#include <unordered_map>
#include <vector>

namespace mlir {

const auto ROCK_SQL_BUSY_TIMEOUT_MS = 60000;

class SQLite {
public:
  class impl;
  std::unique_ptr<impl> pImpl;

  using ResultType = std::vector<std::unordered_map<std::string, std::string>>;
  SQLite();
  SQLite(const std::string &filename_, bool is_system);
  ~SQLite();
  SQLite(SQLite &&) noexcept;
  SQLite &operator=(SQLite &&) noexcept;
  SQLite &operator=(const SQLite &) = delete;
  bool valid() const;
  ResultType exec(const std::string &query) const;
  int changes() const;
  int retry(std::function<int()>) const;
  static int retry(std::function<int()> f, std::string filename);
  std::string errorMessage() const;
};

class DbRecord {
#define DEBUG_TYPE "rock-sqlite-dbrecord"

public:
  bool getValues(const std::string &id, std::string &values) const;
  bool setValues(const std::string &id, const std::string &values);
  auto getSize() const { return map.size(); }

  /// Get VALUES associated with ID under the current KEY and delivers those to
  /// a member function of a class T object. T shall have the "bool
  /// deserialize(const std::string& str)" member function available.
  ///
  /// Returns false if there is none ID:VALUES in the record or in case of any
  /// error, e.g. if VALUES cannot be deserialized due to incorrect format.
  template <class T> bool getValues(const std::string &id, T &values) const {
    std::string s;
    if (!getValues(id, s))
      return false;

    const bool ok = values.deserialize(s);
    if (!ok)

      LLVM_DEBUG(llvm::dbgs() << "Perf db record is obsolete or corrupt: " << s
                              << ". Performance may degrade.\n");
    return ok;
  }

private:
  std::string key;
  std::unordered_map<std::string, std::string> map;

#undef DEBUG_TYPE
};

template <class Vector, class T> void printVector(Vector v) {
  std::for_each(v.begin(), v.end(),
                [](T item) { llvm::outs() << " " << item << " "; });
  llvm::outs() << "\n";
}

class SQLitePerfDb {
#define DEBUG_TYPE "rock-sqlite-perfdb"

public:
  std::string filename;
  std::string arch;
  size_t num_cu;
  bool dbInvalid;
  SQLite sql;

  SQLitePerfDb(const std::string &filename_, bool is_system,
               const std::string &arch_, std::size_t num_cu_);

  template <typename T>
  inline std::optional<DbRecord> findRecord(const T &problemConfig) {
    if (dbInvalid)
      return {};

    std::string query = problemConfig.queryClause();
    // clang-format off
    auto select_query =
        "SELECT solver, params "
        "FROM perf_db "
        "INNER JOIN " + problemConfig.tableName() + " "
        "ON perf_db.config = " + problemConfig.tableName() +".id "
        "WHERE "
        "( " + query + " )"
        "AND (arch = '" + arch + "' ) "
        "AND (num_cu = '" + std::to_string(num_cu) + "');";
      LLVM_DEBUG(llvm::dbgs() << "SQLite Query: " << select_query << "\n");
      
      SQLite::ResultType execRes = sql.exec(select_query);
      DbRecord rec;
      std::for_each(execRes.begin(), execRes.end(), 
              [&rec, &execRes](auto row){
              rec.setValues(row["solver"], row["params"]);
              if (row == execRes.back()) LLVM_DEBUG(llvm::dbgs() << "\n");
              });

      if (rec.getSize() == 0)
        return {};
      else
        return std::optional<DbRecord>(rec);
    }

    template <class T, class V>
    inline bool load(const T& problemConfig, const std::string& id, V& values)
    {
        if(dbInvalid)
            return false;
        const auto record = findRecord(problemConfig);

        if(!record)
            return false;
        return record->getValues(id, values);
    }

#undef DEBUG_TYPE
};

SQLitePerfDb getDb(const llvm::SmallString<8> &arch, size_t num_cu);
} // namespace MLIR

#endif // MLIR_ENABLE_SQLITE
#endif // MLIR_DIALECT_ROCK_SQLITE_DB_H
