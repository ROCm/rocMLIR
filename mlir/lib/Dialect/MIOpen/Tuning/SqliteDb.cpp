#if __MLIR_ENABLE_SQLITE__

#include "mlir/Dialect/MIOpen/Tuning/SqliteDb.h"
#include "llvm/Support/Debug.h"

#include <experimental/filesystem>
#include <thread>

using namespace mlir;
using llvm::dbgs;

class SQLite::impl {
  struct SQLiteCloser {
    void operator()(sqlite3 *ptr) {
      const auto cFilename = sqlite3_db_filename(ptr, "main");
      std::string filename_((cFilename == nullptr) ? "" : cFilename);
      SQLite::retry([&]() { return sqlite3_close(ptr); }, filename_);
      // Future: Sync the file back to disk, unless disk I/O is disabled
      // Get the page_count: pragma page_count;
      // Get the page_size:  pragma page_size;
      // Buffer size is page_count * page_size
    }
  };
  using Sqlite3Ptr = std::unique_ptr<sqlite3, SQLiteCloser>;
  int createFileDb(const std::experimental::filesystem::path &filepath,
                   bool isSystem) {
    sqlite3 *ptr_tmp = nullptr;
    int rc = 0;
    if (isSystem) {

      rc = sqlite3_open_v2(filepath.string().c_str(), &ptr_tmp,
                           SQLITE_OPEN_READONLY, nullptr);
    } else {
      llvm::errs() << "FATAL ERROR! Does not support user db"
                   << "\n";
    }
    ptrDb = Sqlite3Ptr{ptr_tmp};
    return rc;
  }

public:
  impl(const std::string &filename_, bool isSystem) {
    std::experimental::filesystem::path filepath(filename_);
    int rc = 0;
    rc = createFileDb(filepath, isSystem);
    sqlite3_busy_timeout(ptrDb.get(), MIOPEN_SQL_BUSY_TIMEOUT_MS);
    isValid = (rc == 0);
    if (!isValid) {
      llvm::errs() << "FATAL ERROR! COULD NOT OPEN DB CONNECTION to "
                   << filename_ << "\n";
    } else {
      LLVM_DEBUG(dbgs() << "Successfully opened connection to PerfDb.\n");
    }
  }

  Sqlite3Ptr ptrDb = nullptr;
  bool isValid;
};

static int findCallback(void *_res, int argc, char **argv, char **azColName) {
  SQLite::ResultType *res = static_cast<SQLite::ResultType *>(_res);
  std::unordered_map<std::string, std::string> record;
  for (auto i = 0; i < argc; i++)
    record[azColName[i]] = (argv[i] != nullptr) ? argv[i] : "NULL";
  if (res != nullptr)
    res->push_back(record);
  return 0;
}

SQLite::SQLite() : pImpl(nullptr) {}
SQLite::SQLite(const std::string &filename_, bool isSystem)
    : pImpl{std::make_unique<impl>(filename_, isSystem)} {}
SQLite::~SQLite() = default;
SQLite::SQLite(SQLite &&) noexcept = default;
SQLite &SQLite::operator=(SQLite &&) noexcept = default;
SQLite::ResultType SQLite::exec(const std::string &query) const {
  SQLite::ResultType res;
  {
    auto rc = retry([&]() {
      return sqlite3_exec(pImpl->ptrDb.get(), query.c_str(), findCallback,
                          static_cast<void *>(&res), nullptr);
    });
    if (rc != SQLITE_OK) {
      llvm::errs() << "Query[" << query << "] failed: " << errorMessage();
    }
  }
  return res;
}

int SQLite::retry(std::function<int()> f, std::string filename) {
  int rc = f();
  if (rc == SQLITE_BUSY) {
    llvm::errs() << "Timeout while waiting for Database: " + filename;
  }
  return rc;
}

int SQLite::retry(std::function<int()> f) const {
  std::string filename(sqlite3_db_filename(pImpl->ptrDb.get(), "main"));
  return SQLite::retry(f, filename);
}

int SQLite::changes() const { return sqlite3_changes(pImpl->ptrDb.get()); }

std::string SQLite::errorMessage() const {
  std::string errMsg = "Internal error while accessing SQLite database: ";
  return errMsg + sqlite3_errmsg(pImpl->ptrDb.get());
}
bool SQLite::valid() const { return pImpl->isValid; }

bool DbRecord::setValues(const std::string &id, const std::string &values) {
  // No need to update the file if values are the same:
  const auto it = map.find(id);
  if (it == map.end() || it->second != values) {
    LLVM_DEBUG(dbgs() << key << ", content "
                      << (it == map.end() ? "inserted" : "overwritten") << ": "
                      << id << ':' << values);
    map[id] = values;
    return true;
  }
  LLVM_DEBUG(dbgs() << ", content is the same, not changed:" << id << ':'
                    << values);
  return false;
}

bool DbRecord::getValues(const std::string &id, std::string &values) const {
  const auto it = map.find(id);

  if (it == map.end()) {
    LLVM_DEBUG(dbgs() << '=' << id << ':' << "<values not found>"
                      << "\n");
    return false;
  }

  values = it->second;
  LLVM_DEBUG(dbgs() << key << '=' << id << ':' << values << "\n");
  return true;
}

SQLitePerfDb::SQLitePerfDb(const std::string &filename_, bool isSystem,
                           const std::string &arch_, std::size_t num_cu_)
    : filename(filename_), arch(arch_), num_cu(num_cu_) {
  if (filename.empty()) {
    dbInvalid = true;
    return;
  }

  sql = SQLite{filename_, isSystem};
  if (!sql.valid()) {
    dbInvalid = true;
  } else {
    dbInvalid = false;
  }
}

SQLitePerfDb mlir::getDb(const llvm::SmallString<8> &arch, std::size_t num_cu) {
  // DB path: "/opt/rocm/miopen/share/miopen/db/miopen.db"
  return {MIOPEN_SYSTEM_DB_PATH, true, std::string(arch), num_cu};
}

#endif // MLIR_ENABLE_SQLITE
