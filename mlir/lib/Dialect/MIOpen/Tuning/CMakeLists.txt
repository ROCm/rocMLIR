add_mlir_dialect_library(MLIRMIOpenTuning
  SqliteDb.cpp
  GridwiseGemmParams.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen/Tuning

  DEPENDS
  MLIRSupport
)

if (MLIR_ENABLE_SQLITE)
    # MLIR now depends on SQLite and existence of miopen perfdb
    set (SQLITE3_FOUND 0)
    find_package(PkgConfig)
    pkg_check_modules(SQLITE3 REQUIRED sqlite3)
    if (SQLITE3_FOUND)
      set(MIOPEN_PATH ${ROCM_PATH}/miopen)
      set(MIOPEN_SYSTEM_DB_PATH "${MIOPEN_PATH}/share/miopen/db/miopen.db")
      if (EXISTS "${MIOPEN_SYSTEM_DB_PATH}")
        message("miopen system perfdb path: ${MIOPEN_SYSTEM_DB_PATH}")
        add_definitions(-D__MLIR_ENABLE_SQLITE__)
        add_definitions(-DMIOPEN_SYSTEM_DB_PATH="${MIOPEN_SYSTEM_DB_PATH}")
      else ()
        message(STATUS "miopen system perfdb not found, disabling sqlite support")
      endif()
    endif()
endif()

target_include_directories(MLIRMIOpenTuning 
  PRIVATE 
  ${SQLITE3_STATIC_INCLUDE_DIRS}
)

target_compile_options(MLIRMIOpenTuning 
  PRIVATE 
  ${SQLITE3_STATIC_CFLAGS}
)

target_link_libraries(MLIRMIOpenTuning
  PRIVATE
  MLIRIR
  stdc++fs
  ${SQLITE3_STATIC_LDFLAGS}
)
