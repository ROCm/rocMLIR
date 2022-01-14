add_mlir_dialect_library(MLIRMIOpenTuning
  SqliteDb.cpp
  GridwiseGemmParams.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MIOpen/Tuning

  DEPENDS
  MLIRSupport
)

if (MLIR_MIOPEN_SQLITE_ENABLED)
  # MLIR now depends on SQLite and existence of miopen perfdb
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  set (SQLITE3_FOUND 0)
  find_package(SQLite3)
  if (SQLITE3_FOUND)
    set(MIOPEN_PATH ${ROCM_PATH}/miopen)
    if (NOT MIOPEN_SYSTEM_DB_PATH)
      set(MIOPEN_SYSTEM_DB_PATH "${MIOPEN_PATH}/share/miopen/db/miopen.db")
    endif()
    if (EXISTS "${MIOPEN_SYSTEM_DB_PATH}")
      message(STATUS "miopen system perfdb path: ${MIOPEN_SYSTEM_DB_PATH}")
      add_definitions(-D__MLIR_ENABLE_SQLITE__)
      add_definitions(-DMIOPEN_SYSTEM_DB_PATH="${MIOPEN_SYSTEM_DB_PATH}")
    else ()
      message(STATUS "miopen system perfdb not found, disabling sqlite support")
    endif()
  endif()
endif()

target_include_directories(MLIRMIOpenTuning
  PRIVATE
  ${SQLITE3_INCLUDE_DIRS}
)

target_link_libraries(MLIRMIOpenTuning
  PRIVATE
  MLIRIR
  ${SQLITE3_LIBRARIES}
  ${CMAKE_DL_LIBS}
)
