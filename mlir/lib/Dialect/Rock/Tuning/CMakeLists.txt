add_mlir_dialect_library(MLIRRockTuning
  ConvContext.cpp
  GemmContext.cpp
  SqliteDb.cpp
  GridwiseGemmParams.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rock/Tuning

  DEPENDS
  MLIRSupport
  MLIRRockPassIncGen
)

if (ROCMLIR_SQLITE_ENABLED)
  # MLIR now depends on SQLite and existence of rock perfdb
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  set (SQLITE3_FOUND 0)
  find_package(SQLite3)
  if (SQLITE3_FOUND)
    set(ROCK_PATH ${ROCM_PATH}/rock)
    if (NOT ROCK_SYSTEM_DB_PATH)
      set(ROCK_SYSTEM_DB_PATH "${ROCK_PATH}/share/rock/db/rock.db")
    endif()
    if (EXISTS "${ROCK_SYSTEM_DB_PATH}")
      message(STATUS "rock system perfdb path: ${ROCK_SYSTEM_DB_PATH}")
      add_definitions(-D__MLIR_ENABLE_SQLITE__)
      add_definitions(-DROCK_SYSTEM_DB_PATH="${ROCK_SYSTEM_DB_PATH}")
    else ()
      message(STATUS "rock system perfdb not found, disabling sqlite support")
    endif()
  endif()
endif()

target_include_directories(MLIRRockTuning
  PRIVATE
  ${SQLITE3_INCLUDE_DIRS}
)

target_link_libraries(MLIRRockTuning
  PRIVATE
  MLIRRockUtility
  MLIRIR
  ${SQLITE3_LIBRARIES}
  ${CMAKE_DL_LIBS}
)
