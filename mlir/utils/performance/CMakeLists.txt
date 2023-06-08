add_subdirectory(common)
if("rocblas" IN_LIST ROCMLIR_ENABLE_BENCHMARKS)
  add_subdirectory(rocblas-benchmark-driver)
endif()
if("ck" IN_LIST ROCMLIR_ENABLE_BENCHMARKS)
  add_subdirectory(ck-benchmark-driver)
endif()

set(PERFORMANCE_SCRIPTS
  convertRocBlasToPerfRunner.py
  createGemmPerformanceReports.py
  createFusionPerformanceReports.py
  createPerformanceReports.py
  perfCommonUtils.py
  perfRunner.py
  parameterSweeps.py
  perfRegressionReport.py
  reportUtils.py
  tuningRunner.py
)

list(TRANSFORM PERFORMANCE_SCRIPTS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

add_custom_target(ci-performance-scripts
	COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_SCRIPTS} ${ROCMLIR_BIN_DIR})
