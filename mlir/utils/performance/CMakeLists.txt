add_subdirectory(common)
add_subdirectory(rocblas-benchmark-driver)
add_subdirectory(ck-benchmark-driver)

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

list(TRANSFORM PERFORMANCE_SCRIPTS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)

add_custom_target(ci-performance-scripts
	COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_SCRIPTS} ${ROCMLIR_BIN_DIR})
