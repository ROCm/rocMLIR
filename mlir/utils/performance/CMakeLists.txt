set(PERFORMANCE_SCRIPTS
  createPerformanceReports.py
  MIOpenDriver.py
  parameterSweeps.py
  perfRegressionReport.py
  reportUtils.py
)

list(TRANSFORM PERFORMANCE_SCRIPTS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

add_custom_target(ci-performance-scripts
	COMMAND ${CMAKE_COMMAND} -E copy ${PERFORMANCE_SCRIPTS} ${ROCMLIR_BIN_DIR})
