add_lld_unittests(LLDAsLibELFTests
  ROCm.cpp
  SomeDrivers.cpp
)

target_link_libraries(LLDAsLibELFTests
  PRIVATE
  lldCommon
  lldELF
)
