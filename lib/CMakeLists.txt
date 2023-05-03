add_subdirectory(Dialect)

if(MHAL_ENABLE_TRANSFORMS)
  add_subdirectory(Conversion)
  add_subdirectory(Support)
endif()
