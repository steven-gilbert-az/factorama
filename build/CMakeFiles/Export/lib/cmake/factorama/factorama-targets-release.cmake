#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "factorama::factorama" for configuration "Release"
set_property(TARGET factorama::factorama APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(factorama::factorama PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libfactorama.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS factorama::factorama )
list(APPEND _IMPORT_CHECK_FILES_FOR_factorama::factorama "${_IMPORT_PREFIX}/lib/libfactorama.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
