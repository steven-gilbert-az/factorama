# Python bindings for Factorama using pybind11

find_package(pybind11 REQUIRED)

# Generate docstring headers from Doxygen XML if available
set(DOCSTRING_FILE "${CMAKE_CURRENT_BINARY_DIR}/docstring.hpp")

# Try to find pybind11-mkdoc (works even without BUILD_DOCS)
find_program(PYBIND11_MKDOC pybind11-mkdoc)
if(PYBIND11_MKDOC)
    # Get all header files
    file(GLOB_RECURSE FACTORAMA_HEADERS "${CMAKE_SOURCE_DIR}/src/factorama/*.hpp")

    # Extract docstrings from headers
    add_custom_command(
        OUTPUT ${DOCSTRING_FILE}
        COMMAND ${PYBIND11_MKDOC}
            -I"${CMAKE_SOURCE_DIR}/src"
            $<$<TARGET_EXISTS:Eigen3::Eigen>:-I"$<TARGET_PROPERTY:Eigen3::Eigen,INTERFACE_INCLUDE_DIRECTORIES>">
            ${FACTORAMA_HEADERS}
            -o "${DOCSTRING_FILE}"
        DEPENDS ${FACTORAMA_HEADERS}
        COMMENT "Extracting docstrings from C++ headers"
        VERBATIM
    )
    add_custom_target(generate_docstrings DEPENDS ${DOCSTRING_FILE})
    set(HAVE_DOCSTRINGS TRUE)
    message(STATUS "pybind11-mkdoc found: ${PYBIND11_MKDOC}")
else()
    message(STATUS "pybind11-mkdoc not found. Python bindings will have minimal docstrings.")
    set(HAVE_DOCSTRINGS FALSE)
endif()

# Create empty docstring file if not generating
if(NOT HAVE_DOCSTRINGS)
    file(WRITE ${DOCSTRING_FILE} "// Empty docstring file\n")
endif()

pybind11_add_module(_factorama
    factorama/_factorama.cpp
)

target_link_libraries(_factorama PRIVATE factorama)

# Add docstring include path
target_include_directories(_factorama PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Add dependency on docstring generation
if(HAVE_DOCSTRINGS)
    add_dependencies(_factorama generate_docstrings)
endif()

# Ensure the .so lands inside the Python package
set_target_properties(_factorama PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python_bindings/python/factorama"
)

# Set properties for better Python integration
target_compile_definitions(_factorama PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Install the extension module for wheel building
install(TARGETS _factorama
    LIBRARY DESTINATION factorama
    ARCHIVE DESTINATION factorama
    RUNTIME DESTINATION factorama
)