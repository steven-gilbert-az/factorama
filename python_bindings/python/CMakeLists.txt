# Python bindings for Factorama using pybind11

find_package(pybind11 REQUIRED)

pybind11_add_module(_factorama
    factorama/_factorama.cpp
)

target_link_libraries(_factorama PRIVATE factorama)

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