# Python bindings for Factorama using pybind11

find_package(pybind11 REQUIRED)

pybind11_add_module(_factorama
    factorama/_factorama.cpp
)

target_link_libraries(_factorama PRIVATE factorama)

# Set properties for better Python integration
target_compile_definitions(_factorama PRIVATE VERSION_INFO=${PROJECT_VERSION})