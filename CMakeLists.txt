cmake_minimum_required(VERSION 3.16)
project(factorama VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compile options for better debugging and warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(GTSAM "Enable building of GTSAM example" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)

# Find required packages
# Try to find system Eigen first
find_package(Eigen3 3.4 QUIET NO_MODULE)

if (Eigen3_FOUND)
    message(STATUS "Found system Eigen3: ${Eigen3_DIR}")
else()
    message(WARNING "System Eigen3 not found. Using vendored Eigen3.")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    FetchContent_MakeAvailable(eigen)
    find_package(Eigen3 REQUIRED NO_MODULE)
endif()

# Find Catch2 only if building tests
if(BUILD_TESTS)
    find_package(Catch2 3 REQUIRED)
    enable_testing()
endif()


# === Library target ===
add_library(factorama)
target_sources(factorama
    PRIVATE
        src/factorama/bearing_observation_factor.cpp
        src/factorama/bearing_projection_factor_2d.cpp
        src/factorama/factor_graph.cpp
        src/factorama/inverse_range_bearing_factor.cpp  
        src/factorama/sparse_optimizer.cpp
        src/factorama/pose_between_factors.cpp
)

# Set include directories for the library
target_include_directories(factorama
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

# Link libraries
target_link_libraries(factorama
    PUBLIC
        Eigen3::Eigen
)

# Set compile features
target_compile_features(factorama PUBLIC cxx_std_17)

# Enable position-independent code for Python bindings
set_target_properties(factorama PROPERTIES POSITION_INDEPENDENT_CODE ON)




# === Examples ===
if(BUILD_EXAMPLES)

    add_executable(optimize src/factorama_test/sparse_optimize.cpp)
    target_link_libraries(optimize PRIVATE factorama)

    add_executable(example1 examples/01_simple_localization.cpp)
    target_link_libraries(example1 PRIVATE factorama)
    
    add_executable(example2 examples/02_landmark_triangulation.cpp)
    target_link_libraries(example2 PRIVATE factorama)
endif()


# === GTSAM support ===
if(GTSAM)
    message(STATUS "Building GTSAM benchmark test...")
    
    find_package(GTSAM REQUIRED)
    
    add_executable(gtsam_benchmark src/factorama_test/gtsam_benchmark.cpp)
    
    target_link_libraries(gtsam_benchmark
        PRIVATE
            gtsam
            factorama
    )
    
    target_compile_features(gtsam_benchmark PRIVATE cxx_std_17)
endif()

# === Tests ===
if(BUILD_TESTS)
    include(CTest)
    
    # Helper function to create test executables
    function(add_factorama_test test_name test_file)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name}
            PRIVATE
                factorama
                Catch2::Catch2WithMain
        )
        target_compile_features(${test_name} PRIVATE cxx_std_17)
    endfunction()
    
    # Create test executables
    add_factorama_test(unit_test src/factorama_test/unit_test.cpp)
    add_factorama_test(integration_test src/factorama_test/integration_test.cpp)
    add_factorama_test(optimizer_test src/factorama_test/optimizer_test.cpp)
    
    # Solver benchmark executable (standalone, not a Catch2 test)
    add_executable(solver_benchmark src/factorama_test/solver_benchmark.cpp)
    target_link_libraries(solver_benchmark
        PRIVATE
            factorama
    )
    target_compile_features(solver_benchmark PRIVATE cxx_std_17)
    
    # Discover and register tests
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.10)
        include(Catch)
        catch_discover_tests(unit_test)
        catch_discover_tests(integration_test)
        catch_discover_tests(optimizer_test)
    else()
        add_test(NAME unit_test COMMAND unit_test)
        add_test(NAME integration_test COMMAND integration_test)
        add_test(NAME optimizer_test COMMAND optimizer_test)
    endif()
endif()

# === Installation ===
include(GNUInstallDirs)

# Install library target
install(TARGETS factorama
    EXPORT factorama-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY src/factorama/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/factorama
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install export targets
install(EXPORT factorama-targets
    FILE factorama-targets.cmake
    NAMESPACE factorama::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/factorama
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    factorama-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/factorama-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/factorama-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/factorama
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/factorama-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/factorama-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/factorama
)

# === Python bindings ===
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings/python)
endif()