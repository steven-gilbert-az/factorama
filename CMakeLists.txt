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

# Print helpful build type information
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build Type: Debug - Full debug symbols, no optimization (slower but best for debugging)")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build Type: Release - Full optimization, no debug symbols (fastest performance)")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Build Type: RelWithDebInfo - Optimized with debug symbols (good performance with debugging capability)")
else()
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
endif()

# Size optimization for Release builds only
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # add_compile_options(-ffunction-sections)
    add_compile_options(-ffunction-sections -fdata-sections)
    add_link_options(-Wl,--gc-sections -s)
    message(STATUS "Adding size optimization flags to Release build")
endif()

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)
option(BUILD_DOCS "Build documentation with Doxygen" OFF)


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
        src/factorama/bearing_observation_factor_2d.cpp
        src/factorama/bearing_projection_factor_2d.cpp
        src/factorama/factor_graph.cpp
        src/factorama/generic_between_factor.cpp
        src/factorama/inverse_range_bearing_factor.cpp
        src/factorama/inverse_range_variable.cpp
        src/factorama/linear_velocity_factor.cpp
        src/factorama/plane_factor.cpp
        src/factorama/plane_prior_factor.cpp
        src/factorama/pose_2d_between_factor.cpp
        src/factorama/pose_2d_prior_factor.cpp
        src/factorama/pose_2d_variable.cpp
        src/factorama/pose_prior_factors.cpp
        src/factorama/pose_variable.cpp
        src/factorama/range_bearing_factor_2d.cpp
        src/factorama/sparse_optimizer.cpp
        src/factorama/pose_between_factors.cpp
        src/factorama/random_utils.cpp
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

    add_executable(random_bench examples/random_benchmarks.cpp)
    target_link_libraries(random_bench PRIVATE factorama)

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
    add_factorama_test(variable_tests src/factorama_test/variable_tests.cpp)
    add_factorama_test(factor_tests src/factorama_test/factor_tests.cpp)
    add_factorama_test(integration_test src/factorama_test/integration_test.cpp)
    add_factorama_test(optimizer_test src/factorama_test/optimizer_test.cpp)
    add_factorama_test(stopwatch_test src/factorama_test/stopwatch_test.cpp)
    
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
        catch_discover_tests(variable_tests)
        catch_discover_tests(factor_tests)
        catch_discover_tests(integration_test)
        catch_discover_tests(optimizer_test)
        catch_discover_tests(stopwatch_test)
    else()
        add_test(NAME unit_test COMMAND unit_test)
        add_test(NAME variable_tests COMMAND variable_tests)
        add_test(NAME factor_tests COMMAND factor_tests)
        add_test(NAME integration_test COMMAND integration_test)
        add_test(NAME optimizer_test COMMAND optimizer_test)
        add_test(NAME stopwatch_test COMMAND stopwatch_test)
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

# === Documentation ===
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    if(Doxygen_FOUND)
        # Configure Doxyfile
        configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)

        # Add Doxygen target
        add_custom_target(doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating Doxygen XML"
            VERBATIM
        )

        # Find Sphinx
        find_program(SPHINX_EXECUTABLE sphinx-build)
        if(SPHINX_EXECUTABLE)
            # Add Sphinx target
            add_custom_target(sphinx
                COMMAND ${SPHINX_EXECUTABLE} -b html
                    ${CMAKE_SOURCE_DIR}/docs
                    ${CMAKE_BINARY_DIR}/docs/sphinx
                DEPENDS doxygen
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Building unified documentation with Sphinx"
                VERBATIM
            )

            # Main docs target builds both
            add_custom_target(docs DEPENDS sphinx)

            message(STATUS "Documentation targets configured:")
            message(STATUS "  'make doxygen' - Generate Doxygen XML")
            message(STATUS "  'make sphinx'  - Build unified docs with Sphinx")
            message(STATUS "  'make docs'    - Build complete documentation")
            message(STATUS "Unified docs will be in: ${CMAKE_BINARY_DIR}/docs/sphinx/")
        else()
            # Fallback to Doxygen only
            add_custom_target(docs DEPENDS doxygen)
            message(STATUS "Sphinx not found. Install with: pip install sphinx breathe sphinx-rtd-theme")
            message(STATUS "Doxygen-only documentation will be in: ${CMAKE_BINARY_DIR}/docs/html/")
        endif()
    else()
        message(WARNING "Doxygen not found. Documentation will not be built.")
    endif()
endif()