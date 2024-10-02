cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
# Note: 3.13 for target_sources with relative paths

add_library(rootba_povar)

target_include_directories(rootba_povar PUBLIC "${CMAKE_SOURCE_DIR}/src")
target_compile_features(rootba_povar PUBLIC cxx_std_${ROOTBA_CXX_STANDARD})
set_target_properties(rootba_povar PROPERTIES CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # don't use pedantic for GCC, since it warns on extra ';' --> fix wise enum
  target_compile_options(rootba_povar PUBLIC -Wpedantic)
endif()
target_compile_options(rootba_povar PUBLIC -ftemplate-backtrace-limit=0 -Wall -Wextra -Wnon-virtual-dtor -Wsign-compare -Wno-unused-parameter -march=${CXX_MARCH})
target_compile_definitions(rootba_povar PUBLIC ${ROOTBA_COMPILE_DEFINITIONS})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # offsetof is fine even for non POD types...
  target_compile_options(rootba_povar PUBLIC -Wno-invalid-offsetof)
endif()

target_link_libraries(rootba_povar
  PUBLIC
    rootba_povar::Sophus
    rootba_povar::visit_struct
    rootba_povar::wise_enum
    rootba_povar::enum_flags
    rootba_povar::nameof
    rootba_povar::toml11
    rootba_povar::nlohmann_json
    #rootba_povar::basalt_headers
    rootba_povar::magic_enum
    rootba_povar::pprint
    rootba_povar::Cereal
    fmt::fmt
    absl::flat_hash_map
    Eigen3::Eigen
    glog::glog
    TBB::tbb
)

add_subdirectory(testing)
add_subdirectory(util)
add_subdirectory(options)
add_subdirectory(bal)
add_subdirectory(sc)
add_subdirectory(solver)
add_subdirectory(cg)
add_subdirectory(cli)
