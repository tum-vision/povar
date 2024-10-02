# IPO seems to result in strange crashes in some configuration; also hard to debug even in Debug build...

#include(CheckIPOSupported)
#check_ipo_supported(LANGUAGES CXX RESULT IPO_SUPPORTED OUTPUT IPO_OUTPUT)
#message(STATUS "Link-time optimization support: ${IPO_SUPPORTED}")
#if(IPO_SUPPORTED)
#  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
#else()
#  message(STATUS "${IPO_OUTPUT}")
#endif()

add_executable(bal bal.cpp)
target_link_libraries(bal rootba_povar_solver rootba_povar_cli)

#add_executable(bal_sc bal_sc.cpp)
#target_link_libraries(bal_sc rootba_povar_solver rootba_povar_cli)

