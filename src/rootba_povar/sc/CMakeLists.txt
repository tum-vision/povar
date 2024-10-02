add_library(rootba_povar_sc linearization_varproj.hpp linearization_power_varproj.hpp)
set_target_properties(rootba_povar_sc PROPERTIES LINKER_LANGUAGE CXX)


target_sources(rootba_povar_sc
  PUBLIC
    linearization_sc.hpp
    landmark_block.hpp
  PRIVATE
)

target_link_libraries(rootba_povar_sc
  PUBLIC
    rootba_povar
)