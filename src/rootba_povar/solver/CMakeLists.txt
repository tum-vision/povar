add_library(rootba_povar_solver linearizor_power_varproj.cpp linearizor_power_varproj.hpp)


target_sources(rootba_povar_solver
  PUBLIC
    bal_bundle_adjustment.hpp
    linearizor.hpp
    linearizor_base.hpp
    linearizor_sc.hpp
    solver_summary.hpp
  PRIVATE
    bal_bundle_adjustment.cpp
    linearizor.cpp
    linearizor_base.cpp
    linearizor_sc.cpp
)

target_link_libraries(rootba_povar_solver
  PUBLIC
    rootba_povar
    rootba_povar_cg
    rootba_povar_sc
)

