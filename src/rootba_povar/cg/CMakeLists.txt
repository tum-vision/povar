add_library(rootba_povar_cg)
set_target_properties(rootba_povar_cg PROPERTIES LINKER_LANGUAGE CXX)


target_sources(rootba_povar_cg
  PUBLIC
    block_sparse_matrix.hpp
    conjugate_gradient.hpp
    preconditioner.hpp
    utils.hpp
)

target_link_libraries(rootba_povar_cg
  PUBLIC
    rootba_povar
)

