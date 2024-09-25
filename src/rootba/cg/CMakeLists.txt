add_library(rootba_cg)
set_target_properties(rootba_cg PROPERTIES LINKER_LANGUAGE CXX)


target_sources(rootba_cg
  PUBLIC
    block_sparse_matrix.hpp
    conjugate_gradient.hpp
    preconditioner.hpp
    utils.hpp
)

target_link_libraries(rootba_cg
  PUBLIC
    rootba
)

if(ROOTBA_ENABLE_TESTING)

  rootba_add_test(test_block_sparse_matrix
      block_sparse_matrix.test.cpp
    LINK_LIBRARIES
    rootba_cg
    rootba_testing
  )

endif()
