/**
BSD 3-Clause License

This file is part of the RootBA project.
https://github.com/NikolausDemmel/rootba

Copyright (c) 2021, Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#pragma once

#include <mutex>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "rootba/bal/bal_problem.hpp"
#include "rootba/qr/linearization_utils.hpp"
#include "rootba/sc/landmark_block.hpp"
#include "rootba/util/assert.hpp"
#include "rootba/util/cast.hpp"
#include "rootba/util/format.hpp"

namespace rootba {

template <typename Scalar_, int POSE_SIZE_>
class LinearizationSC {
 public:
  using Scalar = Scalar_;
  static constexpr int POSE_SIZE = POSE_SIZE_;

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using Vec12 = Eigen::Matrix<Scalar, 12, 1>;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  using Mat36 = Eigen::Matrix<Scalar, 3, 6>;

  using Options = typename LandmarkBlockSC<Scalar, POSE_SIZE>::Options;

    struct Summary {
        enum TerminationType {
            LINEAR_SOLVER_NO_CONVERGENCE,
            LINEAR_SOLVER_SUCCESS,
            LINEAR_SOLVER_FAILURE
        };

        TerminationType termination_type;
        std::string message;
        int num_iterations = 0;
    };

  LinearizationSC(BalProblem<Scalar>& bal_problem, const Options& options,
                  const bool enable_reduction_alg = true)
      : options_(options),
        bal_problem_(bal_problem),
        num_cameras_(bal_problem_.cameras().size()),
        pose_mutex_(num_cameras_) {
    const size_t num_landmarks = bal_problem_.landmarks().size();
    landmark_blocks_.resize(num_landmarks);

    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        landmark_blocks_[r].allocate_landmark(bal_problem_.landmarks()[r],
                                              options_);
      }
    };

    tbb::blocked_range<size_t> range(0, num_landmarks);
    tbb::parallel_for(range, body);

    // H_pp_mutex_ won't be allocated when this class is inherited,
    // only used in get_Hb
    if (enable_reduction_alg && options_.reduction_alg == 1) {
      // TODO@demmeln: why not just `H_pp_mutex_.resize()`?
      // TODO@tin: Type must be moveable for resizing/
      // https://stackoverflow.com/questions/16465633/how-can-i-use-something-like-stdvectorstdmutex
      std::vector<std::mutex> tmp(num_cameras_ * num_cameras_);
      H_pp_mutex_.swap(tmp);  // avoid move and copy
    }
  };

  // return value `false` indicates numerical failure --> linearization at this
  // state is unusable. Numeric check is only performed for residuals that were
  // considered to be used (valid), which depends on use_valid_projections_only
  // setting.
  bool linearize_problem_refine() {
    ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

    size_t num_landmarks = landmark_blocks_.size();

    auto body = [&](const tbb::blocked_range<size_t>& range,
                    bool numerically_valid) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        landmark_blocks_[r].linearize_landmark_refine(bal_problem_.cameras());
        numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
      }
      return numerically_valid;
    };

    tbb::blocked_range<size_t> range(0, num_landmarks);
    const bool numerically_valid =
        tbb::parallel_reduce(range, true, body, std::logical_and<>());

    return numerically_valid;
  };

    bool linearize_problem() {
        ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

        size_t num_landmarks = landmark_blocks_.size();

        auto body = [&](const tbb::blocked_range<size_t>& range,
                        bool numerically_valid) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].linearize_landmark(bal_problem_.cameras());
                numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
            }
            return numerically_valid;
        };

        tbb::blocked_range<size_t> range(0, num_landmarks);
        const bool numerically_valid =
                tbb::parallel_reduce(range, true, body, std::logical_and<>());

        return numerically_valid;
    };

    bool linearize_problem_pOSE(int alpha) {
        ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

        size_t num_landmarks = landmark_blocks_.size();

        auto body = [&](const tbb::blocked_range<size_t>& range,
                        bool numerically_valid) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].linearize_landmark_pOSE(bal_problem_.cameras(), alpha);
                numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
            }
            return numerically_valid;
        };

        tbb::blocked_range<size_t> range(0, num_landmarks);
        const bool numerically_valid =
                tbb::parallel_reduce(range, true, body, std::logical_and<>());

        return numerically_valid;
    };

    bool linearize_problem_projective_space_homogeneous() {
        ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

        size_t num_landmarks = landmark_blocks_.size();

        auto body = [&](const tbb::blocked_range<size_t>& range,
                        bool numerically_valid) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].linearize_landmark_projective_space_homogeneous(bal_problem_.cameras());
                numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
            }
            return numerically_valid;
        };

        tbb::blocked_range<size_t> range(0, num_landmarks);
        const bool numerically_valid =
                tbb::parallel_reduce(range, true, body, std::logical_and<>());

        return numerically_valid;
    };

    bool linearize_problem_expOSE(int alpha, bool init) {
        ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

        size_t num_landmarks = landmark_blocks_.size();

        auto body = [&](const tbb::blocked_range<size_t>& range,
                        bool numerically_valid) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].linearize_landmark_expOSE(bal_problem_.cameras(), alpha, init);
                numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
            }
            return numerically_valid;
        };

        tbb::blocked_range<size_t> range(0, num_landmarks);
        const bool numerically_valid =
                tbb::parallel_reduce(range, true, body, std::logical_and<>());

        return numerically_valid;
    };

  size_t num_cols_reduced() const { return num_cameras_ * 9; }
    size_t num_cols_reduced_expOSE() const { return num_cameras_ * POSE_SIZE; }
  size_t num_cols_reduced_pOSE() const { return num_cameras_ * POSE_SIZE; }
  size_t num_cols_reduced_joint() const { return num_cameras_ * 11; }

    VecX get_Jp_diag2_projective_space() const {
        struct Reductor {
            Reductor(size_t num_rows,
                     const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                     landmark_blocks)
                    : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                res.setZero(num_rows);
            }

            void operator()(const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks[r];
                    lb.add_Jp_diag2_projective_space(res);
                }
            }

            Reductor(Reductor& a, tbb::split /*unused*/)
                    : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                res.setZero(num_rows);
            };

            inline void join(const Reductor& b) { res += b.res; }

            size_t num_rows;
            //const std::vector<LandmarkBlockSC<Scalar, 15>>& landmark_blocks;
            const std::vector<LandmarkBlockSC<Scalar, 11>>& landmark_blocks;
            VecX res;
        };

        Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_reduce(range, r);

        // TODO: double check including vs not including pose damping here in usage
        // and make it clear in API; see also getJpTJp_blockdiag

        // Note: ignore damping here

        return r.res;
    }


    Scalar landmark_closed_form_pOSE(int alpha, const VecX& pose_inc) {
        ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

        auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].landmark_closed_form_pOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
            }
            return l_diff;
        };
        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        Scalar l_diff =
                tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
        return l_diff;
    }

    Summary solve_direct_pOSE(BlockSparseMatrix<Scalar> & H_pp, const VecX& b_p, VecX& accum) const {
        ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == POSE_SIZE * num_cameras_);

        Summary summary;

        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> H = H_pp.get_sparse_matrix();
        Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar, Eigen::RowMajor>> solver;
        accum = solver.compute(H).solve(-b_p);
        //Eigen::SparseLU<Eigen::SparseMatrix<Scalar, Eigen::RowMajor>> solver;
        //solver.analyzePattern(H);
        //solver.factorize(H);
        //accum = solver.solve(-b_p);
        return summary;
    };

    Summary solve_direct_expOSE(BlockSparseMatrix<Scalar> & H_pp, const VecX& b_p, VecX& accum) const {
        ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == POSE_SIZE * num_cameras_);

        Summary summary;

        Eigen::SparseMatrix<Scalar, Eigen::RowMajor> H = H_pp.get_sparse_matrix();
        Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar, Eigen::RowMajor>> solver;
        accum = solver.compute(H).solve(-b_p);

        return summary;
    };

    void update_y_tilde_expose() {
        //ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].update_y_tilde_expose(bal_problem_.cameras());
            }
            //return l_diff;
        };
        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);
    }

    void initialize_y_tilde_expose() {
        //ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].update_y_tilde_expose_initialize(bal_problem_.cameras());
            }
            //return l_diff;
        };
        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);
    }



    Scalar landmark_closed_form_expOSE(int alpha, const VecX& pose_inc) {
        ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

        auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].landmark_closed_form_expOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
            }
            return l_diff;
        };
        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        Scalar l_diff =
                tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
        return l_diff;
    }

    Scalar back_substitute_joint(const VecX& pose_inc) {
        ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));

        auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].back_substitute_joint(pose_inc, l_diff, bal_problem_.cameras());
            }
            return l_diff;
        };

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        Scalar l_diff =
                tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
        return l_diff;
    }

  Scalar back_substitute(const VecX& pose_inc) {
    ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 9));

    auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        landmark_blocks_[r].back_substitute(pose_inc, l_diff);
      }
      return l_diff;
    };

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    Scalar l_diff =
        tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
    return l_diff;
  }

  VecX get_Jp_diag2() const {
    struct Reductor {
      Reductor(size_t num_rows,
               const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                   landmark_blocks)
          : num_rows(num_rows), landmark_blocks(landmark_blocks) {
        res.setZero(num_rows);
      }

      void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
          const auto& lb = landmark_blocks[r];
          lb.add_Jp_diag2(res);
        }
      }

      Reductor(Reductor& a, tbb::split /*unused*/)
          : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
        res.setZero(num_rows);
      };

      inline void join(const Reductor& b) { res += b.res; }

      size_t num_rows;
      const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
      VecX res;
    };

    Reductor r(num_cameras_ * POSE_SIZE, landmark_blocks_);

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    tbb::parallel_reduce(range, r);

    // TODO: double check including vs not including pose damping here in usage
    // and make it clear in API; see also getJpTJp_blockdiag

    // Note: ignore damping here

    return r.res;
  }

    VecX get_Jp_diag2_golden() const {
        struct Reductor {
            Reductor(size_t num_rows,
                     const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                     landmark_blocks)
                    : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                res.setZero(num_rows);
            }

            void operator()(const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks[r];
                    lb.add_Jp_diag2(res);
                }
            }

            Reductor(Reductor& a, tbb::split /*unused*/)
                    : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                res.setZero(num_rows);
            };

            inline void join(const Reductor& b) { res += b.res; }

            size_t num_rows;
            const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
            VecX res;
        };

        Reductor r(num_cameras_ * 9, landmark_blocks_);

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_reduce(range, r);

        // TODO: double check including vs not including pose damping here in usage
        // and make it clear in API; see also getJpTJp_blockdiag

        // Note: ignore damping here

        return r.res;
    }

    VecX get_Jp_diag2_pOSE() const {
        struct Reductor {
            Reductor(size_t num_rows,
                     const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                     landmark_blocks)
                    : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                res.setZero(num_rows);
            }

            void operator()(const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks[r];
                    lb.add_Jp_diag2_pOSE(res);
                }
            }

            Reductor(Reductor& a, tbb::split /*unused*/)
                    : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                res.setZero(num_rows);
            };

            inline void join(const Reductor& b) { res += b.res; }

            size_t num_rows;
            const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
            VecX res;
        };

        Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_reduce(range, r);

        // TODO: double check including vs not including pose damping here in usage
        // and make it clear in API; see also getJpTJp_blockdiag

        // Note: ignore damping here

        return r.res;
    }

    VecX get_Jp_diag2_expOSE() const {
        struct Reductor {
            Reductor(size_t num_rows,
                     const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                     landmark_blocks)
                    : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                res.setZero(num_rows);
            }

            void operator()(const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks[r];
                    lb.add_Jp_diag2_expOSE(res);
                }
            }

            Reductor(Reductor& a, tbb::split /*unused*/)
                    : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                res.setZero(num_rows);
            };

            inline void join(const Reductor& b) { res += b.res; }

            size_t num_rows;
            const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
            VecX res;
        };

        Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_reduce(range, r);

        // TODO: double check including vs not including pose damping here in usage
        // and make it clear in API; see also getJpTJp_blockdiag

        // Note: ignore damping here

        return r.res;
    }


    void scale_Jl_cols() {
    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        landmark_blocks_[r].scale_Jl_cols();
      }
    };

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    tbb::parallel_for(range, body);
  }

    void scale_Jl_cols_homogeneous() {
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].scale_Jl_cols_homogeneous();
            }
        };

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);
    }

    void scale_Jl_cols_pOSE() {
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].scale_Jl_cols_pOSE();
            }
        };

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);
    }

  void scale_Jp_cols(const VecX& jacobian_scaling) {
    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        landmark_blocks_[r].scale_Jp_cols(jacobian_scaling);
      }
    };

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    tbb::parallel_for(range, body);
  }

    void scale_Jp_cols_pOSE(const VecX& jacobian_scaling) {
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].scale_Jp_cols_pOSE(jacobian_scaling);
            }
        };

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);
    }

    void scale_Jp_cols_joint(const VecX& jacobian_scaling) {
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].scale_Jp_cols_joint(jacobian_scaling);
            }
        };

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);
    }

    void scale_Jp_cols_expOSE(const VecX& jacobian_scaling) {
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].scale_Jp_cols_expOSE(jacobian_scaling);
            }
        };

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);
    }

  /// compute scale and apply scaling in one go, e.g. for unit tests.
  void compute_Jp_scale_and_scale_Jp_cols() {
    const VecX Jp_scale2 = get_Jp_diag2();
    const VecX Jp_scaling =
        compute_jacobi_scaling(Jp_scale2, options_.jacobi_scaling_eps);
    scale_Jp_cols(Jp_scaling);
  }

  void set_pose_damping(const Scalar lambda) {
    ROOTBA_ASSERT(lambda >= 0);

    pose_damping_diagonal_ = lambda;
  }

    void set_pose_damping_pOSE(const Scalar lambda) {
        ROOTBA_ASSERT(lambda >= 0);

        pose_damping_diagonal_pOSE_ = lambda;
    }

    void set_pose_damping_expOSE(const Scalar lambda) {
        ROOTBA_ASSERT(lambda >= 0);

        pose_damping_diagonal_expOSE_ = lambda;
    }

    void set_landmark_damping_joint(const Scalar lambda) {
        ROOTBA_ASSERT(lambda >= 0);

        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].set_landmark_damping_joint(lambda);
            }
        };

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);
    }

    bool linearize_nullspace() {
        ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

        size_t num_landmarks = landmark_blocks_.size();

        auto body = [&](const tbb::blocked_range<size_t>& range,
                        bool numerically_valid) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].linearize_nullspace(bal_problem_.cameras());
                numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
            }
            return numerically_valid;
        };

        tbb::blocked_range<size_t> range(0, num_landmarks);
        const bool numerically_valid =
                tbb::parallel_reduce(range, true, body, std::logical_and<>());

        return numerically_valid;
    };


    inline bool has_pose_damping() const { return pose_damping_diagonal_ > 0; }
    inline bool has_pose_damping_pOSE() const { return pose_damping_diagonal_pOSE_ > 0; }
    inline bool has_pose_damping_expOSE() const { return pose_damping_diagonal_expOSE_ > 0; }

  void set_landmark_damping(const Scalar lambda) {
    ROOTBA_ASSERT(lambda >= 0);

    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        landmark_blocks_[r].set_landmark_damping(lambda);
      }
    };

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    tbb::parallel_for(range, body);
  }

  void get_Hb(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
    if (options_.reduction_alg == 0) {
      //std::cout << "get_hb_r starts \n";
      get_hb_r(H_pp, b_p);
    } else if (options_.reduction_alg == 1) {
      //std::cout << "get_hb_f starts \n";
      get_hb_f(H_pp, b_p);
    } else {
      LOG(FATAL) << "options_.reduction_alg " << options_.reduction_alg
                 << " is not supported.";
    }
    H_pp.recompute_keys();
  }

    void get_Hb_pOSE(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        if (options_.reduction_alg == 0) {
            get_hb_r_pOSE(H_pp, b_p);
        } else if (options_.reduction_alg == 1) {
            get_hb_f_pOSE(H_pp, b_p);
        } else {
            LOG(FATAL) << "options_.reduction_alg " << options_.reduction_alg
                       << " is not supported.";
        }
        H_pp.recompute_keys();
    }

    void get_Hb_joint(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        get_hb_f_joint(H_pp, b_p);
        H_pp.recompute_keys();
    }

    void get_Hb_expOSE(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        if (options_.reduction_alg == 0) {
            get_hb_r_expOSE(H_pp, b_p);
        } else if (options_.reduction_alg == 1) {
            get_hb_f_expOSE(H_pp, b_p);
        } else {
            LOG(FATAL) << "options_.reduction_alg " << options_.reduction_alg
                       << " is not supported.";
        }
        H_pp.recompute_keys();
    }

  BlockDiagonalAccumulator<Scalar> get_jacobi() const {
    struct Reductor {
      Reductor(const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& lm_blocks)
          : lm_blocks(lm_blocks) {}

      Reductor(Reductor& a, tbb::split /*unused*/) : lm_blocks(a.lm_blocks) {}

      void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
          const auto& lb = lm_blocks[r];
          lb.add_jp_t_jp_blockdiag(accum);
        }
      }

      inline void join(Reductor& b) { accum.join(b.accum); }

      BlockDiagonalAccumulator<Scalar> accum;
      const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& lm_blocks;
    };

    Reductor r(landmark_blocks_);

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    tbb::parallel_reduce(range, r);

    if (has_pose_damping()) {
      r.accum.add_diag(
          num_cameras_, POSE_SIZE,
          VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_));
    }

    // TODO: double check including vs not including pose damping here in usage
    // and make it clear in API; see also getJp_diag2

    return r.accum;
  }

  void print_block(const std::string& filename, size_t block_idx) {
    landmark_blocks_[block_idx].printStorage(filename);
  }

 protected:
  void get_hb_r(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
    struct Reductor {
      Reductor(const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                   landmark_blocks,
               size_t num_cameras)
          : landmark_blocks(landmark_blocks),
            num_cameras(num_cameras),
            H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
        b_p.setZero(num_cameras * POSE_SIZE);
      }

      Reductor(Reductor& a, tbb::split /*unused*/)
          : landmark_blocks(a.landmark_blocks),
            num_cameras(a.num_cameras),
            H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
        b_p.setZero(a.num_cameras * POSE_SIZE);
      }

      void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
          const auto& lb = landmark_blocks[r];
          lb.add_Hb(H_pp, b_p);
        }
      }

      inline void join(Reductor& other) {
        H_pp.join(other.H_pp);
        b_p += other.b_p;
      }

      const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
      size_t num_cameras;

      BlockSparseMatrix<Scalar> H_pp;
      VecX b_p;
    };

    Reductor r(landmark_blocks_, num_cameras_);

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    tbb::parallel_reduce(range, r);

    if (has_pose_damping()) {
      r.H_pp.add_diag(
          num_cameras_, POSE_SIZE,
          VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_));
    }

    H_pp = std::move(r.H_pp);
    b_p = std::move(r.b_p);
  }

    void get_hb_r_pOSE(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        struct Reductor {
            Reductor(const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
            landmark_blocks,
                     size_t num_cameras)
                    : landmark_blocks(landmark_blocks),
                      num_cameras(num_cameras),
                      H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
                b_p.setZero(num_cameras * POSE_SIZE);
            }

            Reductor(Reductor& a, tbb::split /*unused*/)
                    : landmark_blocks(a.landmark_blocks),
                      num_cameras(a.num_cameras),
                      H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
                b_p.setZero(a.num_cameras * POSE_SIZE);
            }

            void operator()(const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks[r];
                    lb.add_Hb_pOSE(H_pp, b_p);
                }
            }

            inline void join(Reductor& other) {
                H_pp.join(other.H_pp);
                b_p += other.b_p;
            }

            const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
            size_t num_cameras;

            BlockSparseMatrix<Scalar> H_pp;
            VecX b_p;
        };

        Reductor r(landmark_blocks_, num_cameras_);

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_reduce(range, r);

        if (has_pose_damping_pOSE()) {
            r.H_pp.add_diag(
                    num_cameras_, POSE_SIZE,
                    VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_pOSE_));

        }

        H_pp = std::move(r.H_pp);
        b_p = std::move(r.b_p);
    }



    void get_hb_r_expOSE(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        struct Reductor {
            Reductor(const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
            landmark_blocks,
                     size_t num_cameras)
                    : landmark_blocks(landmark_blocks),
                      num_cameras(num_cameras),
                      H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
                b_p.setZero(num_cameras * POSE_SIZE);
            }

            Reductor(Reductor& a, tbb::split /*unused*/)
                    : landmark_blocks(a.landmark_blocks),
                      num_cameras(a.num_cameras),
                      H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
                b_p.setZero(a.num_cameras * POSE_SIZE);
            }

            void operator()(const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks[r];
                    lb.add_Hb_expOSE(H_pp, b_p);
                }
            }

            inline void join(Reductor& other) {
                H_pp.join(other.H_pp);
                b_p += other.b_p;
            }

            const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
            size_t num_cameras;

            BlockSparseMatrix<Scalar> H_pp;
            VecX b_p;
        };

        Reductor r(landmark_blocks_, num_cameras_);

        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_reduce(range, r);

        if (has_pose_damping_pOSE()) {
            r.H_pp.add_diag(
                    num_cameras_, POSE_SIZE,
                    VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_expOSE_));
        }

        H_pp = std::move(r.H_pp);
        b_p = std::move(r.b_p);
    }



  void get_hb_f(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
    ROOTBA_ASSERT(H_pp_mutex_.size() == num_cameras_ * num_cameras_);

    // Fill H_pp and b_p
    b_p.setZero(H_pp.rows);
    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const auto& lb = landmark_blocks_[r];
        lb.add_Hb(H_pp, b_p, H_pp_mutex_, pose_mutex_, num_cameras_);
      }
    };
    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    tbb::parallel_for(range, body);

    if (has_pose_damping()) {
      H_pp.add_diag(
          num_cameras_, 9,
          VecX::Constant(num_cameras_ * 9, pose_damping_diagonal_));
    }
  }

    void get_hb_f_pOSE(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        ROOTBA_ASSERT(H_pp_mutex_.size() == num_cameras_ * num_cameras_);

        // Fill H_pp and b_p
        b_p.setZero(H_pp.rows);
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                const auto& lb = landmark_blocks_[r];
                lb.add_Hb_pOSE(H_pp, b_p, H_pp_mutex_, pose_mutex_, num_cameras_);
            }
        };
        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);

        if (has_pose_damping_pOSE()) {
            //std::cout << "pose_damping_diagonal_pOSE_ = " << pose_damping_diagonal_pOSE_ << "\n";
            H_pp.add_diag(
                    num_cameras_, POSE_SIZE,
                    VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_pOSE_));
            //std::cout << "VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_pOSE_) = " << VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_pOSE_) << "\n";

        }
    }

    void get_hb_f_joint(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        ROOTBA_ASSERT(H_pp_mutex_.size() == num_cameras_ * num_cameras_);

        // Fill H_pp and b_p
        b_p.setZero(H_pp.rows);
        //diag_joint.setZero(H_pp.rows);
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                const auto& lb = landmark_blocks_[r];
                lb.add_Hb_joint(H_pp, b_p, H_pp_mutex_, pose_mutex_, num_cameras_);
            }
        };
        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);

        {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& cam = bal_problem_.cameras().at(r);

                    Vec12 camera_space_matrix;
                    camera_space_matrix(0) = cam.space_matrix(0,0);
                    camera_space_matrix(1) = cam.space_matrix(0,1);
                    camera_space_matrix(2) = cam.space_matrix(0,2);
                    camera_space_matrix(3) = cam.space_matrix(0,3);
                    camera_space_matrix(4) = cam.space_matrix(1,0);
                    camera_space_matrix(5) = cam.space_matrix(1,1);
                    camera_space_matrix(6) = cam.space_matrix(1,2);
                    camera_space_matrix(7) = cam.space_matrix(1,3);
                    camera_space_matrix(8) = cam.space_matrix(2,0);
                    camera_space_matrix(9) = cam.space_matrix(2,1);
                    camera_space_matrix(10) = cam.space_matrix(2,2);
                    camera_space_matrix(11) = cam.space_matrix(2,3);

                    auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                    H_pp.add(r , r, std::move(Proj_pose.transpose()*pose_damping_diagonal_*Proj_pose));

                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);
        }

    }


    void get_hb_f_expOSE(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        ROOTBA_ASSERT(H_pp_mutex_.size() == num_cameras_ * num_cameras_);

        // Fill H_pp and b_p
        b_p.setZero(H_pp.rows);
        auto body = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                const auto& lb = landmark_blocks_[r];
                lb.add_Hb_expOSE(H_pp, b_p, H_pp_mutex_, pose_mutex_, num_cameras_);
            }
        };
        tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
        tbb::parallel_for(range, body);

        if (has_pose_damping_expOSE()) {
            H_pp.add_diag(
                    num_cameras_, POSE_SIZE,
                    VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_expOSE_));
        }
    }



  Options options_;
  BalProblem<Scalar>& bal_problem_;
  size_t num_cameras_;

  mutable std::vector<std::mutex> H_pp_mutex_;
  mutable std::vector<std::mutex> pose_mutex_;

  std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>> landmark_blocks_;

  Scalar pose_damping_diagonal_ = 0;
  Scalar pose_damping_diagonal_pOSE_ = 0;
  Scalar pose_damping_diagonal_expOSE_ = 0;
};

}  // namespace rootba
