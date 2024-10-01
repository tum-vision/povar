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

#include "rootba_povar/bal/bal_problem.hpp"
#include "rootba_povar/sc/landmark_block.hpp"
#include "rootba_povar/util/assert.hpp"
#include "rootba_povar/util/cast.hpp"
#include "rootba_povar/util/format.hpp"

namespace rootba_povar {

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

      std::vector<std::mutex> tmp(num_cameras_ * num_cameras_);
      H_pp_mutex_.swap(tmp);  // avoid move and copy

  };

  // return value `false` indicates numerical failure --> linearization at this
  // state is unusable. Numeric check is only performed for residuals that were
  // considered to be used (valid), which depends on use_valid_projections_only
  // setting.

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

    bool linearize_problem_pOSE(Scalar alpha) {
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
            const std::vector<LandmarkBlockSC<Scalar, 12>>& landmark_blocks;
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


    Scalar back_substitute_pOSE(Scalar alpha, const VecX& pose_inc) {
        ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 12));

        auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
            for (size_t r = range.begin(); r != range.end(); ++r) {
                landmark_blocks_[r].back_substitute_pOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
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

        return summary;
    };

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


  void set_pose_damping(const Scalar lambda) {
    ROOTBA_ASSERT(lambda >= 0);

    pose_damping_diagonal_ = lambda;
  }

    void set_pose_damping_pOSE(const Scalar lambda) {
        ROOTBA_ASSERT(lambda >= 0);

        pose_damping_diagonal_pOSE_ = lambda;
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


    void get_Hb_pOSE(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        get_hb_f_pOSE(H_pp, b_p);
        H_pp.recompute_keys();
    }

    void get_Hb_joint(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        get_hb_f_joint(H_pp, b_p);
        H_pp.recompute_keys();
    }


  void print_block(const std::string& filename, size_t block_idx) {
    landmark_blocks_[block_idx].printStorage(filename);
  }

 protected:
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
            H_pp.add_diag(
                    num_cameras_, POSE_SIZE,
                    VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_pOSE_));
        }
    }

    void get_hb_f_joint(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
        ROOTBA_ASSERT(H_pp_mutex_.size() == num_cameras_ * num_cameras_);

        // Fill H_pp and b_p
        b_p.setZero(H_pp.rows);
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



  Options options_;
  BalProblem<Scalar>& bal_problem_;
  size_t num_cameras_;

  mutable std::vector<std::mutex> H_pp_mutex_;
  mutable std::vector<std::mutex> pose_mutex_;

  std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>> landmark_blocks_;

  Scalar pose_damping_diagonal_ = 0;
  Scalar pose_damping_diagonal_pOSE_ = 0;
};

}  // namespace rootba_povar
