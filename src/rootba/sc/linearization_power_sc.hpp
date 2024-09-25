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

#include <Eigen/Dense>
#include <basalt/utils/sophus_utils.hpp>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "rootba/bal/bal_problem.hpp"
#include "rootba/cg/conjugate_gradient.hpp"
#include "rootba/sc/landmark_block.hpp"
#include "rootba/sc/linearization_sc.hpp"
#include "rootba/util/assert.hpp"

#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

namespace rootba {

template <typename Scalar_, int POSE_SIZE_>
class LinearizationPowerSC : private LinearizationSC<Scalar_, POSE_SIZE_> {
 public:
  using Scalar = Scalar_;
  static constexpr int POSE_SIZE = POSE_SIZE_;
  using Base = LinearizationSC<Scalar_, POSE_SIZE_>;

  using Vec2 = Eigen::Matrix<Scalar_, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar_, 4, 1>;
  using VecX = Eigen::Matrix<Scalar_, Eigen::Dynamic, 1>;
  using MatX = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;
  using Mat36 = Eigen::Matrix<Scalar_, 3, 6>;
  using RowMatX =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using PerSolveOptions =
      typename ConjugateGradientsSolver<Scalar>::PerSolveOptions;
  using SparseMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

  struct Options {
    typename LandmarkBlockSC<Scalar, POSE_SIZE>::Options sc_options;
    size_t power_sc_iterations = 20;
  };

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

  LinearizationPowerSC(BalProblem<Scalar>& bal_problem, const Options& options)
      : Base(bal_problem, options.sc_options, false),
        m_(options.power_sc_iterations),
        b_inv_(POSE_SIZE * num_cameras_, POSE_SIZE),
        hll_inv_(3 * landmark_blocks_.size(), 3) {
    ROOTBA_ASSERT(pose_mutex_.size() == num_cameras_);
  }

  void prepare_Hb(VecX& b_p) {
    b_inv_.setZero(POSE_SIZE * num_cameras_, POSE_SIZE);
    b_p.setZero(num_cameras_ * POSE_SIZE);

    {
      auto body = [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
          const auto& lb = landmark_blocks_.at(r);
          auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
          lb.get_Hll_inv_add_Hpp_b(b_inv_, hll_inv, b_p, pose_mutex_);
        }
      };

      tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
      tbb::parallel_for(range, body);
    }

    {
      auto body = [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
          auto b_inv =
              b_inv_.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * r, 0);
          b_inv.diagonal().array() += pose_damping_diagonal_;
          b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
              MatX::Identity(POSE_SIZE, POSE_SIZE));
        }
      };

      tbb::blocked_range<size_t> range(0, num_cameras_);
      tbb::parallel_for(range, body);
    }
  }

  Summary solve(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
    ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == POSE_SIZE * num_cameras_);

    Summary summary;

    // - (B^-1 * E_0)^i * B^-1 * g
    accum = right_mul_b_inv(-b_p);
    if (m_ > 0) {
      const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

      VecX tmp = accum;
      for (size_t i = 1; i <= m_; ++i) {
        tmp = right_mul_b_inv(right_mul_e0(tmp));
        accum += tmp;

        // Check if converged
        const Scalar iter_norm =
            pso.q_tolerance > 0 || pso.r_tolerance > 0 ? tmp.norm() : 0;
        if (pso.q_tolerance > 0) {
          const Scalar zeta = i * iter_norm / accum.norm();
          if (zeta < pso.q_tolerance) {
            summary.termination_type = Summary::LINEAR_SOLVER_SUCCESS;
            summary.num_iterations = i;
            std::stringstream ss;
            ss << "Iteration: " << summary.num_iterations
               << " Convergence. zeta = " << zeta << " < " << pso.q_tolerance;
            summary.message = ss.str();
            return summary;
          }
        }
        if (pso.r_tolerance > 0 && iter_norm / norm_0 < pso.r_tolerance) {
          summary.termination_type = Summary::LINEAR_SOLVER_SUCCESS;
          summary.num_iterations = i;
          std::stringstream ss;
          ss << "Iteration: " << summary.num_iterations
             << " Convergence. |r| = " << iter_norm / norm_0 << " < "
             << pso.r_tolerance;
          summary.message = ss.str();
          return summary;
        }
      }
    }

    summary.termination_type = Summary::LINEAR_SOLVER_NO_CONVERGENCE;
    summary.num_iterations = m_;
    summary.message = "Maximum number of iterations reached.";
    return summary;
  }

  void print_block(const std::string& filename, size_t block_idx) {
    landmark_blocks_[block_idx].printStorage(filename);
  }

  // For debugging only
  inline auto get_b_inv(const size_t cam_idx) const {
    return b_inv_.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * cam_idx, 0);
  }

  // For debugging only
  VecX right_multiply(const VecX& x) const {
    const auto jacobi = Base::get_jacobi();

    VecX res(POSE_SIZE * num_cameras_);
    for (size_t i = 0; i < num_cameras_; ++i) {
      const auto u = jacobi.block_diagonal.at(std::make_pair(i, i));
      const auto v = x.template segment<POSE_SIZE>(POSE_SIZE * i);
      res.template segment<POSE_SIZE>(POSE_SIZE * i) = u * v;
    }

    res -= right_mul_e0(x);
    return res;
  }

  // make selected base class methods public
  using Base::back_substitute;
  using Base::compute_Jp_scale_and_scale_Jp_cols;
  using Base::get_Jp_diag2;
  using Base::linearize_problem;
  using Base::num_cols_reduced;
  using Base::scale_Jl_cols;
  using Base::scale_Jp_cols;
  using Base::set_landmark_damping;
  using Base::set_pose_damping;

  inline VecX right_mul_b_inv(const VecX& x) const {
    ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * POSE_SIZE);

    VecX res(num_cameras_ * POSE_SIZE);

    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const auto u =
            b_inv_.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * r, 0);
        const auto v = x.template segment<POSE_SIZE>(POSE_SIZE * r);
        res.template segment<POSE_SIZE>(POSE_SIZE * r) = u * v;
      }
    };

    tbb::blocked_range<size_t> range(0, num_cameras_);
    tbb::parallel_for(range, body);

    return res;
  }

  inline VecX right_mul_e0(const VecX& x) const {
    ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * POSE_SIZE);

    VecX res = VecX::Zero(num_cameras_ * POSE_SIZE);

    auto body = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const auto& lb = landmark_blocks_.at(r);

        const auto& pose_indices = lb.get_pose_idx();
        const size_t num_obs = pose_indices.size();

        VecX jp_x(num_obs * 2);
        // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
        for (size_t i = 0; i < num_obs; ++i) {
          const auto u = lb.get_Jpi(i);
          const auto v =
              x.template segment<9>(pose_indices.at(i) * 9);
          jp_x.template segment<2>(i * 2) = u * v;
        }

        const auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
        const auto jl = lb.get_Jl();

        const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

        // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
        for (size_t i = 0; i < num_obs; ++i) {
          const auto u = lb.get_Jpi(i);
          const auto v = tmp.template segment<2>(i * 2);
          const size_t pose_idx = pose_indices.at(i);

          {
            std::scoped_lock lock(pose_mutex_.at(pose_idx));
            res.template segment<9>(pose_idx * 9) +=
                u.transpose() * v;
          }
        }
      }
    };

    tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
    tbb::parallel_for(range, body);

    return res;
  }

  MatX get_M() const {

    SparseMat Jp = get_sparse_Jp();
    SparseMat Jl = get_sparse_Jl();
    SparseMat Hpp_inv = get_sparse_Hpp_inv();
    SparseMat Hll_inv = get_sparse_Hll_inv();

    MatX R = Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;

    return R;
  }

  MatX get_inverted_matrix(int & num_iteration) const {

    SparseMat Jp = get_sparse_Jp();
    SparseMat Jl = get_sparse_Jl();

    SparseMat Hpp_inv = get_sparse_Hpp_inv();
    SparseMat Hll_inv = get_sparse_Hll_inv();

    MatX accm = Eigen::Matrix<Scalar_,-1,-1>::Identity(Hpp_inv.rows(),Hpp_inv.cols());
    MatX tmp = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols());

    MatX sum = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols()) - Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;
    for (size_t i = 1; i <= num_iteration; ++i) {
      //std::cout << "i = " << i << "\n";
      //std::cout << "num_iteration = " << num_iteration << "\n";
      tmp = Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp * tmp;
      accm += tmp;
    }

    return sum.inverse() - accm;

  }

  MatX get_R(int & num_iteration) const {

    SparseMat Jp = get_sparse_Jp();
    SparseMat Jl = get_sparse_Jl();

    SparseMat Hpp_inv = get_sparse_Hpp_inv();
    SparseMat Hll_inv = get_sparse_Hll_inv();

    MatX accm = Eigen::Matrix<Scalar_,-1,-1>::Identity(Hpp_inv.rows(),Hpp_inv.cols());
    MatX tmp = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols());

    MatX sum = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols()) - Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;
    for (size_t i = 1; i <= num_iteration; ++i) {
      //std::cout << "i = " << i << "\n";
      //std::cout << "num_iteration = " << num_iteration << "\n";
      tmp = Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp * tmp;
      accm += tmp;
    }

    return (sum.inverse() - accm)*Hpp_inv;

  }

  MatX get_power_S(int & num_iteration) const {

    SparseMat Jp = get_sparse_Jp();
    SparseMat Jl = get_sparse_Jl();

    SparseMat Hpp_inv = get_sparse_Hpp_inv();
    SparseMat Hll_inv = get_sparse_Hll_inv();

    MatX accm = Eigen::Matrix<Scalar_,-1,-1>::Identity(Hpp_inv.rows(),Hpp_inv.cols());
    MatX tmp = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols());

    //MatX sum = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols()) - Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;
    for (size_t i = 1; i <= num_iteration; ++i) {
      tmp = Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp * tmp;
      accm += tmp;
    }

    return accm * Hpp_inv;

  }

  MatX get_S() const {

    SparseMat Jp = get_sparse_Jp();
    SparseMat Jl = get_sparse_Jl();

    SparseMat Hpp_inv = get_sparse_Hpp_inv();
    SparseMat Hll_inv = get_sparse_Hll_inv();

    MatX accm = Eigen::Matrix<Scalar_,-1,-1>::Identity(Hpp_inv.rows(),Hpp_inv.cols());
    MatX tmp = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols());

    MatX sum = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols()) - Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;

    return sum.inverse() * Hpp_inv;

  }

  // For debugging only
  // TODO: refactor to avoid duplicating with similar methods in implicitSC
  SparseMat get_sparse_Jp() const {
    std::vector<size_t> lm_obs_indices;
    size_t num_obs = 0;
    lm_obs_indices.reserve(landmark_blocks_.size());
    for (const auto& lm_block : landmark_blocks_) {
      lm_obs_indices.push_back(num_obs);
      num_obs += lm_block.num_poses();
    }

    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(num_obs * 2 * POSE_SIZE);

    for (size_t lm_idx = 0; lm_idx < landmark_blocks_.size(); ++lm_idx) {
      const auto& lb = landmark_blocks_[lm_idx];
      const size_t row_idx = lm_obs_indices[lm_idx] * 2;

      const auto& pose_indices = lb.get_pose_idx();
      for (size_t j = 0; j < pose_indices.size(); ++j) {
        const auto block = lb.get_Jpi(j);
        const size_t col_idx = pose_indices[j];

        const size_t row_offset = row_idx + j * 2;
        const size_t col_offset = col_idx * POSE_SIZE;

        for (Eigen::Index row = 0; row < 2; ++row) {
          for (Eigen::Index col = 0; col < POSE_SIZE; ++col) {
            triplets.emplace_back(row + row_offset, col + col_offset,
                                  block(row, col));
          }
        }
      }
    }

    // build sparse matrix
    SparseMat res(num_obs * 2, num_cameras_ * POSE_SIZE);
    if (!triplets.empty()) {
      res.setFromTriplets(triplets.begin(), triplets.end());
    }

    return res;
  }

  // For debugging only
  // TODO: refactor to avoid duplicating with similar methods in implicitSC
  SparseMat get_sparse_Jl() const {
    std::vector<size_t> lm_obs_indices;
    size_t num_obs = 0;
    lm_obs_indices.reserve(landmark_blocks_.size());
    for (const auto& lm_block : landmark_blocks_) {
      lm_obs_indices.push_back(num_obs);
      num_obs += lm_block.num_poses();
    }

    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(num_obs * 2 * 3);

    for (size_t lm_idx = 0; lm_idx < landmark_blocks_.size(); ++lm_idx) {
      const auto& lb = landmark_blocks_[lm_idx];


      const size_t row_idx = lm_obs_indices[lm_idx] * 2;
      const size_t col_offset = lm_idx * 3;

      const size_t num_lm_obs = lb.get_pose_idx().size();
      for (size_t j = 0; j < num_lm_obs; ++j) {
        const auto block = lb.get_Jli(j);
        const size_t row_offset = row_idx + j * 2;

        for (Eigen::Index row = 0; row < 2; ++row) {
          for (Eigen::Index col = 0; col < 3; ++col) {
            triplets.emplace_back(row + row_offset, col + col_offset,
                                  block(row, col));
          }
        }
      }
    }

    // build sparse matrix
    SparseMat res(num_obs * 2, landmark_blocks_.size() * 3);
    if (!triplets.empty()) {
      res.setFromTriplets(triplets.begin(), triplets.end());
    }

    return res;
  }



  // TODO: refactor to avoid duplicating with similar methods in implicitSC
  SparseMat get_sparse_Hll_inv() const {
    BlockSparseMatrix<Scalar, 3> Hll_inv(3 * landmark_blocks_.size(),
                                         3 * landmark_blocks_.size());
    for (size_t i = 0; i < landmark_blocks_.size(); ++i) {
      //const auto& lb = landmark_blocks_[i];
      Eigen::Matrix<Scalar, 3, 3> m = hll_inv_.template block<3, 3>(3 * i, 0);
      //Eigen::Matrix<Scalar, 3, 3> m = lb.get_Hll_inv();
      //std::cout << "m Hll_inv.norm() = " << m.norm() << "\n";
      Hll_inv.add(i, i, std::move(m));
    }
    return Hll_inv.get_sparse_matrix();
  }

  /*// TODO: refactor to avoid duplicating with similar methods in implicitSC
  SparseMat get_sparse_Hll_inv() const {
    BlockSparseMatrix<Scalar, 3> Hll_inv(3 * landmark_blocks_.size(),
                                         3 * landmark_blocks_.size());
    for (size_t i = 0; i < landmark_blocks_.size(); ++i) {
      const auto& lb = landmark_blocks_[i];
      Eigen::Matrix<Scalar, 3, 3> m = hll_inv_.template block<3, 3>(3 * i, 0);
      //Eigen::Matrix<Scalar, 3, 3> m = lb.get_Hll_inv();
      Hll_inv.add(i, i, std::move(m));
    }

    return Hll_inv.get_sparse_matrix();
    //return hll_inv_.get_sparse_matrix();
  }*/

  // TODO: refactor to avoid duplicating with similar methods in implicitSC
  SparseMat get_sparse_Hpp_inv() const {
    BlockDiagonalAccumulator<Scalar> Hpp;
    //RowMatX Hpp;
    {
      auto body = [&](const tbb::blocked_range<size_t>& range) {
        for (size_t r = range.begin(); r != range.end(); ++r) {
          const auto& lb = landmark_blocks_[r];
          lb.add_Hpp(Hpp, pose_mutex_);
        }
      };
      tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
      tbb::parallel_for(range, body);

      if (pose_damping_diagonal_ > 0) {
        Hpp.add_diag(
            num_cameras_, POSE_SIZE,
            VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_));
      }
    }

    BlockSparseMatrix<Scalar, POSE_SIZE> Hpp_inv(num_cameras_ * POSE_SIZE,
                                                 num_cameras_ * POSE_SIZE);
    {
      auto body = [&](const typename IndexedBlocks<Scalar>::range_type& range) {
        for (const auto& [idx, block] : range) {
          Eigen::Matrix<Scalar, POSE_SIZE, POSE_SIZE> inv_block =
              block.inverse();
          Hpp_inv.add(idx.first, idx.second, std::move(inv_block));
        }
      };
      tbb::parallel_for(Hpp.block_diagonal.range(), body);
    }

    return Hpp_inv.get_sparse_matrix();
  }

 protected:
  using Base::landmark_blocks_;
  using Base::num_cameras_;
  using Base::pose_damping_diagonal_;
  using Base::pose_mutex_;

  const size_t m_;
  // TODO@demmeln(LOW, Tin): Don't call it b_inv. b is already used for RCS
  // gradient. I know in Simon's notes it's called B, but in this code base
  // `Hpp_inv_` probably makes more sense.
  RowMatX b_inv_;
  RowMatX hll_inv_;
};

}  // namespace rootba
