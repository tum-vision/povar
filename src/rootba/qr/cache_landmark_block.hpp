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

#include <memory>
#include <mutex>

#include <Eigen/Dense>

#include "rootba/bal/bal_bundle_adjustment_helper.hpp"
#include "rootba/bal/bal_problem.hpp"
#include "rootba/bal/bal_residual_options.hpp"
#include "rootba/cg/block_sparse_matrix.hpp"
#include "rootba/qr/landmark_block.hpp"

namespace rootba {

template <typename Scalar>
class FactorQRBlock;

// TODO@demmeln(LOW, Niko+Tin): I'm not sure if the terminology "cache" is right
// here. It sounds like this landmark block caches something, but it's more the
// opposite. Instead of caching, it recomputes things the QR on the fly. The
// scratch memory is also not really a cache, so those variables (cache_ptr)
// should maybe be renamed. Let's discuss first.
template <typename Scalar, int POSE_SIZE>
class CacheLandmarkBlock {
 public:
  using Options = typename LandmarkBlock<Scalar>::Options;
  using State = typename LandmarkBlock<Scalar>::State;

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMatX =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  using RowMat3 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;

  using Landmark = typename BalProblem<Scalar>::Landmark;
  using Camera = typename BalProblem<Scalar>::Camera;
  using Landmarks = typename BalProblem<Scalar>::Landmarks;
  using Cameras = typename BalProblem<Scalar>::Cameras;

  using RowMatXMap =
      Eigen::Map<RowMatX, Eigen::internal::traits<MatX>::Alignment>;

  void allocate_landmark(Landmark& lm, const Options& options) {
    options_ = options;

    pose_idx_.clear();
    pose_idx_.reserve(lm.obs.size());
    for (const auto& [cam_idx, obs] : lm.obs) {
      pose_idx_.push_back(cam_idx);
    }

    num_rows_ = pose_idx_.size() * 2;
    lm_idx_ = POSE_SIZE;
    res_idx_ = lm_idx_ + 3;
    num_cols_ = res_idx_ + 1;

    storage_.resize(num_rows_, num_cols_);

    padding_idx_ = pose_idx_.size() * POSE_SIZE;
    num_cache_rows_ = pose_idx_.size() * 2 + 3;  // residuals and lm damping

    size_t pad = padding_idx_ % 4;
    if (pad != 0) {
      padding_size_ = 4 - pad;
    }

    cache_lm_idx_ = padding_idx_ + padding_size_;
    cache_res_idx_ = cache_lm_idx_ + 3;
    num_cache_cols_ = cache_res_idx_ + 1;

    state_ = State::ALLOCATED;
    lm_ptr_ = &lm;
  }

  inline void set_landmark_damping(const Scalar lambda) { lambda_ = lambda; };

  inline void linearize_landmark(const Cameras& cameras) {
    ROOTBA_ASSERT(state_ == State::ALLOCATED ||
                  state_ == State::NUMERICAL_FAILURE ||
                  state_ == State::LINEARIZED);

    storage_.setZero();

    bool numerically_valid = true;

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx = pose_idx_[i];
      size_t obs_idx = i * 2;

      const auto& obs = lm_ptr_->obs.at(cam_idx);
      const auto& cam = cameras.at(cam_idx);

      typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
      typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
      typename BalBundleAdjustmentHelper<Scalar>::MatRL Jl;

      Vec2 res;
      Vec3 v_init;
      const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point(
          obs.pos, lm_ptr_->p_w, cam.T_c_w, cam.intrinsics, true, res, false, &Jp, &Ji,
          &Jl);

      if (!options_.use_valid_projections_only || valid) {
        numerically_valid = numerically_valid && Jl.array().isFinite().all() &&
                            Jp.array().isFinite().all() &&
                            Ji.array().isFinite().all() &&
                            res.array().isFinite().all();

        const Scalar res_squared = res.squaredNorm();
        const auto [weighted_error, weight] =
            BalBundleAdjustmentHelper<Scalar>::compute_error_weight(
                options_.residual_options, res_squared);
        const Scalar sqrt_weight = std::sqrt(weight);

        storage_.template block<2, 6>(obs_idx, 0) = sqrt_weight * Jp;
        storage_.template block<2, 3>(obs_idx, 6) = sqrt_weight * Ji;
        storage_.template block<2, 3>(obs_idx, lm_idx_) = sqrt_weight * Jl;
        storage_.template block<2, 1>(obs_idx, res_idx_) = sqrt_weight * res;
      }
    }

    if (numerically_valid) {
      state_ = State::LINEARIZED;
    } else {
      state_ = State::NUMERICAL_FAILURE;
    }
  }

    inline void linearize_landmark_pOSE(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == State::ALLOCATED ||
                      state_ == State::NUMERICAL_FAILURE ||
                      state_ == State::LINEARIZED);

        storage_.setZero();

        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 2;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL Jl;

            Vec2 res;
            Vec3 v_init;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point(
                    obs.pos, lm_ptr_->p_w, cam.T_c_w, cam.intrinsics, true, res, false, &Jp, &Ji,
                    &Jl);

            if (!options_.use_valid_projections_only || valid) {
                numerically_valid = numerically_valid && Jl.array().isFinite().all() &&
                                    Jp.array().isFinite().all() &&
                                    Ji.array().isFinite().all() &&
                                    res.array().isFinite().all();

                const Scalar res_squared = res.squaredNorm();
                const auto [weighted_error, weight] =
                BalBundleAdjustmentHelper<Scalar>::compute_error_weight(
                        options_.residual_options, res_squared);
                const Scalar sqrt_weight = std::sqrt(weight);

                storage_.template block<2, 6>(obs_idx, 0) = sqrt_weight * Jp;
                storage_.template block<2, 3>(obs_idx, 6) = sqrt_weight * Ji;
                storage_.template block<2, 3>(obs_idx, lm_idx_) = sqrt_weight * Jl;
                storage_.template block<2, 1>(obs_idx, res_idx_) = sqrt_weight * res;
            }
        }

        if (numerically_valid) {
            state_ = State::LINEARIZED;
        } else {
            state_ = State::NUMERICAL_FAILURE;
        }
    }

  inline void add_Jp_diag2(VecX& res) const {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx = pose_idx_[i];
      res.template segment<POSE_SIZE>(POSE_SIZE * cam_idx) +=
          storage_.template block<2, POSE_SIZE>(2 * i, 0)
              .colwise()
              .squaredNorm();
    }
  }

  inline void scale_Jl_cols() {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
    // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
    Jl_col_scale_ =
        (options_.jacobi_scaling_eps +
         storage_.block(0, lm_idx_, num_rows_, 3).colwise().norm().array())
            .inverse();

    storage_.block(0, lm_idx_, num_rows_, 3) *= Jl_col_scale_.asDiagonal();
  }

  inline void scale_Jp_cols(const VecX& jacobian_scaling) {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx = pose_idx_[i];

      storage_.template block<2, POSE_SIZE>(2 * i, 0) *=
          jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
              .asDiagonal();
    }
  }

  inline void add_Q2TJp_T_Q2TJp_blockdiag(
      BlockDiagonalAccumulator<Scalar>& accu,
      std::vector<std::mutex>* pose_mutex, Scalar* cache_ptr) const {
    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);

    // we include the dampening rows (may be zeros if no dampening set)
    if (pose_mutex) {
      for (size_t i = 0; i < pose_idx_.size(); i++) {
        const auto Q2T_Jp =
            qr_cache.block(3, POSE_SIZE * i, num_cache_rows_ - 3, POSE_SIZE);

        const size_t cam_idx = pose_idx_[i];
        {
          MatX tmp = Q2T_Jp.transpose() * Q2T_Jp;
          std::scoped_lock lock(pose_mutex->at(cam_idx));
          accu.add(cam_idx, std::move(tmp));
        }
      }
    } else {
      for (size_t i = 0; i < pose_idx_.size(); i++) {
        auto Q2T_Jp =
            qr_cache.block(3, POSE_SIZE * i, num_cache_rows_ - 3, POSE_SIZE);
        const size_t cam_idx = pose_idx_[i];
        accu.add(cam_idx, Q2T_Jp.transpose() * Q2T_Jp);
      }
    }
  }

  inline void add_Q2TJp_T_Q2Tr(VecX& res, Scalar* cache_ptr) const {
    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);

    VecX x_pose_reduced =
        qr_cache
            .bottomLeftCorner(num_cache_rows_ - 3, padding_idx_ + padding_size_)
            .adjoint() *
        qr_cache.col(cache_res_idx_).tail(num_cache_rows_ - 3);
    // (Q2^T * Jp)^T * Q2^Tr

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx = pose_idx_[i];
      res.template segment<POSE_SIZE>(POSE_SIZE * cam_idx) +=
          x_pose_reduced.template segment<POSE_SIZE>(POSE_SIZE * i);
    }
  }

  inline size_t num_Q2T_rows() const { return num_rows_; }

  inline size_t num_reduced_cams() const { return pose_idx_.size(); }

  inline bool is_numerical_failure() const {
    return state_ == State::NUMERICAL_FAILURE;
  }

  inline void add_Q2TJp_T_Q2TJp_mult_x(VecX& res, const VecX& x_pose,
                                       std::vector<std::mutex>* pose_mutex,
                                       Scalar* cache_ptr) const {
    ROOTBA_ASSERT(res.size() == x_pose.size());

    RowMatX cache_mat;
    if (!cache_ptr) {
      cache_mat.resize(num_cache_rows_, num_cache_cols_);
      cache_ptr = cache_mat.data();
    }
    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);

    perform_qr(cache_ptr);

    VecX x_pose_reduced(padding_idx_ + padding_size_);

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      const size_t cam_idx = pose_idx_[i];
      x_pose_reduced.template segment<POSE_SIZE>(POSE_SIZE * i) =
          x_pose.template segment<POSE_SIZE>(POSE_SIZE * cam_idx);
    }
    x_pose_reduced.tail(padding_size_).setConstant(0);

    const auto block = qr_cache.bottomLeftCorner(num_cache_rows_ - 3,
                                                 padding_idx_ + padding_size_);

    const VecX tmp = block * x_pose_reduced;
    x_pose_reduced.noalias() = block.adjoint() * tmp;

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      const size_t cam_idx = pose_idx_[i];

      if (pose_mutex == nullptr) {
        res.template segment<POSE_SIZE>(POSE_SIZE * cam_idx) +=
            x_pose_reduced.template segment<POSE_SIZE>(POSE_SIZE * i);
      } else {
        std::scoped_lock lock(pose_mutex->at(cam_idx));

        res.template segment<POSE_SIZE>(POSE_SIZE * cam_idx) +=
            x_pose_reduced.template segment<POSE_SIZE>(POSE_SIZE * i);
      }
    }
  }

  void stage2(Scalar lambda, const VecX* jacobian_scaling,
              BlockDiagonalAccumulator<Scalar>* precond_block_diagonal,
              VecX& bref, Scalar* cache_ptr) {
    RowMatX cache_mat;
    if (!cache_ptr) {
      cache_mat.resize(num_cache_rows_, num_cache_cols_);
      cache_ptr = cache_mat.data();
    }

    // 1. scale jacobian
    if (jacobian_scaling) {
      scale_Jp_cols(*jacobian_scaling);
    }

    // 2. dampen landmarks
    set_landmark_damping(lambda);

    // 3. fill landmark damping and qr the whole landmark block
    perform_qr(cache_ptr);

    // 4. compute block diagonal preconditioner (SCHUR_JACOBI)
    if (precond_block_diagonal) {
      add_Q2TJp_T_Q2TJp_blockdiag(*precond_block_diagonal, nullptr, cache_ptr);
    }

    // 5. compute rhs of reduced camera normal equations
    add_Q2TJp_T_Q2Tr(bref, cache_ptr);
  }

  void add_factor(FactorQRBlock<Scalar>& factor_block,
                  BlockDiagonalAccumulator<Scalar>& precond_block_diagonal,
                  Scalar* cache_ptr) {
    ROOTBA_ASSERT(cache_ptr);

    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);

    const auto block = qr_cache.bottomLeftCorner(num_cache_rows_ - 3,
                                                 padding_idx_ + padding_size_);

    // qr_cache is row major, constructing the whole block could be better
    const MatX squared_block = block.transpose() * block;

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t cam_i = pose_idx_[i];

      for (size_t j = 0; j < pose_idx_.size(); ++j) {
        const size_t cam_j = pose_idx_[j];
        const auto h = squared_block.block(i * POSE_SIZE, j * POSE_SIZE,
                                           POSE_SIZE, POSE_SIZE);

        factor_block.add(cam_i, cam_j, h);
        if (i == j) {
          precond_block_diagonal.add(cam_j, h);  // move?
        }
      }
    }
  }

  void stage2(Scalar lambda, const VecX* jacobian_scaling,
              BlockDiagonalAccumulator<Scalar>& precond_block_diagonal,
              VecX& bref, FactorQRBlock<Scalar>& factor_block,
              Scalar* cache_ptr) {
    ROOTBA_ASSERT(cache_ptr);

    // 1. scale jacobian
    if (jacobian_scaling) {
      scale_Jp_cols(*jacobian_scaling);
    }

    // 2. dampen landmarks
    set_landmark_damping(lambda);

    // 3. fill landmark damping and qr the whole landmark block
    perform_qr(cache_ptr);

    // 4. fill factor block and compute block diagonal preconditioner
    // (SCHUR_JACOBI)
    add_factor(factor_block, precond_block_diagonal, cache_ptr);

    // 5. compute rhs of reduced camera normal equations
    add_Q2TJp_T_Q2Tr(bref, cache_ptr);
  }

  void back_substitute(const VecX& pose_inc, Scalar& l_diff,
                       Scalar* cache_ptr) {
    RowMatX cache_mat;
    if (!cache_ptr) {
      cache_mat.resize(num_cache_rows_, num_cache_cols_);
      cache_ptr = cache_mat.data();
    }
    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);

    std::vector<Eigen::JacobiRotation<Scalar>> damping_rotations;
    perform_qr(damping_rotations, cache_ptr);

    VecX pose_inc_reduced(padding_idx_ + padding_size_);
    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx = pose_idx_[i];
      pose_inc_reduced.template segment<POSE_SIZE>(POSE_SIZE * i) =
          pose_inc.template segment<POSE_SIZE>(POSE_SIZE * cam_idx);
    }
    pose_inc_reduced.tail(padding_size_).setConstant(0);

    const auto Q1T_Jl = qr_cache.template block<3, 3>(0, cache_lm_idx_)
                            .template triangularView<Eigen::Upper>();

    const auto Q1T_Jp = qr_cache.topLeftCorner(3, padding_idx_ + padding_size_);
    const auto Q1T_r = qr_cache.col(cache_res_idx_).template head<3>();

    Vec3 inc = -Q1T_Jl.solve(Q1T_r + Q1T_Jp * pose_inc_reduced);

    // We want to compute the model cost change. The model function is
    //
    //     L(inc) = F(x) + incT JT r + 0.5 incT JT J inc
    //
    // and thus the expect decrease in cost for the computed increment is
    //
    //     l_diff = L(0) - L(inc)
    //            = - incT JT r - 0.5 incT JT J inc.
    //            = - incT JT (r + 0.5 J inc)
    //            = - (J inc)T (r + 0.5 (J inc))

    // TODO@demmeln(LOW, Niko): compute model cost change from small landmark
    // block (Jp / Jl / r before marginalization). --> no need to save givens
    // rotations.

    // undo damping before we compute the model cost difference
    undo_landmark_damping(damping_rotations, cache_ptr);

    VecX QT_J_inc = qr_cache.topLeftCorner(num_cache_rows_ - 3,
                                           padding_idx_ + padding_size_) *
                    pose_inc_reduced;

    QT_J_inc.template head<3>() += Q1T_Jl * inc;

    auto QT_r = qr_cache.col(cache_res_idx_).head(num_cache_rows_ - 3);
    l_diff -= QT_J_inc.transpose() * (0.5 * QT_J_inc + QT_r);

    // TODO: detect and handle case like ceres, allowing a few iterations but
    // stopping eventually
    if (!inc.array().isFinite().all() ||
        !lm_ptr_->p_w.array().isFinite().all()) {
      std::cout << "=================================" << std::endl;
      std::cout << "inc\n" << inc.transpose() << std::endl;
      std::cout << "lm_ptr->p_w\n" << lm_ptr_->p_w.transpose() << std::endl;
      // std::cout << "QT_Jl\n" << QT_Jl << std::endl;
      // std::cout << "get_Q1Tr\n" << get_Q1Tr().transpose() << std::endl;
      // std::cout << "get_Q1TJp_postmult_x(pose_inc)\n"
      //           << get_Q1TJp_postmult_x(pose_inc).transpose() << std::endl;

      std::cout << "Storage_lm\n" << qr_cache.rightCols(4) << std::endl;

      std::cout.flush();
      LOG(FATAL) << "Numerical failure";
    }

    // Note: scale only after computing model cost change
    inc.array() *= Jl_col_scale_.array();
    lm_ptr_->p_w += inc;
  }

  inline size_t num_rows() { return num_cache_rows_; }
  inline size_t num_cols() { return num_cache_cols_; }
  inline State get_state() const { return state_; }
  inline const std::vector<size_t>& get_pose_idx() const { return pose_idx_; }

 protected:
  inline void perform_qr(Scalar* cache_ptr) const {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    copy_to_cache(cache_ptr);
    perform_qr_householder(true, cache_ptr);
  }

  inline void perform_qr(
      std::vector<Eigen::JacobiRotation<Scalar>>& damping_rotations,
      Scalar* cache_ptr) const {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    copy_to_cache(cache_ptr);
    perform_qr_householder(false, cache_ptr);
    apply_landmark_damping(damping_rotations, cache_ptr);
  }

  inline void apply_landmark_damping(

      std::vector<Eigen::JacobiRotation<Scalar>>& damping_rotations,
      Scalar* cache_ptr) const {
    damping_rotations.clear();
    damping_rotations.reserve(6);

    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);

    qr_cache.template block<3, 3>(num_cache_rows_ - 3, cache_lm_idx_)
        .diagonal()
        .setConstant(sqrt(lambda_));

    // apply dampening and remember rotations to undo
    for (int n = 0; n < 3; n++) {
      for (int m = 0; m <= n; m++) {
        damping_rotations.emplace_back();
        damping_rotations.back().makeGivens(
            qr_cache(n, cache_lm_idx_ + n),
            qr_cache(num_cache_rows_ - 3 + n - m, cache_lm_idx_ + n));
        qr_cache.applyOnTheLeft(num_cache_rows_ - 3 + n - m, n,
                                damping_rotations.back());
      }
    }
  }

  inline void undo_landmark_damping(

      std::vector<Eigen::JacobiRotation<Scalar>>& damping_rotations,
      Scalar* cache_ptr) const {
    ROOTBA_ASSERT(damping_rotations.size() == 6);

    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);

    // undo dampening
    for (int n = 2; n >= 0; n--) {
      for (int m = n; m >= 0; m--) {
        qr_cache.applyOnTheLeft(num_cache_rows_ - 3 + n - m, n,
                                damping_rotations.back().adjoint());
        damping_rotations.pop_back();
      }
    }

    qr_cache.template block<3, 3>(num_cache_rows_ - 3, cache_lm_idx_)
        .diagonal()
        .setZero();
  }

  inline void copy_to_cache(Scalar* cache_ptr) const {
    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);
    qr_cache.setZero();

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t obs_idx = i * 2;
      const size_t pose_idx = i * POSE_SIZE;
      qr_cache.template block<2, POSE_SIZE>(obs_idx, pose_idx) =
          storage_.template block<2, POSE_SIZE>(obs_idx, 0);
    }

    qr_cache.template block(0, cache_lm_idx_, num_rows_, 4) =
        storage_.template block(0, lm_idx_, num_rows_, 4);
  }

  inline void perform_qr_householder(const bool damping,
                                     Scalar* cache_ptr) const {
    RowMatXMap qr_cache(cache_ptr, num_cache_rows_, num_cache_cols_);

    qr_cache.template block<3, 3>(num_cache_rows_ - 3, cache_lm_idx_)
        .diagonal()
        .setConstant(sqrt(lambda_));

    VecX temp_vector1(num_cache_cols_);
    VecX temp_vector2 =
        damping ? VecX(num_cache_rows_) : VecX(num_cache_rows_ - 3);

    for (size_t k = 0; k < 3; ++k) {
      const size_t remaining_rows =
          damping ? num_cache_rows_ - k : num_cache_rows_ - k - 3;

      Scalar beta;
      Scalar tau;
      qr_cache.col(cache_lm_idx_ + k)
          .segment(k, remaining_rows)
          .makeHouseholder(temp_vector2, tau, beta);

      qr_cache.block(k, 0, remaining_rows, num_cache_cols_)
          .applyHouseholderOnTheLeft(temp_vector2, tau, temp_vector1.data());
    }
  }

  // Dense storage for pose Jacobians, padding, landmark Jacobians and
  // residuals [J_p | pad | J_l | res]
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      storage_;

  std::vector<size_t> pose_idx_;

  // indices and sizes of "small" landmark block that is stored
  size_t lm_idx_;
  size_t res_idx_;
  size_t num_cols_;
  size_t num_rows_;

  // indices and sizes of "big" landmark block that is temporarily computed in
  // scratch memory ("cache")
  size_t cache_lm_idx_;
  size_t cache_res_idx_;
  size_t num_cache_cols_;
  size_t num_cache_rows_;

  size_t padding_idx_ = 0;
  size_t padding_size_ = 0;

  Scalar lambda_ = 0;
  Vec3 Jl_col_scale_;

  Options options_;

  State state_ = State::UNINITIALIZED;

  Landmark* lm_ptr_ = nullptr;
};
}  // namespace rootba
