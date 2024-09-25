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

#include <fstream>
#include <mutex>
#include <cmath>



#include <Eigen/Dense>

#include "rootba/bal/bal_bundle_adjustment_helper.hpp"
#include "rootba/bal/bal_problem.hpp"
#include "rootba/bal/bal_residual_options.hpp"
#include "rootba/bal/solver_options.hpp"
#include "rootba/cg/block_sparse_matrix.hpp"
#include "rootba/util/assert.hpp"
#include "rootba/util/format.hpp"

namespace rootba {
template <typename, int>
class FactorSCBlock;

template <typename Scalar, int POSE_SIZE>
class LandmarkBlockSC {
 public:
  struct Options {
    // use Householder instead of Givens for marginalization
    bool use_householder = true;

    // use_valid_projections_only: if true, set invalid projection's
    // residual and jacobian to 0; invalid means z <= 0
    bool use_valid_projections_only = true;

    // huber norm with given threshold, else squared norm
    BalResidualOptions residual_options;

    // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
    // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
    Scalar jacobi_scaling_eps = 1.0;

    // TODO@demmeln: Remove preconditioner_type; I don't think we need this. And
    // the idea was also that this is independent of solver and SolverOptions
    // (SolverOptions is used only in the "solver" module, not "sc" or "qr").
    // PS: if we were to keep it, it would need a default value.

    // JACOBI or SCHUR_JACOBI
    SolverOptions::PreconditionerType preconditioner_type;


    // 0: parallel_reduce (may use more memory)
    // 1: parallel_for with mutex
    int reduction_alg = 1;

    // Precompute Jp'Jl for computing and preconditioner right_mulitply when
    // SCHUR_JACOBI is selected (only implicit and factor sc solvers)
    bool cache_hessian_blocks = false;

    bool jp_t_jl_on_the_fly = false;

    // Try to merge the remaining non-factor landmarks into the exisitng factors
    // group as the last phase.
    bool merge_factor = true;

    size_t max_factor_size = std::numeric_limits<size_t>::max();
  };

  enum State { UNINITIALIZED = 0, ALLOCATED, NUMERICAL_FAILURE, LINEARIZED };

  inline bool is_numerical_failure() const {
    return state_ == NUMERICAL_FAILURE;
  }

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using Vec5 = Eigen::Matrix<Scalar, 5, 1>;
  using Vec12 = Eigen::Matrix<Scalar, 12, 1>;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat11 = Eigen::Matrix<Scalar, 1,1>;
  using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
  using Mat4 = Eigen::Matrix<Scalar,4,4>;
  using Mat36 = Eigen::Matrix<Scalar, 3, 6>;

  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMatX =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  using Landmark = typename BalProblem<Scalar>::Landmark;
  using Camera = typename BalProblem<Scalar>::Camera;
  using Landmarks = typename BalProblem<Scalar>::Landmarks;
  using Cameras = typename BalProblem<Scalar>::Cameras;

  inline void allocate_landmark(Landmark& lm, const Options& options) {
    options_ = options;

    pose_idx_.clear();
    pose_idx_.reserve(lm.obs.size());
    for (const auto& [cam_idx, obs] : lm.obs) {
      pose_idx_.push_back(cam_idx);
    }

    num_rows_ = pose_idx_.size() * 2;



    //lm_idx_ = 8;
      //lm_idx_ = 11;
      lm_idx_ = 9;
      lm_idx_affine_ = 11;
      res_idx_affine_ = lm_idx_affine_ + 3;
      obs_idx_affine_ = res_idx_affine_ + 1;
      num_cols_affine_ = obs_idx_affine_ + 1;
      lm_idx_homogeneous_ = 15;
    //lm_idx_ = POSE_SIZE;
    res_idx_ = lm_idx_ + 3;
    res_idx_homogeneous_ = lm_idx_homogeneous_ + 4;
    //num_cols_ = res_idx_ + 1;
    //obs_idx_ = res_idx_ + 2; // @Simon: we store obs
    obs_idx_ = res_idx_ + 1;
    obs_idx_homogeneous_ = res_idx_homogeneous_ + 1;
    num_cols_ = obs_idx_ + 1;
    num_cols_homogeneous_ = obs_idx_homogeneous_ + 1;
    //storage_.resize(num_rows_, num_cols_);

    lm_idx_nullspace_ = 11;
    //res_idx_nullspace_ = lm_idx_nullspace_ + 3;
    num_cols_nullspace_ = lm_idx_nullspace_ + 3;
    num_rows_nullspace_ = pose_idx_.size() * 2;

    num_rows_pOSE_ = pose_idx_.size() * 4;
    lm_idx_pOSE_ = 15;
    res_idx_pOSE_ = lm_idx_pOSE_ + 3;
    obs_idx_pOSE_ = res_idx_pOSE_ + 1;
    num_cols_pOSE_ = obs_idx_pOSE_ + 1;

      num_rows_RpOSE_ = pose_idx_.size() * 3;
      lm_idx_RpOSE_ = 11;
      res_idx_RpOSE_ = lm_idx_RpOSE_ + 3;
      obs_idx_RpOSE_ = res_idx_RpOSE_ + 1;
      num_cols_RpOSE_ = obs_idx_RpOSE_ + 1;

      num_rows_RpOSE_ML_ = pose_idx_.size();
      lm_idx_RpOSE_ML_ = 11;
      res_idx_RpOSE_ML_ = lm_idx_RpOSE_ML_ + 3;
      obs_idx_RpOSE_ML_ = res_idx_RpOSE_ML_ + 1;
      num_cols_RpOSE_ML_ = obs_idx_RpOSE_ML_ + 1;


      num_rows_expOSE_ = pose_idx_.size() * 3;
      lm_idx_expOSE_ = 15;
      res_idx_expOSE_ = lm_idx_expOSE_ + 3;
      obs_idx_expOSE_ = res_idx_expOSE_ + 1;
      num_cols_expOSE_ = obs_idx_expOSE_ + 1;

    num_rows_pOSE_rOSE_ = pose_idx_.size() * 5;
    lm_idx_pOSE_rOSE_ = 15;
    res_idx_pOSE_rOSE_ = lm_idx_pOSE_rOSE_ + 3;
    obs_idx_pOSE_rOSE_ = res_idx_pOSE_rOSE_ + 1;
    num_cols_pOSE_rOSE_ = obs_idx_pOSE_rOSE_ + 1;

    num_rows_rOSE_ = pose_idx_.size() * 3;
    lm_idx_rOSE_ = 15;
    res_idx_rOSE_ = lm_idx_rOSE_ + 3;
    obs_idx_rOSE_ = res_idx_rOSE_ + 1;
    num_cols_rOSE_ = obs_idx_rOSE_ + 1;

    num_rows_pOSE_homogeneous_ = pose_idx_.size() * 4;
    lm_idx_pOSE_homogeneous_ = 15;
    res_idx_pOSE_homogeneous_ = lm_idx_pOSE_homogeneous_ + 4;
    obs_idx_pOSE_homogeneous_ = res_idx_pOSE_homogeneous_ + 1;
    num_cols_pOSE_homogeneous_ = obs_idx_pOSE_homogeneous_ + 1;

    storage_.resize(num_rows_, num_cols_homogeneous_);
    storage_affine_.resize(num_rows_, num_cols_affine_);
    storage_pOSE_.resize(num_rows_pOSE_, num_cols_pOSE_);
      storage_RpOSE_.resize(num_rows_RpOSE_, num_cols_RpOSE_);
      storage_RpOSE_ML_.resize(num_rows_RpOSE_ML_, num_cols_RpOSE_ML_);
    storage_expOSE_.resize(num_rows_expOSE_, num_cols_expOSE_);
    storage_pOSE_rOSE_.resize(num_rows_pOSE_rOSE_, num_cols_pOSE_rOSE_);
    storage_rOSE_.resize(num_rows_rOSE_, num_cols_rOSE_);
    storage_pOSE_homogeneous_.resize(num_rows_pOSE_homogeneous_, num_cols_pOSE_homogeneous_);

    storage_nullspace_.resize(num_rows_nullspace_, num_cols_nullspace_);

    state_ = ALLOCATED;

    lm_ptr_ = &lm;
  }

  // may set state to NumericalFailure --> linearization at this state is
  // unusable. Numeric check is only performed for residuals that were
  // considered to be used (valid), which depends on use_valid_projections_only
  // setting.
  inline void linearize_landmark(const Cameras& cameras) {
    ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                  state_ == LINEARIZED);

    storage_.setZero(num_rows_, num_cols_);

    bool numerically_valid = true;

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx = pose_idx_[i];
      size_t obs_idx = i * 2;
      size_t pose_idx = 0;

      const auto& obs = lm_ptr_->obs.at(cam_idx);
      const auto& cam = cameras.at(cam_idx);

      typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
      typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
      typename BalBundleAdjustmentHelper<Scalar>::MatRL Jl;

      Vec2 res;
      const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point(
          obs.pos, lm_ptr_->p_w, cam.T_c_w, cam.intrinsics, true, res , &Jp, &Ji,
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

        storage_.template block<2, 6>(obs_idx, pose_idx) = sqrt_weight * Jp;
        storage_.template block<2, 3>(obs_idx, pose_idx + 6) = sqrt_weight * Ji;
        storage_.template block<2, 3>(obs_idx, lm_idx_) = sqrt_weight * Jl;
        storage_.template block<2, 1>(obs_idx, res_idx_) = sqrt_weight * res;
        storage_.template block<2, 1>(obs_idx, obs_idx_) = sqrt_weight * obs.pos; //@Simon: do we really need sqrt_weight?
      }
    }

    if (numerically_valid) {
      state_ = LINEARIZED;
    } else {
      state_ = NUMERICAL_FAILURE;
    }
  }

    inline void linearize_landmark_affine_space(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_affine_.setZero(num_rows_, num_cols_affine_);

        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 2;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_affine_space Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL Jl;

            Vec3 v_init;

            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_affine_space(
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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

                storage_affine_.template block<2, 8>(obs_idx, pose_idx) = sqrt_weight * Jp;
                //storage_.template block<2,3>(obs_idx, pose_idx + 8).setZero(); // = 0;
                storage_affine_.template block<2, 3>(obs_idx, pose_idx + 8) = sqrt_weight * Ji;
                storage_affine_.template block<2, 3>(obs_idx, lm_idx_affine_) = sqrt_weight * Jl;
                storage_affine_.template block<2, 1>(obs_idx, res_idx_affine_) = sqrt_weight * res;
                storage_affine_.template block<2, 1>(obs_idx, obs_idx_affine_) = sqrt_weight * obs.pos; //@Simon: do we really need sqrt_weight?
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_pOSE(const Cameras& cameras, int alpha) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_pOSE_.setZero(num_rows_pOSE_, num_cols_pOSE_);
        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 4;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI_pOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE Jl;

            Vec3 v_init;
            Vec4 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE(alpha,
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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
                storage_pOSE_.template block<4, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                //storage_.template block<2,3>(obs_idx, pose_idx + 8).setZero(); // = 0;
                //std::cout << "in landmark_block    l247     Jp.norm() = " << Jp.norm() << "\n";
                storage_pOSE_.template block<4, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_pOSE_.template block<4, 3>(obs_idx, lm_idx_pOSE_) = sqrt_weight * Jl;
                storage_pOSE_.template block<4, 1>(obs_idx, res_idx_pOSE_) = sqrt_weight * res;
                //storage_pOSE_.template block<2, 1>(obs_idx, obs_idx_pOSE_).setZero();
                storage_pOSE_.template block<2, 1>(obs_idx+2, obs_idx_pOSE_) = sqrt_weight * obs.pos;
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_RpOSE(const Cameras& cameras, double alpha) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_RpOSE_.setZero(num_rows_RpOSE_, num_cols_RpOSE_);
        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 3;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_RpOSE Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI_RpOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_RpOSE Jl;

            Vec3 v_init;
            Vec3 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_RpOSE(alpha,
                                                                                       obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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
                storage_RpOSE_.template block<3, 8>(obs_idx, pose_idx) = sqrt_weight * Jp;
                //storage_.template block<2,3>(obs_idx, pose_idx + 8).setZero(); // = 0;
                //std::cout << "in landmark_block    l247     Jp.norm() = " << Jp.norm() << "\n";
                storage_RpOSE_.template block<3, 3>(obs_idx, pose_idx + 8) = sqrt_weight * Ji;
                storage_RpOSE_.template block<3, 3>(obs_idx, lm_idx_RpOSE_) = sqrt_weight * Jl;
                storage_RpOSE_.template block<3, 1>(obs_idx, res_idx_RpOSE_) = sqrt_weight * res;
                //storage_pOSE_.template block<2, 1>(obs_idx, obs_idx_pOSE_).setZero();
                //storage_RpOSE_.template block<2, 1>(obs_idx+2, obs_idx_RpOSE_) = sqrt_weight * obs.pos;
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_RpOSE_refinement(const Cameras& cameras, double alpha) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_RpOSE_.setZero(num_rows_RpOSE_, num_cols_RpOSE_);
        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 3;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_RpOSE Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI_RpOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_RpOSE Jl;

            Vec3 v_init;
            Vec3 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_RpOSE_refinement(alpha,
                                                                                        obs.pos, obs.rpose_eq, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res , &Jp, &Ji,
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
                storage_RpOSE_.template block<3, 8>(obs_idx, pose_idx) = sqrt_weight * Jp;
                //storage_.template block<2,3>(obs_idx, pose_idx + 8).setZero(); // = 0;
                //std::cout << "in landmark_block    l247     Jp.norm() = " << Jp.norm() << "\n";
                storage_RpOSE_.template block<3, 3>(obs_idx, pose_idx + 8) = sqrt_weight * Ji;
                storage_RpOSE_.template block<3, 3>(obs_idx, lm_idx_RpOSE_) = sqrt_weight * Jl;
                storage_RpOSE_.template block<3, 1>(obs_idx, res_idx_RpOSE_) = sqrt_weight * res;
                //storage_pOSE_.template block<2, 1>(obs_idx, obs_idx_pOSE_).setZero();
                //storage_RpOSE_.template block<2, 1>(obs_idx+2, obs_idx_RpOSE_) = sqrt_weight * obs.pos;
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_RpOSE_ML(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_RpOSE_ML_.setZero(num_rows_RpOSE_ML_, num_cols_RpOSE_ML_);
        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_RpOSE_ML Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI_RpOSE_ML Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_RpOSE_ML Jl;

            Mat11 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_RpOSE_ML(obs.pos, obs.rpose_eq, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res , &Jp, &Ji,
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
                storage_RpOSE_ML_.template block<1, 8>(obs_idx, pose_idx) = sqrt_weight * Jp;
                //storage_.template block<2,3>(obs_idx, pose_idx + 8).setZero(); // = 0;
                //std::cout << "in landmark_block    l542     Jp.norm() = " << Jp.norm() << "\n";
                //storage_RpOSE_ML_.template block<1, 3>(obs_idx, pose_idx + 8) = sqrt_weight * Ji;
                storage_RpOSE_ML_.template block<1, 3>(obs_idx, lm_idx_RpOSE_ML_) = sqrt_weight * Jl;
                storage_RpOSE_ML_.template block<1, 1>(obs_idx, res_idx_RpOSE_ML_) = sqrt_weight * res;
                //storage_pOSE_.template block<2, 1>(obs_idx, obs_idx_pOSE_).setZero();
                //storage_RpOSE_.template block<2, 1>(obs_idx+2, obs_idx_RpOSE_) = sqrt_weight * obs.pos;
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }


    inline void linearize_landmark_expOSE(const Cameras& cameras, int alpha, bool init) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_expOSE_.setZero(num_rows_expOSE_, num_cols_expOSE_);
        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 3;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_expOSE Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI_expOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_expOSE Jl;

            Vec3 v_init;
            Vec3 res;
            //std::cout << "lm_ptr_->p_w = " << lm_ptr_->p_w << "\n";
            //std::cout << "lm_ptr_->p_w_backup() = " << lm_ptr_->p_w_backup() << "\n";
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_expOSE(alpha,
                                                                                       obs.pos, obs.y_tilde, lm_ptr_->p_w, lm_ptr_->p_w_backup(), cam.space_matrix, cam.space_matrix_backup(), cam.intrinsics, true, res, init , &Jp, &Ji,
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
                storage_expOSE_.template block<3, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_expOSE_.template block<3, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_expOSE_.template block<3, 3>(obs_idx, lm_idx_expOSE_) = sqrt_weight * Jl;
                storage_expOSE_.template block<3, 1>(obs_idx, res_idx_expOSE_) = sqrt_weight * res;
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }


    inline void linearize_landmark_pOSE_rOSE(const Cameras& cameras, int alpha) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_pOSE_rOSE_.setZero(num_rows_pOSE_rOSE_, num_cols_pOSE_rOSE_);
        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 5;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE_rOSE Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI_pOSE_rOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE_rOSE Jl;

            Vec3 v_init;
            Vec5 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE_rOSE(alpha,
                                                                                       obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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
                storage_pOSE_rOSE_.template block<5, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_pOSE_rOSE_.template block<5, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_pOSE_rOSE_.template block<5, 3>(obs_idx, lm_idx_pOSE_rOSE_) = sqrt_weight * Jl;
                storage_pOSE_rOSE_.template block<5, 1>(obs_idx, res_idx_pOSE_rOSE_) = sqrt_weight * res;
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_rOSE(const Cameras& cameras, int alpha) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_rOSE_.setZero(num_rows_rOSE_, num_cols_rOSE_);
        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 3;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_rOSE Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI_rOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_rOSE Jl;

            Vec3 v_init;
            Vec3 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_rOSE(alpha,
                                                                                       obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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
                storage_rOSE_.template block<3, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_rOSE_.template block<3, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_rOSE_.template block<3, 3>(obs_idx, lm_idx_rOSE_) = sqrt_weight * Jl;
                storage_rOSE_.template block<3, 1>(obs_idx, res_idx_rOSE_) = sqrt_weight * res;
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_pOSE_homogeneous(const Cameras& cameras, int alpha) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_pOSE_homogeneous_.setZero(num_rows_pOSE_homogeneous_, num_cols_pOSE_homogeneous_);
        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 4;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE Jp;

            typename BalBundleAdjustmentHelper<Scalar>::MatRI_pOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE_homogeneous Jl;

            Vec4 v_init;
            Vec4 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE_homogeneous(alpha,
                                                                                       obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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
                storage_pOSE_homogeneous_.template block<4, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_pOSE_homogeneous_.template block<4, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_pOSE_homogeneous_.template block<4, 4>(obs_idx, lm_idx_pOSE_homogeneous_) = sqrt_weight * Jl;
                storage_pOSE_homogeneous_.template block<4, 1>(obs_idx, res_idx_pOSE_homogeneous_) = sqrt_weight * res;
                storage_pOSE_homogeneous_.template block<2, 1>(obs_idx+2, obs_idx_pOSE_homogeneous_) = sqrt_weight * obs.pos;
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_projective_space(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_.setZero(num_rows_, num_cols_);

        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 2;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL Jl;

            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space(
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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

                storage_.template block<2, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_.template block<2, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_.template block<2, 3>(obs_idx, lm_idx_) = sqrt_weight * Jl;
                storage_.template block<2, 1>(obs_idx, res_idx_) = sqrt_weight * res;
                storage_.template block<2, 1>(obs_idx, obs_idx_) = sqrt_weight * obs.pos; //@Simon: do we really need sqrt_weight?
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_projective_space_homogeneous(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_homogeneous_.setZero(num_rows_, num_cols_homogeneous_);

        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 2;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_projective_space_homogeneous Jl;
            Vec2 res;

            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space_homogeneous(
                    obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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

                storage_homogeneous_.template block<2, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_homogeneous_.template block<2, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_homogeneous_.template block<2, 4>(obs_idx, lm_idx_homogeneous_) = sqrt_weight * Jl;
                storage_homogeneous_.template block<2, 1>(obs_idx, res_idx_homogeneous_) = sqrt_weight * res;
                storage_homogeneous_.template block<2, 1>(obs_idx, obs_idx_homogeneous_) = sqrt_weight * obs.pos; //@Simon: do we really need sqrt_weight?
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_landmark_projective_space_homogeneous_RpOSE(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_homogeneous_.setZero(num_rows_, num_cols_homogeneous_);

        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 2;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_projective_space_homogeneous Jl;
            Vec2 res;

            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space_homogeneous_RpOSE(
                    obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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

                storage_homogeneous_.template block<2, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_homogeneous_.template block<2, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_homogeneous_.template block<2, 4>(obs_idx, lm_idx_homogeneous_) = sqrt_weight * Jl;
                storage_homogeneous_.template block<2, 1>(obs_idx, res_idx_homogeneous_) = sqrt_weight * res;
                storage_homogeneous_.template block<2, 1>(obs_idx, obs_idx_homogeneous_) = sqrt_weight * obs.pos; //@Simon: do we really need sqrt_weight?
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }



    inline void linearize_landmark_projective_space_homogeneous_storage(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_.setZero(num_rows_, num_cols_homogeneous_);

        storage_nullspace_.setZero(num_rows_nullspace_, num_cols_nullspace_);

        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 2;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            //typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_projective_space_homogeneous Jl;
            Vec2 res;

            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space_homogeneous(
                    obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, false , &Jp, &Ji,
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

                storage_.template block<2, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_.template block<2, 3>(obs_idx, pose_idx + 12) = sqrt_weight * Ji;
                storage_.template block<2, 4>(obs_idx, lm_idx_) = sqrt_weight * Jl;
                storage_.template block<2, 1>(obs_idx, res_idx_homogeneous_) = sqrt_weight * res;
                storage_.template block<2, 1>(obs_idx, obs_idx_homogeneous_) = sqrt_weight * obs.pos; //@Simon: do we really need sqrt_weight?

                auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
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

                //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
                auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                //auto jp_proj = jp * Proj_pose;


                //auto jl_proj = sqrt_weight * Jl * Proj;
                storage_nullspace_.template block<2, 11>(obs_idx, pose_idx) = sqrt_weight * Jp * Proj_pose;
                storage_nullspace_.template block<2, 3>(obs_idx, lm_idx_nullspace_) = sqrt_weight * Jl * Proj;


            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }

    inline void linearize_nullspace(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_nullspace_.setZero(num_rows_nullspace_, num_cols_nullspace_);

        bool numerically_valid = true;
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 2;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

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

            storage_nullspace_.template block<2, 11>(obs_idx, pose_idx) = storage_homogeneous_.template block<2, 12>(obs_idx, pose_idx) * Proj_pose;
            storage_nullspace_.template block<2, 3>(obs_idx, lm_idx_nullspace_) = storage_homogeneous_.template block<2, 4>(obs_idx, lm_idx_homogeneous_) * Proj;

        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }



    inline void linearize_landmark_refine(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == ALLOCATED || state_ == NUMERICAL_FAILURE ||
                      state_ == LINEARIZED);

        storage_.setZero(num_rows_, num_cols_);

        bool numerically_valid = true;

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            size_t obs_idx = i * 2;
            size_t pose_idx = 0;

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL Jl;

            Vec3 v_init;

            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_refine(
                    obs.pos, lm_ptr_->p_w, cam.T_c_w, cam.intrinsics, true, res , &Jp, &Ji,
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

                storage_.template block<2, 6>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_.template block<2, 3>(obs_idx, pose_idx + 6) = sqrt_weight * Ji;
                storage_.template block<2, 3>(obs_idx, lm_idx_) = sqrt_weight * Jl;
                storage_.template block<2, 1>(obs_idx, res_idx_) = sqrt_weight * res;
                storage_.template block<2, 1>(obs_idx, obs_idx_) = sqrt_weight * obs.pos; //@Simon: do we really need sqrt_weight?
            }
        }

        if (numerically_valid) {
            state_ = LINEARIZED;
        } else {
            state_ = NUMERICAL_FAILURE;
        }
    }



    inline void add_Jp_diag2(VecX& res) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx = pose_idx_[i];
      res.template segment<9>(9 * cam_idx) +=
          storage_.template block<2, 9>(2 * i, 0)
              .colwise()
              .squaredNorm();
    }
  }

    inline void add_Jp_diag2_affine_space(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<11>(11 * cam_idx) +=
                    storage_affine_.template block<2, 11>(2 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }

    inline void add_Jp_diag2_pOSE(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<15>(15 * cam_idx) +=
                    storage_pOSE_.template block<4, 15>(4 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }

    inline void add_Jp_diag2_RpOSE(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<11>(11 * cam_idx) +=
                    storage_RpOSE_.template block<3, 11>(3 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }

    inline void add_Jp_diag2_RpOSE_ML(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<11>(11 * cam_idx) +=
                    storage_RpOSE_ML_.template block<1, 11>(i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }


    inline void add_Jp_diag2_expOSE(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<15>(15 * cam_idx) +=
                    storage_expOSE_.template block<3, 15>(3 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }


    inline void add_Jp_diag2_pOSE_rOSE(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<15>(15 * cam_idx) +=
                    storage_pOSE_rOSE_.template block<5, 15>(5 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }

    inline void add_Jp_diag2_rOSE(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<15>(15 * cam_idx) +=
                    storage_rOSE_.template block<3, 15>(3 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }


    inline void add_Jp_diag2_pOSE_homogeneous(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<12>(12 * cam_idx) +=
                    storage_pOSE_homogeneous_.template block<4, 12>(4 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }

    inline void add_Jp_diag2_projective_space(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<15>(15 * cam_idx) +=
                    storage_homogeneous_.template block<2, 15>(2 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }

    inline void add_Jp_diag2_projective_space_RpOSE(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<15>(15 * cam_idx) +=
                    storage_homogeneous_.template block<2, 15>(2 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }

  inline void scale_Jl_cols() {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
    // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
    Jl_col_scale =
        (options_.jacobi_scaling_eps +
         storage_.block(0, lm_idx_, num_rows_, 3).colwise().norm().array())
            .inverse();

    storage_.block(0, lm_idx_, num_rows_, 3) *= Jl_col_scale.asDiagonal();
  }

    inline void scale_Jl_cols_pOSE() {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        Jl_col_scale_pOSE =
                (options_.jacobi_scaling_eps +
                 storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3).colwise().norm().array())
                        .inverse();

        storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3) *= Jl_col_scale_pOSE.asDiagonal();
    }

    inline void scale_Jl_cols_RpOSE() {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        Jl_col_scale_RpOSE =
                (options_.jacobi_scaling_eps +
                 storage_RpOSE_.block(0, lm_idx_RpOSE_, num_rows_RpOSE_, 3).colwise().norm().array())
                        .inverse();

        storage_RpOSE_.block(0, lm_idx_RpOSE_, num_rows_RpOSE_, 3) *= Jl_col_scale_RpOSE.asDiagonal();
    }

    inline void scale_Jl_cols_RpOSE_ML() {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        Jl_col_scale_RpOSE =
                (options_.jacobi_scaling_eps +
                 storage_RpOSE_ML_.block(0, lm_idx_RpOSE_ML_, num_rows_RpOSE_ML_, 3).colwise().norm().array())
                        .inverse();

        storage_RpOSE_ML_.block(0, lm_idx_RpOSE_ML_, num_rows_RpOSE_ML_, 3) *= Jl_col_scale_RpOSE.asDiagonal();
    }

    inline void scale_Jl_cols_expOSE() {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        Jl_col_scale_expOSE =
                (options_.jacobi_scaling_eps +
                 storage_expOSE_.block(0, lm_idx_expOSE_, num_rows_expOSE_, 3).colwise().norm().array())
                        .inverse();

        storage_expOSE_.block(0, lm_idx_expOSE_, num_rows_expOSE_, 3) *= Jl_col_scale_expOSE.asDiagonal();
    }


    inline void scale_Jl_cols_pOSE_homogeneous() {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        Jl_col_scale_pOSE_homogeneous =
                (options_.jacobi_scaling_eps +
                 storage_pOSE_homogeneous_.block(0, lm_idx_pOSE_homogeneous_, num_rows_pOSE_homogeneous_, 4).colwise().norm().array())
                        .inverse();

        storage_pOSE_homogeneous_.block(0, lm_idx_pOSE_homogeneous_, num_rows_pOSE_homogeneous_, 4) *= Jl_col_scale_pOSE_homogeneous.asDiagonal();
    }

    inline void scale_Jl_cols_homogeneous() {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        Jl_col_scale_homogeneous =
                (options_.jacobi_scaling_eps +
                 storage_homogeneous_.block(0, lm_idx_homogeneous_, num_rows_, 4).colwise().norm().array())
                        .inverse();

        storage_homogeneous_.block(0, lm_idx_homogeneous_, num_rows_, 4) *= Jl_col_scale_homogeneous.asDiagonal();
    }

    inline void scale_Jl_cols_homogeneous_RpOSE() {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        Jl_col_scale_homogeneous =
                (options_.jacobi_scaling_eps +
                 storage_homogeneous_.block(0, lm_idx_homogeneous_, num_rows_, 4).colwise().norm().array())
                        .inverse();

        storage_homogeneous_.block(0, lm_idx_homogeneous_, num_rows_, 4) *= Jl_col_scale_homogeneous.asDiagonal();
    }


    inline void scale_Jp_cols(const VecX& jacobian_scaling) {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx = pose_idx_[i];

      storage_.template block<2, 9>(2 * i, 0) *=
          jacobian_scaling.template segment<9>(9 * cam_idx)
              .asDiagonal();
    }
  }

    inline void scale_Jp_cols_joint(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_homogeneous_.template block<2, 15>(2 * i, 0) *=
                    jacobian_scaling.template segment<15>(15 * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_affine(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_affine_.template block<2, 11>(2 * i, 0) *=
                    jacobian_scaling.template segment<11>(11 * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_pOSE(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_pOSE_.template block<4, POSE_SIZE>(4 * i, 0) *=
                    jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_RpOSE(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_RpOSE_.template block<3, POSE_SIZE>(3 * i, 0) *=
                    jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_RpOSE_ML(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_RpOSE_ML_.template block<1, POSE_SIZE>(i, 0) *=
                    jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_expOSE(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_expOSE_.template block<3, POSE_SIZE>(3 * i, 0) *=
                    jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_pOSE_rOSE(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_pOSE_rOSE_.template block<5, POSE_SIZE>(5 * i, 0) *=
                    jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_rOSE(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_rOSE_.template block<3, POSE_SIZE>(3 * i, 0) *=
                    jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_pOSE_homogeneous(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_pOSE_homogeneous_.template block<4, 15>(4 * i, 0) *=
                    jacobian_scaling.template segment<15>(15 * cam_idx)
                            .asDiagonal();
        }
    }

    inline void scale_Jp_cols_projective(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_.template block<2, POSE_SIZE>(2 * i, 0) *=
                    jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
                            .asDiagonal();
        }
    }

  inline State get_state() const { return state_; }

  inline void set_landmark_damping(Scalar lambda) { lambda_ = lambda; }

  inline void set_landmark_damping_joint(Scalar lambda) { lambda_lm_landmark_ = lambda; }

  inline size_t num_poses() const { return pose_idx_.size(); }


  inline const std::vector<size_t>& get_pose_idx() const { return pose_idx_; }

  inline auto get_lm_ptr() const { return lm_ptr_; }

  inline auto get_Jli(const size_t obs_idx) const {
    return storage_.template block<2, 3>(2 * obs_idx, lm_idx_);
  }

    inline auto get_Jl() const {
        return storage_.template middleCols<3>(lm_idx_);
    }

    inline auto get_Jl_homogeneous() const {
        return storage_.template middleCols<4>(lm_idx_);
    }

    inline auto get_Jl_homogeneous_riemannian_manifold() const {
        //const auto jl = storage_.template middleCols<4>(lm_idx_);
        return storage_.template middleCols<4>(lm_idx_);
        //auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

        //return jl * Proj;
    }

    inline auto get_Jl_homogeneous_riemannian_manifold_storage() const {

        return storage_nullspace_.template middleCols<3>(lm_idx_nullspace_);

    }

    inline auto get_Jl_homogeneous_riemannian_manifold_storage_RpOSE() const {

        return storage_nullspace_.template middleCols<3>(lm_idx_nullspace_);

    }

  inline auto get_Jl_affine_space() const {
    return storage_affine_.template middleCols<3>(lm_idx_affine_);
  }

    inline auto get_Jl_pOSE() const {
        return storage_pOSE_.template middleCols<3>(lm_idx_pOSE_);
    }

    inline auto get_Jl_RpOSE() const {
        return storage_RpOSE_.template middleCols<3>(lm_idx_RpOSE_);
    }

    inline auto get_Jl_RpOSE_ML() const {
        return storage_RpOSE_ML_.template middleCols<3>(lm_idx_RpOSE_ML_);
    }


    inline auto get_Jl_expOSE() const {
        return storage_expOSE_.template middleCols<3>(lm_idx_expOSE_);
    }

    inline auto get_Jl_pOSE_rOSE() const {
        return storage_pOSE_rOSE_.template middleCols<3>(lm_idx_pOSE_rOSE_);
    }

    inline auto get_Jl_rOSE() const {
        return storage_rOSE_.template middleCols<3>(lm_idx_rOSE_);
    }

    inline auto get_Jl_pOSE_homogeneous() const {
        return storage_pOSE_homogeneous_.template middleCols<4>(lm_idx_pOSE_);
    }

  inline auto get_Jpi(const size_t obs_idx) const {
    return storage_.template block<2, 9>(2 * obs_idx, 0);
  }

    inline auto get_Jpi_affine_space(const size_t obs_idx) const {
        return storage_affine_.template block<2, 11>(2 * obs_idx, 0);
    }

    inline auto get_Jpi_pOSE(const size_t obs_idx) const {
        return storage_pOSE_.template block<4, 15>(4 * obs_idx, 0);
    }

    inline auto get_Jpi_RpOSE(const size_t obs_idx) const {
        return storage_RpOSE_.template block<3, 11>(3 * obs_idx, 0);
    }

    inline auto get_Jpi_RpOSE_ML(const size_t obs_idx) const {
        return storage_RpOSE_ML_.template block<1, 11>(obs_idx, 0);
    }

    inline auto get_Jpi_expOSE(const size_t obs_idx) const {
        return storage_expOSE_.template block<3, 15>(3 * obs_idx, 0);
    }


    inline auto get_Jpi_pOSE_rOSE(const size_t obs_idx) const {
        return storage_pOSE_rOSE_.template block<5, 15>(5 * obs_idx, 0);
    }

    inline auto get_Jpi_rOSE(const size_t obs_idx) const {
        return storage_rOSE_.template block<3, 15>(3 * obs_idx, 0);
    }

    inline auto get_Jpi_pOSE_riemannian_manifold(const size_t obs_idx) const {
        return storage_pOSE_.template block<4, 12>(4 * obs_idx, 0);
    }

    inline auto get_Jpi_pOSE_homogeneous(const size_t obs_idx) const {
        return storage_pOSE_homogeneous_.template block<4, 12>(4 * obs_idx, 0);
    }

    inline auto get_Jpi_projective_space(const size_t obs_idx) const {
        return storage_.template block<2, 15>(2 * obs_idx, 0);
    }

    inline auto get_Jpi_projective_space_riemannian_manifold(const size_t obs_idx, const Cameras& cameras) const {


        return storage_.template block<2, 12>(2 * obs_idx, 0);
    }


    inline auto get_Jpi_projective_space_riemannian_manifold_storage(const size_t obs_idx, const Cameras& cameras) const {

       return storage_nullspace_.template block<2, 11>(2 * obs_idx, 0);

    }

    inline auto get_Jpi_projective_space_riemannian_manifold_storage_RpOSE(const size_t obs_idx, const Cameras& cameras) const {

        RowMatX storage_tmp;
        storage_tmp.resize(2,13);
        storage_tmp.template block<2,11>(0, 0) = storage_nullspace_.template block<2, 11>(2 * obs_idx, 0);
        storage_tmp.template block<2, 2>(0, 11) = storage_homogeneous_.template block<2,2>(2*obs_idx, 13);
        return storage_tmp;

    }

  //inline auto get_Hll_inv() const { return Hll_inv_; }


  // Fill the explicit reduced H, b linear system by parallel_reduce
  inline void add_Hb(BlockSparseMatrix<Scalar>& accu, VecX& b) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // Compute landmark-landmark block plus inverse
    Mat3 H_ll;
    Mat3 H_ll_inv;
    Vec3 H_ll_inv_bl;
    {
      auto Jl = storage_.block(0, lm_idx_, num_rows_, 3);
      H_ll = Jl.transpose() * Jl;
      H_ll.diagonal().array() += lambda_;
      H_ll_inv_ = H_ll.inverse();

      H_ll_inv_bl =
          H_ll_inv_ *
          (Jl.transpose() *
           storage_.col(
               res_idx_));  // bl = Jl.transpose() * storage_.col(res_idx_)
    }

    // Add pose-pose blocks and
    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx_i = pose_idx_[i];

      auto jp_i = storage_.template block<2, 9>(
          2 * i, 0);

      auto jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
      auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);

      MatX H_pp = jp_i.transpose() * jp_i;
      accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));

      // Schur complement blocks
      for (size_t j = 0; j < pose_idx_.size(); j++) {
        size_t cam_idx_j = pose_idx_[j];
        auto jp_j = storage_.template block<2, 9>(2 * j, 0);
        auto jl_j = storage_.template block<2, 3>(2 * j, lm_idx_);

        MatX H_pl_H_ll_inv_H_lp =
            -jp_i.transpose() * (jl_i * (H_ll_inv_ * (jl_j.transpose() * jp_j)));

        accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
      }

      // add blocks to b
      b.template segment<9>(cam_idx_i * 9) +=
          jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
    }

  }

    inline void add_Hb_pOSE(BlockSparseMatrix<Scalar>& accu, VecX& b) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Mat3 H_ll;
        Mat3 H_ll_inv;
        Vec3 H_ll_inv_bl;
        {
            auto Jl = storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3);
            H_ll = Jl.transpose() * Jl;
            //H_ll.diagonal().array() += lambda_pOSE_;
            H_ll_inv_ = H_ll.inverse();

            H_ll_inv_bl =
                    H_ll_inv_ *
                    (Jl.transpose() *
                     storage_pOSE_.col(
                             res_idx_pOSE_));  // bl = Jl.transpose() * storage_.col(res_idx_)
        }

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx_i = pose_idx_[i];

            auto jp_i = storage_pOSE_.template block<4, POSE_SIZE>(
                    4 * i, 0);  /// WHY COL 0? Possibility: all blocks of Jp are in one
            /// column (aggregate fig. 2(b)
            auto jl_i = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            auto r_i = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);

            MatX H_pp = jp_i.transpose() * jp_i;
            accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                size_t cam_idx_j = pose_idx_[j];
                auto jp_j = storage_pOSE_.template block<4, POSE_SIZE>(4 * j, 0);
                auto jl_j = storage_pOSE_.template block<4, 3>(4 * j, lm_idx_pOSE_);

                MatX H_pl_H_ll_inv_H_lp =
                        -jp_i.transpose() * (jl_i * (H_ll_inv_ * (jl_j.transpose() * jp_j)));

                accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
            }

            // add blocks to b
            b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                    jp_i.transpose() * r_i;
            //b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
            //        jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
        }

    }

    inline void add_Hb_expOSE(BlockSparseMatrix<Scalar>& accu, VecX& b) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Mat3 H_ll;
        Mat3 H_ll_inv;
        Vec3 H_ll_inv_bl;
        {
            auto Jl = storage_expOSE_.block(0, lm_idx_expOSE_, num_rows_expOSE_, 3);
            H_ll = Jl.transpose() * Jl;
            //H_ll.diagonal().array() += lambda_expOSE_;
            H_ll_inv_ = H_ll.inverse();

            H_ll_inv_bl =
                    H_ll_inv_ *
                    (Jl.transpose() *
                     storage_expOSE_.col(
                             res_idx_expOSE_));  // bl = Jl.transpose() * storage_.col(res_idx_)
        }

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx_i = pose_idx_[i];

            auto jp_i = storage_expOSE_.template block<3, POSE_SIZE>(
                    3 * i, 0);  /// WHY COL 0? Possibility: all blocks of Jp are in one
            /// column (aggregate fig. 2(b)
            auto jl_i = storage_expOSE_.template block<3, 3>(3 * i, lm_idx_expOSE_);
            auto r_i = storage_expOSE_.template block<3, 1>(3 * i, res_idx_expOSE_);

            MatX H_pp = jp_i.transpose() * jp_i;
            accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                size_t cam_idx_j = pose_idx_[j];
                auto jp_j = storage_expOSE_.template block<3, POSE_SIZE>(3 * j, 0);
                auto jl_j = storage_expOSE_.template block<3, 3>(3 * j, lm_idx_expOSE_);

                MatX H_pl_H_ll_inv_H_lp =
                        -jp_i.transpose() * (jl_i * (H_ll_inv_ * (jl_j.transpose() * jp_j)));

                accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
            }

            // add blocks to b
            b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                    jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
        }

    }

    // Fill the explicit reduced H, b linear system by parallel_reduce
    inline void add_Hb_varproj(BlockSparseMatrix<Scalar>& accu, VecX& b) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Mat3 H_ll;
        Mat3 H_ll_inv;
        Vec3 H_ll_inv_bl;
        {
            auto Jl = storage_.block(0, lm_idx_, num_rows_, 3);
            H_ll = Jl.transpose() * Jl;
            //H_ll.diagonal().array() += lambda_;
            H_ll_inv_ = H_ll.inverse();

            H_ll_inv_bl =
                    H_ll_inv_ *
                    (Jl.transpose() *
                     storage_.col(
                             res_idx_));  // bl = Jl.transpose() * storage_.col(res_idx_)
        }

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx_i = pose_idx_[i];

            auto jp_i = storage_.template block<2, POSE_SIZE>(
                    2 * i, 0);  /// WHY COL 0? Possibility: all blocks of Jp are in one
            /// column (aggregate fig. 2(b)
            auto jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);

            MatX H_pp = jp_i.transpose() * jp_i;
            accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                size_t cam_idx_j = pose_idx_[j];
                auto jp_j = storage_.template block<2, POSE_SIZE>(2 * j, 0);
                auto jl_j = storage_.template block<2, 3>(2 * j, lm_idx_);

                MatX H_pl_H_ll_inv_H_lp =
                        -jp_i.transpose() * (jl_i * (H_ll_inv_ * (jl_j.transpose() * jp_j)));

                accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
            }

            // add blocks to b
            //b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
            //        jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
            b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                    jp_i.transpose() * r_i;
        }

    }

  // Fill the explicit reduced H, b linear system by parallel_for and mutex
  inline void add_Hb(BlockSparseMatrix<Scalar>& accu, VecX& b,
                     std::vector<std::mutex>& H_pp_mutex,
                     std::vector<std::mutex>& pose_mutex,
                     const size_t num_cam) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // Compute landmark-landmark block plus inverse
    Mat3 H_ll;
    Mat3 H_ll_inv;
    Vec3 H_ll_inv_bl;
    {
      auto Jl = storage_.block(0, lm_idx_, num_rows_, 3);
      H_ll = Jl.transpose() * Jl;
      H_ll.diagonal().array() += lambda_;
      //H_ll_inv_ = H_ll.inverse();
      H_ll_inv = H_ll.inverse();

      H_ll_inv_bl = H_ll_inv * (Jl.transpose() * storage_.col(res_idx_));
    }

    // Add pose-pose blocks and
    for (size_t i = 0; i < pose_idx_.size(); i++) {
      const size_t cam_idx_i = pose_idx_[i];

      auto jp_i = storage_.template block<2, 9>(2 * i, 0);
      auto jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
      auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);

      // TODO: check (Niko: not sure what we wanted to check here...)

      {
        MatX H_pp = jp_i.transpose() * jp_i;
        std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_i));
        accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));
      }

      // Schur complement blocks
      for (size_t j = 0; j < pose_idx_.size(); j++) {
        const size_t cam_idx_j = pose_idx_[j];
        auto jp_j = storage_.template block<2, 9>(2 * j, 0);
        auto jl_j = storage_.template block<2, 3>(2 * j, lm_idx_);

        {
          MatX H_pl_H_ll_inv_H_lp =
              -jp_i.transpose() *
              (jl_i * (H_ll_inv * (jl_j.transpose() * jp_j)));

          std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_j));
          accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
        }
      }

      // add blocks to b
      {
        std::scoped_lock lock(pose_mutex.at(cam_idx_i));
        //b.template segment<9>(cam_idx_i * 9) +=
        //          jp_i.transpose() * r_i;
        b.template segment<9>(cam_idx_i * 9) +=
            jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
      }
    }
  }

    inline void add_Hb_pOSE(BlockSparseMatrix<Scalar>& accu, VecX& b,
                       std::vector<std::mutex>& H_pp_mutex,
                       std::vector<std::mutex>& pose_mutex,
                       const size_t num_cam) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Mat3 H_ll;
        Mat3 H_ll_inv;
        Vec3 H_ll_inv_bl;
        {
            auto Jl = storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3);
            H_ll = Jl.transpose() * Jl;
            //H_ll.diagonal().array() += lambda_pOSE_;
            H_ll_inv_ = H_ll.inverse();
            H_ll_inv = H_ll.inverse();
            H_ll_inv_bl = H_ll_inv * (Jl.transpose() * storage_pOSE_.col(res_idx_pOSE_));
        }

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            const size_t cam_idx_i = pose_idx_[i];
            auto jp_i = storage_pOSE_.template block<4, POSE_SIZE>(4 * i, 0);
            auto jl_i = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            auto r_i = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);

            // TODO: check (Niko: not sure what we wanted to check here...)

            {
                MatX H_pp = jp_i.transpose() * jp_i;
                std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_i));
                accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));
            }

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                const size_t cam_idx_j = pose_idx_[j];
                auto jp_j = storage_pOSE_.template block<4, POSE_SIZE>(4 * j, 0);
                auto jl_j = storage_pOSE_.template block<4, 3>(4 * j, lm_idx_pOSE_);

                {
                    MatX H_pl_H_ll_inv_H_lp =
                            -jp_i.transpose() *
                            (jl_i * (H_ll_inv * (jl_j.transpose() * jp_j)));

                    std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_j));
                    accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
                }
            }

            // add blocks to b
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx_i));
                //b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                //        jp_i.transpose() * r_i;
                b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                        jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
            }
        }
    }

    inline void add_Hb_joint(BlockSparseMatrix<Scalar>& accu, VecX& b,
                            std::vector<std::mutex>& H_pp_mutex,
                            std::vector<std::mutex>& pose_mutex,
                            const size_t num_cam) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Mat3 H_ll;
        Mat3 H_ll_inv;
        Vec3 H_ll_inv_bl;
        {
            auto Jl = storage_nullspace_.block(0, lm_idx_nullspace_, num_rows_nullspace_, 3);
            H_ll = Jl.transpose() * Jl;
            H_ll.diagonal().array() += lambda_lm_landmark_;
            //H_ll_inv_ = H_ll.inverse();
            H_ll_inv = H_ll.inverse();
            H_ll_inv_bl = H_ll_inv * (Jl.transpose() * storage_nullspace_.col(res_idx_nullspace_));
        }

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            const size_t cam_idx_i = pose_idx_[i];

            auto jp_i = storage_nullspace_.template block<2, 11>(2 * i, 0);
            auto jl_i = storage_nullspace_.template block<2, 3>(2 * i, lm_idx_nullspace_);
            auto r_i = storage_nullspace_.template block<2, 1>(2 * i, res_idx_nullspace_);

            // TODO: check (Niko: not sure what we wanted to check here...)

            {
                MatX H_pp = jp_i.transpose() * jp_i;
                std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_i));
                accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));
            }

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                const size_t cam_idx_j = pose_idx_[j];
                auto jp_j = storage_nullspace_.template block<2, 11>(2 * j, 0);
                auto jl_j = storage_nullspace_.template block<2, 3>(2 * j, lm_idx_nullspace_);

                {
                    MatX H_pl_H_ll_inv_H_lp =
                            -jp_i.transpose() *
                            (jl_i * (H_ll_inv * (jl_j.transpose() * jp_j)));

                    std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_j));
                    accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
                }
            }

            // add blocks to b
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx_i));
                //b.template segment<11>(cam_idx_i * 11) +=
                //        jp_i.transpose() * r_i;
                b.template segment<11>(cam_idx_i * 11) +=
                        jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
            }
        }
    }

    inline void add_Hb_expOSE(BlockSparseMatrix<Scalar>& accu, VecX& b,
                            std::vector<std::mutex>& H_pp_mutex,
                            std::vector<std::mutex>& pose_mutex,
                            const size_t num_cam) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Mat3 H_ll;
        Mat3 H_ll_inv;
        Vec3 H_ll_inv_bl;
        {
            auto Jl = storage_expOSE_.block(0, lm_idx_expOSE_, num_rows_expOSE_, 3);
            H_ll = Jl.transpose() * Jl;
            //H_ll.diagonal().array() += lambda_expOSE_;
            H_ll_inv_ = H_ll.inverse();
            H_ll_inv = H_ll.inverse();

            H_ll_inv_bl = H_ll_inv * (Jl.transpose() * storage_expOSE_.col(res_idx_expOSE_));
        }

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            const size_t cam_idx_i = pose_idx_[i];

            auto jp_i = storage_expOSE_.template block<3, POSE_SIZE>(3 * i, 0);
            auto jl_i = storage_expOSE_.template block<3, 3>(3 * i, lm_idx_expOSE_);
            auto r_i = storage_expOSE_.template block<3, 1>(3 * i, res_idx_expOSE_);

            // TODO: check (Niko: not sure what we wanted to check here...)

            {
                MatX H_pp = jp_i.transpose() * jp_i;
                std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_i));
                accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));
            }

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                const size_t cam_idx_j = pose_idx_[j];
                auto jp_j = storage_expOSE_.template block<3, POSE_SIZE>(3 * j, 0);
                auto jl_j = storage_expOSE_.template block<3, 3>(3 * j, lm_idx_expOSE_);

                {
                    MatX H_pl_H_ll_inv_H_lp =
                            -jp_i.transpose() *
                            (jl_i * (H_ll_inv * (jl_j.transpose() * jp_j)));

                    std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_j));
                    accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
                }
            }

            // add blocks to b
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx_i));
                //b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                //        jp_i.transpose() * (r_i);
                b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                        jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
            }
        }
    }

    // Fill the explicit reduced H, b linear system by parallel_for and mutex
    inline void add_Hb_varproj(BlockSparseMatrix<Scalar>& accu, VecX& b,
                       std::vector<std::mutex>& H_pp_mutex,
                       std::vector<std::mutex>& pose_mutex,
                       const size_t num_cam) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Mat3 H_ll;
        Mat3 H_ll_inv;
        Vec3 H_ll_inv_bl;
        {
            auto Jl = storage_.block(0, lm_idx_, num_rows_, 3);
            H_ll = Jl.transpose() * Jl;
            //H_ll.diagonal().array() += lambda_;
            //H_ll_inv_ = H_ll.inverse();
            H_ll_inv = H_ll.inverse();
            H_ll_inv_bl = H_ll_inv * (Jl.transpose() * storage_.col(res_idx_));
        }

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            const size_t cam_idx_i = pose_idx_[i];

            auto jp_i = storage_.template block<2, POSE_SIZE>(2 * i, 0);
            auto jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);

            // TODO: check (Niko: not sure what we wanted to check here...)

            {
                MatX H_pp = jp_i.transpose() * jp_i;
                std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_i));
                accu.add(cam_idx_i, cam_idx_i, std::move(H_pp));
            }

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                const size_t cam_idx_j = pose_idx_[j];
                auto jp_j = storage_.template block<2, POSE_SIZE>(2 * j, 0);
                auto jl_j = storage_.template block<2, 3>(2 * j, lm_idx_);

                {
                    MatX H_pl_H_ll_inv_H_lp =
                            -jp_i.transpose() *
                            (jl_i * (H_ll_inv * (jl_j.transpose() * jp_j)));

                    std::scoped_lock lock(H_pp_mutex.at(cam_idx_i * num_cam + cam_idx_j));
                    accu.add(cam_idx_i, cam_idx_j, std::move(H_pl_H_ll_inv_H_lp));
                }
            }

            // add blocks to b
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx_i));
                b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                        jp_i.transpose() * r_i;
                //b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                //        jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
            }
        }
    }

  inline auto get_Hll_inv() const {
    return H_ll_inv_; }


  // Fill the reduced factor grouped H, b linear system by parallel_for and
  // mutex
  inline void add_Hb(FactorSCBlock<Scalar, 9>& H, VecX& b,
                     std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // Compute landmark-landmark block plus inverse
    Vec3 H_ll_inv_bl;
    MatX jl_hll_jl_t;
      const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
      H_ll_inv_ = jl.transpose() * jl;
      //Mat3 H_ll_inv = jl.transpose() * jl;
      H_ll_inv_.diagonal().array() += lambda_;
      H_ll_inv_ = H_ll_inv_.inverse().eval();
      jl_hll_jl_t = jl * H_ll_inv_ * jl.transpose();

      H_ll_inv_bl = H_ll_inv_ * (jl.transpose() * storage_.col(res_idx_));

    // Add pose-pose blocks and
    for (size_t i = 0; i < pose_idx_.size(); i++) {
      const size_t cam_idx_i = pose_idx_[i];

      const auto jp_i = storage_.template block<2, 9>(2 * i, 0);
      const auto jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
      const auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);

      // add blocks to b
      {
        const VecX tmp = jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
        std::scoped_lock lock(pose_mutex.at(cam_idx_i));
        b.template segment<POSE_SIZE>(cam_idx_i * 9) += tmp;
      }

      // Schur complement blocks
      for (size_t j = 0; j < pose_idx_.size(); j++) {
        const size_t cam_idx_j = pose_idx_[j];
        const auto jp_j = storage_.template block<2, 9>(2 * j, 0);

        const auto tmp = jl_hll_jl_t.template block<2, 2>(2 * i, 2 * j);
        MatX H_pl_H_ll_inv_H_lp;
        if (i == j) {
          H_pl_H_ll_inv_H_lp =
              jp_i.transpose() * jp_i - jp_i.transpose() * tmp * jp_j;
        } else {
          H_pl_H_ll_inv_H_lp = -jp_i.transpose() * tmp * jp_j;
        }
        H.add(cam_idx_i, cam_idx_j, H_pl_H_ll_inv_H_lp);
      }
    }
  }

    inline void add_Hb_pOSE(FactorSCBlock<Scalar, POSE_SIZE>& H, VecX& b,
                       std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Vec3 H_ll_inv_bl;
        MatX jl_hll_jl_t;
        const auto jl = storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3);
        H_ll_inv_ = jl.transpose() * jl;
        //Mat3 H_ll_inv = jl.transpose() * jl;
        //H_ll_inv_.diagonal().array() += lambda_pOSE_;
        H_ll_inv_ = H_ll_inv_.inverse().eval();
        jl_hll_jl_t = jl * H_ll_inv_ * jl.transpose();

        H_ll_inv_bl = H_ll_inv_ * (jl.transpose() * storage_pOSE_.col(res_idx_pOSE_));

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            const size_t cam_idx_i = pose_idx_[i];

            const auto jp_i = storage_pOSE_.template block<4, POSE_SIZE>(4 * i, 0);
            const auto jl_i = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            const auto r_i = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);

            // add blocks to b
            {
                const VecX tmp = jp_i.transpose() * r_i;
                //const VecX tmp = jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
                std::scoped_lock lock(pose_mutex.at(cam_idx_i));
                b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) += tmp;
            }

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                const size_t cam_idx_j = pose_idx_[j];
                const auto jp_j = storage_pOSE_.template block<4, POSE_SIZE>(4 * j, 0);

                const auto tmp = jl_hll_jl_t.template block<4, 4>(4 * i, 4 * j);
                MatX H_pl_H_ll_inv_H_lp;
                if (i == j) {
                    H_pl_H_ll_inv_H_lp =
                            jp_i.transpose() * jp_i - jp_i.transpose() * tmp * jp_j;
                } else {
                    H_pl_H_ll_inv_H_lp = -jp_i.transpose() * tmp * jp_j;
                }
                H.add(cam_idx_i, cam_idx_j, H_pl_H_ll_inv_H_lp);
            }
        }
    }

    inline void add_Hb_expOSE(FactorSCBlock<Scalar, POSE_SIZE>& H, VecX& b,
                            std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Vec3 H_ll_inv_bl;
        MatX jl_hll_jl_t;
        const auto jl = storage_expOSE_.block(0, lm_idx_expOSE_, num_rows_expOSE_, 3);
        H_ll_inv_ = jl.transpose() * jl;
        //Mat3 H_ll_inv = jl.transpose() * jl;
        H_ll_inv_.diagonal().array() += lambda_expOSE_;
        H_ll_inv_ = H_ll_inv_.inverse().eval();
        jl_hll_jl_t = jl * H_ll_inv_ * jl.transpose();

        H_ll_inv_bl = H_ll_inv_ * (jl.transpose() * storage_expOSE_.col(res_idx_expOSE_));

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            const size_t cam_idx_i = pose_idx_[i];

            const auto jp_i = storage_expOSE_.template block<3, POSE_SIZE>(3 * i, 0);
            const auto jl_i = storage_expOSE_.template block<3, 3>(3 * i, lm_idx_expOSE_);
            const auto r_i = storage_expOSE_.template block<3, 1>(3 * i, res_idx_expOSE_);

            // add blocks to b
            {
                const VecX tmp = jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
                std::scoped_lock lock(pose_mutex.at(cam_idx_i));
                b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) += tmp;
            }

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                const size_t cam_idx_j = pose_idx_[j];
                const auto jp_j = storage_expOSE_.template block<3, POSE_SIZE>(3 * j, 0);

                const auto tmp = jl_hll_jl_t.template block<3, 3>(3 * i, 3 * j);
                MatX H_pl_H_ll_inv_H_lp;
                if (i == j) {
                    H_pl_H_ll_inv_H_lp =
                            jp_i.transpose() * jp_i - jp_i.transpose() * tmp * jp_j;
                } else {
                    H_pl_H_ll_inv_H_lp = -jp_i.transpose() * tmp * jp_j;
                }
                H.add(cam_idx_i, cam_idx_j, H_pl_H_ll_inv_H_lp);
            }
        }
    }

    // Fill the reduced factor grouped H, b linear system by parallel_for and
    // mutex
    inline void add_Hb_varproj(FactorSCBlock<Scalar, POSE_SIZE>& H, VecX& b,
                       std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // Compute landmark-landmark block plus inverse
        Vec3 H_ll_inv_bl;
        MatX jl_hll_jl_t;
        const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
        H_ll_inv_ = jl.transpose() * jl;
        //Mat3 H_ll_inv = jl.transpose() * jl;
        //H_ll_inv_.diagonal().array() += lambda_; @Simon: we set lambda_landmark = 0 in varproj algorithm
        H_ll_inv_ = H_ll_inv_.inverse().eval();
        jl_hll_jl_t = jl * H_ll_inv_ * jl.transpose();

        H_ll_inv_bl = H_ll_inv_ * (jl.transpose() * storage_.col(res_idx_));

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            const size_t cam_idx_i = pose_idx_[i];

            const auto jp_i = storage_.template block<2, POSE_SIZE>(2 * i, 0);
            const auto jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            const auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);

            // add blocks to b
            {
                //const VecX tmp = jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl); //@Simon: no right side in varproj algorithm
                const VecX tmp = jp_i.transpose() * r_i;
                std::scoped_lock lock(pose_mutex.at(cam_idx_i));
                b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) += tmp;
            }

            // Schur complement blocks
            for (size_t j = 0; j < pose_idx_.size(); j++) {
                const size_t cam_idx_j = pose_idx_[j];
                const auto jp_j = storage_.template block<2, POSE_SIZE>(2 * j, 0);

                const auto tmp = jl_hll_jl_t.template block<2, 2>(2 * i, 2 * j);
                MatX H_pl_H_ll_inv_H_lp;
                if (i == j) {
                    H_pl_H_ll_inv_H_lp =
                            jp_i.transpose() * jp_i - jp_i.transpose() * tmp * jp_j;
                } else {
                    H_pl_H_ll_inv_H_lp = -jp_i.transpose() * tmp * jp_j;
                }
                H.add(cam_idx_i, cam_idx_j, H_pl_H_ll_inv_H_lp);
            }
        }
    }

  inline void add_Hpp(RowMatX& Hpp, std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t cam_idx = pose_idx_[i];
      const auto Jp = storage_.template block<2, POSE_SIZE>(2 * i, 0);
      const RowMatX tmp = Jp.transpose() * Jp;
      {
        std::scoped_lock lock(pose_mutex.at(cam_idx));
        Hpp.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * cam_idx, 0) += tmp;
      }
    }
  }

  inline void add_Hpp(BlockDiagonalAccumulator<Scalar>& Hpp,
                      std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t cam_idx = pose_idx_[i];
      const auto Jp = storage_.template block<2, POSE_SIZE>(2 * i, 0);
      const MatX tmp = Jp.transpose() * Jp;
      {
        std::scoped_lock lock(pose_mutex[cam_idx]);
        Hpp.add(cam_idx, tmp);
      }
    }
  }

  //template <typename Derived>
  inline void get_lm_landmark(VecX& delta_lm_update, size_t& i) const {
      ROOTBA_ASSERT(state_ == LINEARIZED);
      const auto jl = storage_.block(0, lm_idx_, num_rows_, 4);
      const auto r = storage_.block(0, res_idx_homogeneous_, num_rows_, 1);
      Mat4 Hll_inv = jl.transpose() * jl;
      Hll_inv.diagonal().array() += lambda_lm_landmark_;
      //Mat4 I = Mat4::Identity();
      Hll_inv = Hll_inv.template selfadjointView<Eigen::Upper>().llt().solve(Mat4::Identity());
      //Hll_inv = Hll_inv.householderQr().solve(I).eval();
      //Hll_inv = Hll_inv.ldlt().solve(I).eval();
      //Hll_inv = Hll_inv.inverse().eval();

      VecX b_r = jl.transpose() * r;
      delta_lm_update.template segment<4>(i * 4) = - Hll_inv * b_r;
      lm_ptr_->p_w_homogeneous += delta_lm_update.template segment<4>(i * 4);
      //lm_ptr_->p_w_homogeneous += Hll_inv * b_r;

  }

    inline void get_lm_landmark_riemannian_manifold(VecX& delta_lm_update, size_t& i) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        const auto jl = storage_.block(0, lm_idx_, num_rows_, 4);
        const auto r = storage_.block(0, res_idx_homogeneous_, num_rows_, 1);

        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

        auto jl_proj = jl * Proj;


        Mat3 Hll_inv = jl_proj.transpose() * jl_proj;

        //@Simon: try by adding Proj' * lambda * Proj:
        Hll_inv += Proj.transpose() * lambda_lm_landmark_ * Proj;
        //Hll_inv.diagonal().array() += lambda_lm_landmark_;

        Hll_inv = Hll_inv.inverse().eval();

        VecX b_r = jl_proj.transpose() * r;
        VecX dw = - Hll_inv * b_r;
        VecX inc_proj = Proj * dw;
        inc_proj.array() *= Jl_col_scale_homogeneous.array();
        lm_ptr_->p_w_homogeneous += inc_proj;
        //lm_ptr_->p_w_homogeneous.normalize();
        delta_lm_update.template segment<4>(i * 4) =  inc_proj;

    }


    // power_sc
  template <typename Derived>
  inline void get_Hll_inv_add_Hpp_b(RowMatX& jp_t_jp,
                                    Eigen::MatrixBase<Derived>& Hll_inv,
                                    VecX& b,
                                    std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
    const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
    Hll_inv = jl.transpose() * jl;
    Hll_inv.diagonal().array() += lambda_;
    Hll_inv = Hll_inv.inverse().eval();
    const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t cam_idx = pose_idx_[i];

      const auto jp = storage_.template block<2, 9>(2 * i, 0);
      const auto jl_pose = storage_.template block<2, 3>(2 * i, lm_idx_);
      const auto r = storage_.template block<2, 1>(2 * i, res_idx_);

      // fill b /// CHECK IF WE NEED TO KEEP IT
      const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
      //const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
      const RowMatX H_pp = jp.transpose() * jp;
      {
        std::scoped_lock lock(pose_mutex.at(cam_idx));
        b.template segment<9>(cam_idx * 9) +=
            tmp;  /// seems working without this line....
        jp_t_jp.template block<9, 9>(9 * cam_idx, 0) +=
            H_pp;
      }
    }
  }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_joint(RowMatX& jp_t_jp,
                                      Eigen::MatrixBase<Derived>& Hll_inv,
                                      VecX& b, const Cameras& cameras,
                                      std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        //const auto jl = storage_.block(0, lm_idx_, num_rows_, 4);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
        //auto jl_proj = jl * Proj;

        const auto jl_proj = storage_nullspace_.block(0,lm_idx_nullspace_,num_rows_nullspace_,3);

        Hll_inv = jl_proj.transpose() * jl_proj;
        Hll_inv += Proj.transpose() * lambda_lm_landmark_ * Proj;
        Hll_inv = Hll_inv.inverse().eval();
        const Vec3 hll_inv_bl = Hll_inv * (jl_proj.transpose() * storage_homogeneous_.col(res_idx_homogeneous_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto& cam = cameras.at(cam_idx);


            //const auto jp = storage_.template block<2, 12>(2 * i, 0);
            const auto jp_proj = storage_nullspace_.template block<2,11>(2*i,0);
            //const auto jl_pose = storage_.template block<2, 4>(2 * i, lm_idx_);
            const auto jl_pose = jl_proj.template block<2,3>(2*i,0);
            const auto r = storage_homogeneous_.template block<2, 1>(2 * i, res_idx_homogeneous_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            const VecX tmp = jp_proj.transpose() * (r - jl_pose * hll_inv_bl);
            //const VecX tmp = jp_proj.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp_proj.transpose() * jp_proj;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<11>(cam_idx * 11) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<11, 11>(11 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_joint_RpOSE(RowMatX& jp_t_jp,
                                            Eigen::MatrixBase<Derived>& Hll_inv,
                                            VecX& b, const Cameras& cameras,
                                            std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        //const auto jl = storage_.block(0, lm_idx_, num_rows_, 4);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
        //auto jl_proj = jl * Proj;

        const auto jl_proj = storage_nullspace_.block(0,lm_idx_nullspace_,num_rows_nullspace_,3);

        Hll_inv = jl_proj.transpose() * jl_proj;
        Hll_inv += Proj.transpose() * lambda_lm_landmark_ * Proj;
        Hll_inv = Hll_inv.inverse().eval();
        const Vec3 hll_inv_bl = Hll_inv * (jl_proj.transpose() * storage_homogeneous_.col(res_idx_homogeneous_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto& cam = cameras.at(cam_idx);


            //const auto jp = storage_.template block<2, 12>(2 * i, 0);
            const auto jp_full = get_Jpi_projective_space_riemannian_manifold_storage_RpOSE(i, cameras);
            //const auto jp_proj = storage_nullspace_.template block<2,11>(2*i,0);
            //const auto jp_intrinsics = storage_homogeneous_.template block<2,2>(2*i,13);
            const auto jl_pose = jl_proj.template block<2,3>(2*i,0);
            const auto r = storage_homogeneous_.template block<2, 1>(2 * i, res_idx_homogeneous_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp_proj.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp_full.transpose() * (r - jl_pose * hll_inv_bl);
            //const VecX tmp = jp_proj.transpose() * r; //@Simon: for VarProj
            //const RowMatX H_pp = jp_proj.transpose() * jp_proj;
            const RowMatX H_pp = jp_full.transpose() * jp_full;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<13>(cam_idx * 13) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<13, 13>(13 * cam_idx, 0) +=
                        H_pp;

            }
        }
    }



    // power_varpro
    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_affine_space(RowMatX& jp_t_jp,
                                      Eigen::MatrixBase<Derived>& Hll_inv,
                                      VecX& b,
                                      std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_affine_.block(0, lm_idx_affine_, num_rows_, 3);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        //Hll_inv.diagonal().array() += lambda_;
        //Mat3 I = Mat3::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I).eval();
        //Hll_inv = Hll_inv.ldlt().solve(I).eval();
        Hll_inv = Hll_inv.inverse().eval();

        //const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];

            const auto jp = storage_affine_.template block<2, 11>(2 * i, 0);
            const auto jl_pose = storage_affine_.template block<2, 3>(2 * i, lm_idx_affine_);
            const auto r = storage_affine_.template block<2, 1>(2 * i, res_idx_affine_);


            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            //@Simon: TODO: MISTAKE IN TMP (cout NaN)
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<11>(cam_idx * 11) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<11, 11>(11 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }


    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_pOSE(RowMatX& jp_t_jp,
                                                   Eigen::MatrixBase<Derived>& Hll_inv,
                                                   VecX& b,
                                                   std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        //Hll_inv.diagonal().array() += lambda_;
        Hll_inv = Hll_inv.inverse().eval();
        //Mat3 I = Mat3::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I).eval();
        //Hll_inv = Hll_inv.ldlt().solve(I).eval();

        const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_pOSE_.col(res_idx_pOSE_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_pOSE_.template block<4, 15>(4 * i, 0);
            const auto jl_pose = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            const auto r = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            //const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<15>(cam_idx * 15) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<15, 15>(15 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_RpOSE(RowMatX& jp_t_jp,
                                           Eigen::MatrixBase<Derived>& Hll_inv,
                                           VecX& b,
                                           std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_RpOSE_.block(0, lm_idx_RpOSE_, num_rows_RpOSE_, 3);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        //Hll_inv.diagonal().array() += lambda_;

        //Hll_inv = Hll_inv.inverse().eval();
//@SImon: try
        Hll_inv = Hll_inv.completeOrthogonalDecomposition().pseudoInverse();

        //Mat3 I = Mat3::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I).eval();
        //Hll_inv = Hll_inv.ldlt().solve(I).eval();

        const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_RpOSE_.col(res_idx_RpOSE_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_RpOSE_.template block<3, 11>(3 * i, 0);
            const auto jl_pose = storage_RpOSE_.template block<3, 3>(3 * i, lm_idx_RpOSE_);
            const auto r = storage_RpOSE_.template block<3, 1>(3 * i, res_idx_RpOSE_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<11>(cam_idx * 11) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<11, 11>(11 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_RpOSE_ML(RowMatX& jp_t_jp,
                                            Eigen::MatrixBase<Derived>& Hll_inv,
                                            VecX& b,
                                            std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_RpOSE_ML_.block(0, lm_idx_RpOSE_ML_, num_rows_RpOSE_ML_, 3);

        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        //std::cout << "Hll.norm() = " << Hll_inv.norm() << "\n";
        //Hll_inv.diagonal().array() += lambda_;
        //@Simon: with exact inverse
        //Hll_inv = Hll_inv.inverse().eval();
        //@Simon: with pseudo-inverse
        Hll_inv = Hll_inv.completeOrthogonalDecomposition().pseudoInverse();
        //std::cout << "Hll_inv.norm() = " << Hll_inv.norm() << "\n";
        //Mat3 I = Mat3::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I).eval();
        //Hll_inv = Hll_inv.ldlt().solve(I).eval();

        const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_RpOSE_ML_.col(res_idx_RpOSE_ML_));
        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_RpOSE_ML_.template block<1, 11>(i, 0);
            const auto jl_pose = storage_RpOSE_ML_.template block<1, 3>(i, lm_idx_RpOSE_ML_);
            const auto r = storage_RpOSE_ML_.template block<1, 1>(i, res_idx_RpOSE_ML_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            //const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<11>(cam_idx * 11) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<11, 11>(11 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }



    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_pOSE_poBA(RowMatX& jp_t_jp,
                                           Eigen::MatrixBase<Derived>& Hll_inv,
                                           VecX& b,
                                           std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        Hll_inv.diagonal().array() += lambda_;
        Hll_inv = Hll_inv.inverse().eval();
        //Mat3 I = Mat3::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I).eval();
        //Hll_inv = Hll_inv.ldlt().solve(I).eval();

        const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_pOSE_.col(res_idx_pOSE_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_pOSE_.template block<4, 15>(4 * i, 0);
            const auto jl_pose = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            const auto r = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            //const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<15>(cam_idx * 15) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<15, 15>(15 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }


    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_expOSE(RowMatX& jp_t_jp,
                                           Eigen::MatrixBase<Derived>& Hll_inv,
                                           VecX& b,
                                           std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_expOSE_.block(0, lm_idx_expOSE_, num_rows_expOSE_, 3);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        //Hll_inv.diagonal().array() += lambda_;
        Hll_inv = Hll_inv.inverse().eval();
        //@SImon: try
        const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_expOSE_.col(res_idx_expOSE_));


        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_expOSE_.template block<3, 15>(3 * i, 0);
            const auto jl_pose = storage_expOSE_.template block<3, 3>(3 * i, lm_idx_expOSE_);
            const auto r = storage_expOSE_.template block<3, 1>(3 * i, res_idx_expOSE_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            //const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<15>(cam_idx * 15) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<15, 15>(15 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }


    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_pOSE_rOSE(RowMatX& jp_t_jp,
                                           Eigen::MatrixBase<Derived>& Hll_inv,
                                           VecX& b,
                                           std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        const auto jl = storage_pOSE_rOSE_.block(0, lm_idx_pOSE_rOSE_, num_rows_pOSE_rOSE_, 3);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        Hll_inv = Hll_inv.inverse().eval();

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_pOSE_rOSE_.template block<5, 15>(5 * i, 0);
            const auto jl_pose = storage_pOSE_rOSE_.template block<5, 3>(5 * i, lm_idx_pOSE_rOSE_);
            const auto r = storage_pOSE_rOSE_.template block<5, 1>(5 * i, res_idx_pOSE_rOSE_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<15>(cam_idx * 15) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<15, 15>(15 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_rOSE(RowMatX& jp_t_jp,
                                           Eigen::MatrixBase<Derived>& Hll_inv,
                                           VecX& b,
                                           std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_rOSE_.block(0, lm_idx_rOSE_, num_rows_rOSE_, 3);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        //Hll_inv.diagonal().array() += lambda_;
        Hll_inv = Hll_inv.inverse().eval();
        //Mat3 I = Mat3::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I).eval();
        //Hll_inv = Hll_inv.ldlt().solve(I).eval();

        //const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_rOSE_.template block<3, 15>(3 * i, 0);
            const auto jl_pose = storage_rOSE_.template block<3, 3>(3 * i, lm_idx_pOSE_);
            const auto r = storage_rOSE_.template block<3, 1>(3 * i, res_idx_pOSE_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<15>(cam_idx * 15) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<15, 15>(15 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_pOSE_riemannian_manifold(RowMatX& jp_t_jp,
                                           Eigen::MatrixBase<Derived>& Hll_inv,
                                           VecX& b,const Cameras& cameras,
                                           std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3);

        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        Hll_inv = Hll_inv.inverse().eval();

        //const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto& cam = cameras.at(cam_idx);
            const auto jp = storage_pOSE_.template block<4, 12>(4 * i, 0);
            const auto jl_pose = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            const auto r = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);
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
            auto jp_proj = jp * Proj_pose;
            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp_proj.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp_proj.transpose() * jp_proj;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<11>(cam_idx * 11) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<11, 11>(11 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_pOSE_homogeneous(RowMatX& jp_t_jp,
                                           Eigen::MatrixBase<Derived>& Hll_inv,
                                           VecX& b, const Cameras& cameras,
                                           std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_pOSE_homogeneous_.block(0, lm_idx_pOSE_homogeneous_, num_rows_pOSE_homogeneous_, 4);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
        auto jl_proj = jl * Proj;
        Hll_inv = jl_proj.transpose() * jl_proj; //@Simon: seems to work
        //Hll_inv.diagonal().array() += lambda_;
        Hll_inv = Hll_inv.inverse().eval();
        //Mat3 I = Mat3::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I).eval();
        //Hll_inv = Hll_inv.ldlt().solve(I).eval();

        //const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));
        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto& cam = cameras.at(cam_idx);

            const auto jp = storage_pOSE_homogeneous_.template block<4, 12>(4 * i, 0);
            const auto jl_pose = storage_pOSE_homogeneous_.template block<4, 4>(4 * i, lm_idx_pOSE_homogeneous_);
            const auto r = storage_pOSE_homogeneous_.template block<4, 1>(4 * i, res_idx_pOSE_homogeneous_);

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
            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
            auto jp_proj = jp * Proj_pose;


            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp_proj.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp_proj.transpose() * jp_proj;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<11>(cam_idx * 11) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<11, 11>(11 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }


    // power_varpro
    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_projective_space(RowMatX& jp_t_jp,
                                                   Eigen::MatrixBase<Derived>& Hll_inv,
                                                   VecX& b,
                                                   std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        //Hll_inv.diagonal().array() += lambda_;
        //Hll_inv = Hll_inv.template selfadjointView<Eigen::Upper>().llt().solve(Mat4::Identity());
        Hll_inv = Hll_inv.inverse().eval();
        //const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];

            const auto jp = storage_.template block<2, 15>(2 * i, 0);
            const auto jl_pose = storage_.template block<2, 3>(2 * i, lm_idx_);
            const auto r = storage_.template block<2, 1>(2 * i, res_idx_);


            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp.transpose() * r; //@Simon: for VarProj

            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<15>(cam_idx * 15) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<15, 15>(15 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_projective_space_homogeneous(RowMatX& jp_t_jp,
                                                       Eigen::MatrixBase<Derived>& Hll_inv,
                                                       VecX& b,
                                                       std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_.block(0, lm_idx_, num_rows_, 4);
        Hll_inv = jl.transpose() * jl; //@Simon: seems to work
        //Hll_inv.diagonal().array() += lambda_;
        //Mat4 B = Mat4::Identity();
        //Hll_inv = Hll_inv.solve(B).eval();
        //@Simon: try:
        //Mat4 I = Mat4::Identity();
        Hll_inv = Hll_inv.template selfadjointView<Eigen::Upper>().llt().solve(Mat4::Identity());
        //Hll_inv = Hll_inv.householderQr().solve(I).eval();
        //const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];

            const auto jp = storage_.template block<2, 15>(2 * i, 0);
            const auto jl_pose = storage_.template block<2, 4>(2 * i, lm_idx_);
            const auto r = storage_.template block<2, 1>(2 * i, res_idx_homogeneous_);


            // fill b /// CHECK IF WE NEED TO KEEP IT
            //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const VecX tmp = jp.transpose() * r; //@Simon: for VarProj

            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<15>(cam_idx * 15) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<15, 15>(15 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_projective_space_homogeneous_riemannian_manifold(RowMatX& jp_t_jp,
                                                                   Eigen::MatrixBase<Derived>& Hll_inv,
                                                                   VecX& b, const Cameras& cameras,
                                                                   std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_.block(0, lm_idx_, num_rows_, 4);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

        auto jl_proj = jl * Proj;

        Hll_inv = jl_proj.transpose() * jl_proj;


        Hll_inv = Hll_inv.inverse().eval();


        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];

            const auto& cam = cameras.at(cam_idx);

            //const auto jp = storage_.template block<2, 12>(2 * i, 0);
            //const auto jl_pose = storage_.template block<2, 4>(2 * i, lm_idx_);
            //const auto r = storage_.template block<2, 1>(2 * i, res_idx_homogeneous_);

            const auto jp = storage_.template block<2, 12>(2 * i, 0);
            const auto jl_pose = storage_.template block<2, 4>(2 * i, lm_idx_);
            const auto r = storage_.template block<2, 1>(2 * i, res_idx_homogeneous_);
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

            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
            auto jp_proj = jp * Proj_pose;

            const VecX tmp = jp_proj.transpose() * r; //@Simon: for VarProj

            const RowMatX H_pp = jp_proj.transpose() * jp_proj;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<11>(cam_idx * 11) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<11, 11>(11 * cam_idx, 0) +=
                        H_pp;
            }
        }
    }



    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_refine(RowMatX& jp_t_jp,
                                      Eigen::MatrixBase<Derived>& Hll_inv,
                                      VecX& b,
                                      std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
        const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
        Hll_inv = jl.transpose() * jl;
        Hll_inv.diagonal().array() += lambda_;
        Hll_inv = Hll_inv.inverse().eval();
        const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];

            const auto jp = storage_.template block<2, POSE_SIZE>(2 * i, 0);
            const auto jl_pose = storage_.template block<2, 3>(2 * i, lm_idx_);
            const auto r = storage_.template block<2, 1>(2 * i, res_idx_);

            // fill b /// CHECK IF WE NEED TO KEEP IT
            const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            //const VecX tmp = jp.transpose() * r; //@Simon: for VarProj
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<POSE_SIZE>(cam_idx * POSE_SIZE) +=
                        tmp;  /// seems working without this line....
                jp_t_jp.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * cam_idx, 0) +=
                        H_pp;
            }
        }
    }

  // implicit power schur preconditioner
  template <typename Derived>
  inline void get_Hll_inv_add_Hpp_b_prec(
      RowMatX& jp_t_jp, Eigen::MatrixBase<Derived>& Hll_inv, VecX& b,
      std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
    const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
    Hll_inv = jl.transpose() * jl;
    Hll_inv.diagonal().array() += lambda_;
    Hll_inv = Hll_inv.inverse().eval();
    const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t cam_idx = pose_idx_[i];

      const auto jp = storage_.template block<2, POSE_SIZE>(2 * i, 0);
      const auto jl_pose = storage_.template block<2, 3>(2 * i, lm_idx_);
      const auto r = storage_.template block<2, 1>(2 * i, res_idx_);

      // fill b /// CHECK IF WE NEED TO KEEP IT
      //const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
      const RowMatX H_pp = jp.transpose() * jp;
      {
        std::scoped_lock lock(pose_mutex.at(cam_idx));
        //b.template segment<POSE_SIZE>(cam_idx * POSE_SIZE) += tmp;
         //b.template segment<POSE_SIZE>(cam_idx * POSE_SIZE) += tmp; /// seems
        // working without this line....
        jp_t_jp.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * cam_idx, 0) +=
            H_pp;

      }
    }
  }

  // hybrid_sc
  template <typename Derived>
  inline void get_Hll_inv_add_b_schur_jacobi(
      Eigen::MatrixBase<Derived>& Hll_inv, VecX& b,
      BlockDiagonalAccumulator<Scalar>& schur_jacobi,
      std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
    const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
    Hll_inv = jl.transpose() * jl;
    Hll_inv.diagonal().array() += lambda_;
    Hll_inv = Hll_inv.inverse().eval();

    const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t cam_idx = pose_idx_[i];

      const auto jp_i = storage_.template block<2, POSE_SIZE>(2 * i, 0);
      const auto jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
      const auto r = storage_.template block<2, 1>(2 * i, res_idx_);

      const VecX b_tmp = jp_i.transpose() * (r - jl_i * hll_inv_bl);

      const auto jp_t_jl = jp_i.transpose() * jl_i;
      MatX schur_jacobi_tmp =
          jp_i.transpose() * jp_i - jp_t_jl * (Hll_inv * jp_t_jl.transpose());
      {
        std::scoped_lock lock(pose_mutex.at(cam_idx));
        schur_jacobi.add(cam_idx, std::move(schur_jacobi_tmp));
        b.template segment<POSE_SIZE>(cam_idx * POSE_SIZE) += b_tmp;
      }
    }
  }

  // implicit_sc
  template <typename Derived>
  inline void get_Hll_inv_add_b(Eigen::MatrixBase<Derived>& Hll_inv, VecX& b,
                                std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
    const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
    Hll_inv = jl.transpose() * jl;
    Hll_inv.diagonal().array() += lambda_;
    Hll_inv = Hll_inv.inverse().eval();
    const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t cam_idx = pose_idx_[i];

      const auto jp = storage_.template block<2, POSE_SIZE>(2 * i, 0);
      const auto jl_pose = storage_.template block<2, 3>(2 * i, lm_idx_);
      const auto r = storage_.template block<2, 1>(2 * i, res_idx_);

      // fill b
      const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
      {
        std::scoped_lock lock(pose_mutex.at(cam_idx));
        b.template segment<POSE_SIZE>(cam_idx * POSE_SIZE) += tmp;
      }
    }
  }

  template <typename Derived>
  inline void get_Hll_inv_add_b_ps(RowMatX& jp_t_jp, Eigen::MatrixBase<Derived>& Hll_inv, VecX& b,
                                std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    // fill H_ll_inverse: (Jl'Jl + lm_lambda * I)^-1
    const auto jl = storage_.block(0, lm_idx_, num_rows_, 3);
    Hll_inv = jl.transpose() * jl;
    Hll_inv.diagonal().array() += lambda_;
    Hll_inv = Hll_inv.inverse().eval();
    const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_.col(res_idx_));

    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      const size_t cam_idx = pose_idx_[i];

      const auto jp = storage_.template block<2, POSE_SIZE>(2 * i, 0);
      const auto jl_pose = storage_.template block<2, 3>(2 * i, lm_idx_);
      const auto r = storage_.template block<2, 1>(2 * i, res_idx_);

      // fill b
      const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
      const RowMatX H_pp = jp.transpose() * jp;
      {
        std::scoped_lock lock(pose_mutex.at(cam_idx));
        b.template segment<POSE_SIZE>(cam_idx * POSE_SIZE) += tmp;
        jp_t_jp.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * cam_idx, 0) +=
            H_pp;
      }
    }
  }


  inline void add_jp_t_jp_blockdiag(
      BlockDiagonalAccumulator<Scalar>& accu) const {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      // using auto gives us a "reference" to the block
      auto jp = storage_.template block<2, POSE_SIZE>(2 * i, 0);
      accu.add(pose_idx_.at(i), jp.transpose() * jp);
    }
  }

  inline void add_jp_t_jp_blockdiag(BlockDiagonalAccumulator<Scalar>& accu,
                                    std::vector<std::mutex>& pose_mutex) const {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    for (size_t i = 0; i < pose_idx_.size(); i++) {
      // using auto gives us a "reference" to the block
      auto jp = storage_.template block<2, POSE_SIZE>(2 * i, 0);

      {
        MatX tmp = jp.transpose() * jp;
        std::scoped_lock lock(pose_mutex.at(pose_idx_.at(i)));
        accu.add(pose_idx_.at(i), std::move(tmp));
      }
    }
  }

  void back_substitute(const VecX& pose_inc, Scalar& l_diff) {
    ROOTBA_ASSERT(state_ == LINEARIZED);

    Mat3 H_ll = Mat3::Zero();
    Vec3 tmp = Vec3::Zero();
    VecX J_inc;
    J_inc.setZero(num_rows_);

    // Add pose-pose blocks and
    for (size_t i = 0; i < pose_idx_.size(); i++) {
      size_t cam_idx_i = pose_idx_[i];

      auto jp_i = storage_.template block<2, 9>(2 * i, 0);
      auto jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
      auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);

      H_ll += jl_i.transpose() * jl_i;

      auto p_inc = pose_inc.template segment<9>(cam_idx_i * 9);

      tmp += jl_i.transpose() * (r_i + jp_i * p_inc);
      J_inc.template segment<2>(2 * i) += jp_i * p_inc;
    }

    // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
    // need lambda in the interface
    H_ll.diagonal().array() += lambda_;
    Vec3 inc = -H_ll.inverse() * tmp;
    // Add landmark jacobian cost change
    J_inc += storage_.block(0, lm_idx_, num_rows_, 3) * inc;
    //@Simon: f(x+dx) = f(x) + g^T dx + 1/2 dx^T H dx
    l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_.col(res_idx_));

    // Note: scale only after computing model cost change
    inc.array() *= Jl_col_scale.array();
    lm_ptr_->p_w += inc;
  }

    void back_substitute_joint(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx_i = pose_idx_[i];

            const auto& cam = cameras.at(cam_idx_i);
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
            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
            ///

            auto jp_i = storage_homogeneous_.template block<2, 12>(2 * i, 0);
            auto jl_i = storage_homogeneous_.template block<2, 4>(2 * i, lm_idx_homogeneous_);
            auto r_i = storage_homogeneous_.template block<2, 1>(2 * i, res_idx_homogeneous_);
            auto jl_proj = jl_i * Proj;
            H_ll += jl_proj.transpose() * jl_proj;
            auto p_inc = pose_inc.template segment<11>(cam_idx_i * 11);
            tmp += jl_proj.transpose() * (r_i + jp_i * (Proj_pose * p_inc));
            J_inc.template segment<2>(2 * i) += jp_i * (Proj_pose * p_inc);
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        H_ll += Proj.transpose() * lambda_lm_landmark_ * Proj;
        Vec3 inc = -H_ll.inverse() * tmp;
        VecX inc_proj = Proj * inc;
        // Add landmark jacobian cost change
        J_inc += storage_homogeneous_.block(0, lm_idx_homogeneous_, num_rows_, 4) * inc_proj;
        //@Simon: f(x+dx) = f(x) + g^T dx + 1/2 dx^T H dx
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_homogeneous_.col(res_idx_homogeneous_));
        // Note: scale only after computing model cost change
        inc_proj.array() *= Jl_col_scale_homogeneous.array();
        //inc_proj.array() *= Jl_col_scale.array();
        lm_ptr_->p_w_homogeneous += inc_proj;
        //lm_ptr_->p_w_homogeneous.normalize();

    }

    void back_substitute_joint_RpOSE(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx_i = pose_idx_[i];

            const auto& cam = cameras.at(cam_idx_i);
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
            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
            ///

            auto jp_i = storage_homogeneous_.template block<2, 12>(2 * i, 0);
            auto jr_i = storage_homogeneous_.template block<2, 2>(2 * i, 13);
            auto jl_i = storage_homogeneous_.template block<2, 4>(2 * i, lm_idx_homogeneous_);
            auto r_i = storage_homogeneous_.template block<2, 1>(2 * i, res_idx_homogeneous_);
            auto jl_proj = jl_i * Proj;
            H_ll += jl_proj.transpose() * jl_proj;
            auto p_inc = pose_inc.template segment<11>(cam_idx_i * 13);
            auto p_inc_intrinsics = pose_inc.template segment<2>(cam_idx_i * 13 + 11);
            tmp += jl_proj.transpose() * (r_i + jp_i * (Proj_pose * p_inc) + jr_i * p_inc_intrinsics);
            J_inc.template segment<2>(2 * i) += jp_i * (Proj_pose * p_inc) + jr_i * p_inc_intrinsics;
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        H_ll += Proj.transpose() * lambda_lm_landmark_ * Proj;
        Vec3 inc = -H_ll.inverse() * tmp;
        VecX inc_proj = Proj * inc;
        // Add landmark jacobian cost change
        J_inc += storage_homogeneous_.block(0, lm_idx_homogeneous_, num_rows_, 4) * inc_proj;
        //@Simon: f(x+dx) = f(x) + g^T dx + 1/2 dx^T H dx
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_homogeneous_.col(res_idx_homogeneous_));
        // Note: scale only after computing model cost change
        inc_proj.array() *= Jl_col_scale_homogeneous.array();
        //inc_proj.array() *= Jl_col_scale.array();
        lm_ptr_->p_w_homogeneous += inc_proj;
        //lm_ptr_->p_w_homogeneous.normalize();

    }

    void back_substitute_poBA(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_pOSE_);

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx_i = pose_idx_[i];

            auto jp_i = storage_pOSE_.template block<4, 12>(4 * i, 0);
            auto jl_i = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            auto r_i = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);
            H_ll += jl_i.transpose() * jl_i;
            auto p_inc = pose_inc.template segment<12>(cam_idx_i * 15);
            tmp += jl_i.transpose() * (r_i + jp_i * (p_inc));
            J_inc.template segment<4>(4 * i) += jp_i * (p_inc);
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        H_ll.diagonal().array() += lambda_;
        Vec3 inc = -H_ll.inverse() * tmp;
        // Add landmark jacobian cost change
        J_inc += storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3) * inc;
        //@Simon: f(x+dx) = f(x) + g^T dx + 1/2 dx^T H dx
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_pOSE_.col(res_idx_pOSE_));
        // Note: scale only after computing model cost change
        inc.array() *= Jl_col_scale_pOSE.array();
        lm_ptr_->p_w += inc;

    }


    void landmark_closed_form(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_);

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);


            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian(
                    obs.pos, lm_ptr_->p_w, cam.T_c_w, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....


            auto jp_i = storage_.template block<2, POSE_SIZE>(2 * i, 0);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);
            auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);



            H_ll += jl_i.transpose() * jl_i;
//@Simon: initial version
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //tmp += jl_i.transpose() * (res + jp_i * p_inc);  //@Simon: to derive delta v in VarProj
///
/// @Simon: TRY 2 by using an updated Ju
            auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            auto p_inc_pose = pose_inc.template segment<6>(cam_idx * POSE_SIZE);
            auto p_inc_intr = pose_inc.template segment<3>(cam_idx * POSE_SIZE + 6);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
///

            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            J_inc.template segment<2>(2 * i) += jp_i * p_inc;
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        //H_ll.diagonal().array() += lambda_;
        //@Simon: why a -H_ll???
        //Vec3 inc = -H_ll.inverse() * tmp; //@Simon: is equal to -(Jv(uk + inc)^T Jv(uk+inc)).inverse() * Jv(uk+inc)^T * obs
        //Vec3 lm_update = H_ll.inverse() * tmp; //@Simon: to directly derive v* in VarProj
        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj

        // Add landmark jacobian cost change //@Simon: check and fix for varproj
        //lm_ptr_->p_w.array() *= Jl_col_scale.array().inverse();
        //lm_update.array() *= Jl_col_scale.array();
        //Vec3 inc = lm_update - lm_ptr_->p_w;// * Jl_col_scale.array().inverse();
        //inc.array() *= Jl_col_scale.array().inverse();

        J_inc += storage_.block(0, lm_idx_, num_rows_, 3) * inc;

        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_.col(res_idx_));

        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();

        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj
        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj
    }

    void landmark_closed_form_affine_space(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_);

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_affine_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);


            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_affine_space(
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....


            //auto jp_i = storage_.template block<2, 11>(2 * i, 0);
            auto jp_i = storage_affine_.template block<2, 11>(2 * i, 0);
            //auto ji = storage_affine_.template block<2,3>(2 * i, 11);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_affine_.template block<2, 1>(2 * i, res_idx_affine_);
            auto obs_i = storage_affine_.template block<2,1>(2 * i, obs_idx_affine_);



            H_ll += jl_i.transpose() * jl_i;
//@Simon: initial version
            auto p_inc = pose_inc.template segment<11>(cam_idx * 11);
            //tmp += jl_i.transpose() * (res + jp_i * p_inc);  //@Simon: to derive delta v in VarProj
///
/// @Simon: TRY 2 by using an updated Ju
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //auto p_inc_pose = pose_inc.template segment<8>(cam_idx * 11);
           // auto p_inc_intr = pose_inc.template segment<3>(cam_idx * 11 + 8);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
///

            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            J_inc.template segment<2>(2 * i) += jp_i * p_inc;
            //J_inc.template segment<2>(2 * i) += jp_i * p_inc;
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        //H_ll.diagonal().array() += lambda_;
        //@Simon: why a -H_ll???
        //Vec3 inc = -H_ll.inverse() * tmp; //@Simon: is equal to -(Jv(uk + inc)^T Jv(uk+inc)).inverse() * Jv(uk+inc)^T * obs
        //Vec3 lm_update = H_ll.inverse() * tmp; //@Simon: to directly derive v* in VarProj
        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj

        // Add landmark jacobian cost change //@Simon: check and fix for varproj
        //lm_ptr_->p_w.array() *= Jl_col_scale.array().inverse();
        //lm_update.array() *= Jl_col_scale.array();
        //Vec3 inc = lm_update - lm_ptr_->p_w;// * Jl_col_scale.array().inverse();
        //inc.array() *= Jl_col_scale.array().inverse();

        J_inc += storage_affine_.block(0, lm_idx_affine_, num_rows_, 3) * inc;

        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_affine_.col(res_idx_affine_));

        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();
        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj
        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj
    }

    void landmark_closed_form_pOSE(int alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_pOSE_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_pOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);

            Vec4 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE(alpha,
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....

            //auto jp_i = storage_.template block<2, 11>(2 * i, 0);
            auto jp_i = storage_pOSE_.template block<4, 12>(4 * i, 0);
            //auto ji = storage_affine_.template block<2,3>(2 * i, 11);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);
            auto obs_i = storage_pOSE_.template block<2,1>(4 * i + 2, obs_idx_pOSE_);


            H_ll += jl_i.transpose() * jl_i;
//@Simon: initial version
            auto p_inc = pose_inc.template segment<12>(cam_idx * 15);
            //tmp += jl_i.transpose() * (res + Jp * p_inc);  //@Simon: to derive delta v in VarProj
/// @Simon: TRY 2 by using an updated Ju
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //auto p_inc_pose = pose_inc.template segment<8>(cam_idx * 11);
            // auto p_inc_intr = pose_inc.template segment<3>(cam_idx * 11 + 8);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
///
            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            //J_inc.template segment<4>(4 * i) += jp_i * p_inc;
            J_inc.template segment<4>(4 * i) += Jp * p_inc;
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        //H_ll.diagonal().array() += lambda_;
        //@Simon: why a -H_ll???
        //Vec3 inc = -H_ll.inverse() * tmp; //@Simon: is equal to -(Jv(uk + inc)^T Jv(uk+inc)).inverse() * Jv(uk+inc)^T * obs
        //Vec3 lm_update = H_ll.inverse() * tmp; //@Simon: to directly derive v* in VarProj
        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj
        // Add landmark jacobian cost change //@Simon: check and fix for varproj
        //lm_ptr_->p_w.array() *= Jl_col_scale.array().inverse();
        //lm_update.array() *= Jl_col_scale.array();
        //Vec3 inc = lm_update - lm_ptr_->p_w;// * Jl_col_scale.array().inverse();


        J_inc += storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_pOSE_.col(res_idx_pOSE_));
        //inc.array() *= Jl_col_scale.array();
        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale_pOSE.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();
        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj
        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj
        //lm_ptr_->p_w.normalize();
        //lm_ptr_->p_w.normalize();
    }

    void landmark_closed_form_RpOSE(double alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_RpOSE_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_RpOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_RpOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_RpOSE jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);

            Vec3 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_RpOSE(alpha,
                                                                                                obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....

            //auto jp_i = storage_.template block<2, 11>(2 * i, 0);
            auto jp_i = storage_RpOSE_.template block<3, 8>(3 * i, 0);
            //auto ji = storage_affine_.template block<2,3>(2 * i, 11);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_RpOSE_.template block<3, 1>(3 * i, res_idx_RpOSE_);
            //auto obs_i = storage_RpOSE_.template block<2,1>(3 * i + 2, obs_idx_RpOSE_);


            H_ll += jl_i.transpose() * jl_i;
//@Simon: initial version
            auto p_inc = pose_inc.template segment<8>(cam_idx * 11);
            //tmp += jl_i.transpose() * (res + Jp * p_inc);  //@Simon: to derive delta v in VarProj
/// @Simon: TRY 2 by using an updated Ju
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //auto p_inc_pose = pose_inc.template segment<8>(cam_idx * 11);
            // auto p_inc_intr = pose_inc.template segment<3>(cam_idx * 11 + 8);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            //tmp += jl_i.transpose() * (res + jp_i * p_inc);
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
///
            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            //J_inc.template segment<4>(4 * i) += jp_i * p_inc;
            J_inc.template segment<3>(3 * i) += Jp * p_inc;
        }
        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        //H_ll.diagonal().array() += lambda_;
        //@Simon: why a -H_ll???
        //Vec3 inc = -H_ll.inverse() * tmp; //@Simon: is equal to -(Jv(uk + inc)^T Jv(uk+inc)).inverse() * Jv(uk+inc)^T * obs
        //Vec3 lm_update = H_ll.inverse() * tmp; //@Simon: to directly derive v* in VarProj
        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj
        // Add landmark jacobian cost change //@Simon: check and fix for varproj
        //lm_ptr_->p_w.array() *= Jl_col_scale.array().inverse();
        //lm_update.array() *= Jl_col_scale.array();
        //Vec3 inc = lm_update - lm_ptr_->p_w;// * Jl_col_scale.array().inverse();

        J_inc += storage_RpOSE_.block(0, lm_idx_RpOSE_, num_rows_RpOSE_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_RpOSE_.col(res_idx_RpOSE_));
        //inc.array() *= Jl_col_scale.array();
        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale_pOSE.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();
        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj
        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj
        //lm_ptr_->p_w.normalize();
        //lm_ptr_->p_w.normalize();
    }

    void landmark_closed_form_RpOSE_ML(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_RpOSE_ML_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_RpOSE_ML Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_RpOSE_ML Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_RpOSE_ML jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);

            Mat11 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_RpOSE_ML(obs.pos,obs.rpose_eq, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....

            //auto jp_i = storage_.template block<2, 11>(2 * i, 0);
            auto jp_i = storage_RpOSE_ML_.template block<1, 8>(i, 0);
            //auto ji = storage_affine_.template block<2,3>(2 * i, 11);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_RpOSE_ML_.template block<1, 1>(i, res_idx_RpOSE_ML_);
            //auto obs_i = storage_RpOSE_.template block<2,1>(3 * i + 2, obs_idx_RpOSE_);


            H_ll += jl_i.transpose() * jl_i;
//@Simon: initial version
            auto p_inc = pose_inc.template segment<8>(cam_idx * 11);
            //tmp += jl_i.transpose() * (res + Jp * p_inc);  //@Simon: to derive delta v in VarProj
/// @Simon: TRY 2 by using an updated Ju
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //auto p_inc_pose = pose_inc.template segment<8>(cam_idx * 11);
            // auto p_inc_intr = pose_inc.template segment<3>(cam_idx * 11 + 8);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            //tmp += jl_i.transpose() * (res + jp_i * p_inc);
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
///
            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            //J_inc.template segment<4>(4 * i) += jp_i * p_inc;
            J_inc.template segment<1>(i) += Jp * p_inc;
        }
        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        //H_ll.diagonal().array() += lambda_;
        //@Simon: why a -H_ll???
        //Vec3 inc = -H_ll.inverse() * tmp; //@Simon: is equal to -(Jv(uk + inc)^T Jv(uk+inc)).inverse() * Jv(uk+inc)^T * obs
        //Vec3 lm_update = H_ll.inverse() * tmp; //@Simon: to directly derive v* in VarProj

        //@Simon: with inverse
        //Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj
        //@Simon: with pseudo-inverse
        Vec3 inc = -H_ll.completeOrthogonalDecomposition().pseudoInverse() * tmp;

        // Add landmark jacobian cost change //@Simon: check and fix for varproj
        //lm_ptr_->p_w.array() *= Jl_col_scale.array().inverse();
        //lm_update.array() *= Jl_col_scale.array();
        //Vec3 inc = lm_update - lm_ptr_->p_w;// * Jl_col_scale.array().inverse();

        J_inc += storage_RpOSE_ML_.block(0, lm_idx_RpOSE_ML_, num_rows_RpOSE_ML_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_RpOSE_ML_.col(res_idx_RpOSE_ML_));
        //inc.array() *= Jl_col_scale.array();
        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale_pOSE.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();
        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj
        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj
        //lm_ptr_->p_w.normalize();
        //lm_ptr_->p_w.normalize();
    }

    void landmark_closed_form_RpOSE_refinement(double alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);
        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_RpOSE_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_RpOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_RpOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_RpOSE jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);

            Vec3 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_RpOSE_refinement(alpha,
                                                                                                 obs.pos, obs.rpose_eq, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....

            //auto jp_i = storage_.template block<2, 11>(2 * i, 0);
            auto jp_i = storage_RpOSE_.template block<3, 8>(3 * i, 0);
            //auto ji = storage_affine_.template block<2,3>(2 * i, 11);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_RpOSE_.template block<3, 1>(3 * i, res_idx_RpOSE_);
            //auto obs_i = storage_RpOSE_.template block<2,1>(3 * i + 2, obs_idx_RpOSE_);


            H_ll += jl_i.transpose() * jl_i;
//@Simon: initial version
            auto p_inc = pose_inc.template segment<8>(cam_idx * 11);
            //tmp += jl_i.transpose() * (res + Jp * p_inc);  //@Simon: to derive delta v in VarProj
/// @Simon: TRY 2 by using an updated Ju
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //auto p_inc_pose = pose_inc.template segment<8>(cam_idx * 11);
            // auto p_inc_intr = pose_inc.template segment<3>(cam_idx * 11 + 8);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
///
            //tmp += jl_i.transpose() * (res + jp_i * p_inc);
            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            //J_inc.template segment<4>(4 * i) += jp_i * p_inc;
            J_inc.template segment<3>(3 * i) += Jp * p_inc;
        }
        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        //H_ll.diagonal().array() += lambda_;
        //@Simon: why a -H_ll???
        //Vec3 inc = -H_ll.inverse() * tmp; //@Simon: is equal to -(Jv(uk + inc)^T Jv(uk+inc)).inverse() * Jv(uk+inc)^T * obs
        //Vec3 lm_update = H_ll.inverse() * tmp; //@Simon: to directly derive v* in VarProj
        //Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj
//@Simon: try
        Vec3 inc = -H_ll.completeOrthogonalDecomposition().pseudoInverse() * tmp;

        // Add landmark jacobian cost change //@Simon: check and fix for varproj
        //lm_ptr_->p_w.array() *= Jl_col_scale.array().inverse();
        //lm_update.array() *= Jl_col_scale.array();
        //Vec3 inc = lm_update - lm_ptr_->p_w;// * Jl_col_scale.array().inverse();

        J_inc += storage_RpOSE_.block(0, lm_idx_RpOSE_, num_rows_RpOSE_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_RpOSE_.col(res_idx_RpOSE_));
        //inc.array() *= Jl_col_scale.array();
        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale_pOSE.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();
        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj
        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj
        //lm_ptr_->p_w.normalize();
        //lm_ptr_->p_w.normalize();
    }



    inline void update_y_tilde_expose(const Cameras& cameras) {
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            auto& obs = lm_ptr_->obs.at(cam_idx);
            auto& cam = cameras.at(cam_idx);
            Vec3 y_tmp = cam.space_matrix * lm_ptr_->p_w.homogeneous();

            obs.y_tilde = y_tmp;
        }
  }

    inline void update_y_tilde_expose_initialize(const Cameras& cameras) {
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            auto& obs = lm_ptr_->obs.at(cam_idx);
            auto& cam = cameras.at(cam_idx);
            //Vec3 y_tmp = cam.space_matrix * lm_ptr_->p_w.homogeneous();
            Vec3 y_tmp;
            y_tmp(0) = obs.pos(0) /100.0;
            y_tmp(1) = obs.pos(1)/100.0;
            y_tmp(2) = 1;
            obs.y_tilde = y_tmp;
        }
    }

    inline void rpose_new_equilibrium(const Cameras& cameras) {
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            auto& obs = lm_ptr_->obs.at(cam_idx);
            auto& cam = cameras.at(cam_idx);
            //Vec3 y_tmp = cam.space_matrix * lm_ptr_->p_w.homogeneous();
            obs.rpose_eq[0] = cam.space_matrix.row(0) *  lm_ptr_->p_w.homogeneous();
            obs.rpose_eq[1] = cam.space_matrix.row(1) *  lm_ptr_->p_w.homogeneous();

            //@Simon: for debug:
            //obs.rpose_eq[0] = obs.pos[0];
            //obs.rpose_eq[1] = obs.pos[1];
        }
    }

    void landmark_closed_form_expOSE(int alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_expOSE_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_expOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_expOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_expOSE jl_i;

            Vec3 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_expOSE(alpha,
                                                                                                obs.pos, obs.y_tilde, lm_ptr_->p_w, lm_ptr_->p_w_backup(), cam.space_matrix, cam.space_matrix_backup(), cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            auto jp_i = storage_expOSE_.template block<3, 12>(3 * i, 0);
            auto r_i = storage_expOSE_.template block<3, 1>(3 * i, res_idx_expOSE_);


            H_ll += jl_i.transpose() * jl_i;

            auto p_inc = pose_inc.template segment<12>(cam_idx * 15);

            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7

            J_inc.template segment<3>(3 * i) += Jp * p_inc;
        }

        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj



        J_inc += storage_expOSE_.block(0, lm_idx_expOSE_, num_rows_expOSE_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_expOSE_.col(res_idx_expOSE_));
        //inc.array() *= Jl_col_scale.array();
        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale_pOSE.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();
        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj

        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj

    }

    void landmark_closed_form_pOSE_rOSE(int alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_pOSE_rOSE_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE_rOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_pOSE_rOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE_rOSE jl_i;


            Vec5 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE_rOSE(alpha,
                                                                                                obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)

            auto jp_i = storage_pOSE_rOSE_.template block<5, 12>(5 * i, 0);

            auto r_i = storage_pOSE_rOSE_.template block<5, 1>(5 * i, res_idx_pOSE_rOSE_);


            H_ll += jl_i.transpose() * jl_i;
            auto p_inc = pose_inc.template segment<12>(cam_idx * 15);

            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7

            J_inc.template segment<5>(5 * i) += Jp * p_inc;
        }

        Vec3 inc = - H_ll.inverse() * tmp;

        J_inc += storage_pOSE_rOSE_.block(0, lm_idx_pOSE_rOSE_, num_rows_pOSE_rOSE_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_pOSE_rOSE_.col(res_idx_pOSE_rOSE_));

        lm_ptr_->p_w += inc;

    }

    void landmark_closed_form_rOSE(int alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_rOSE_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_rOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_rOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_rOSE jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);

            Vec3 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_rOSE(alpha,
                                                                                                obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....

            //auto jp_i = storage_.template block<2, 11>(2 * i, 0);
            auto jp_i = storage_rOSE_.template block<3, 12>(3 * i, 0);
            //auto ji = storage_affine_.template block<2,3>(2 * i, 11);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_rOSE_.template block<3, 1>(3 * i, res_idx_rOSE_);
            //auto obs_i = storage_rOSE_.template block<1,1>(3 * i + 2, obs_idx_rOSE_);


            H_ll += jl_i.transpose() * jl_i;

            auto p_inc = pose_inc.template segment<12>(cam_idx * 15);

            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7

            J_inc.template segment<3>(3 * i) += Jp * p_inc;
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't

        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj


        J_inc += storage_rOSE_.block(0, lm_idx_rOSE_, num_rows_rOSE_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_rOSE_.col(res_idx_rOSE_));

        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj

    }


    void landmark_closed_form_pOSE_riemannian_manifold(int alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_pOSE_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_pOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);

            Vec4 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE(alpha,
                                                                                                obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....

            //auto jp_i = storage_.template block<2, 11>(2 * i, 0);
            auto jp_i = storage_pOSE_.template block<4, 12>(4 * i, 0);
            //auto ji = storage_affine_.template block<2,3>(2 * i, 11);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);
            auto obs_i = storage_pOSE_.template block<2,1>(4 * i + 2, obs_idx_pOSE_);


            H_ll += jl_i.transpose() * jl_i;
//@Simon: initial version
            auto p_inc = pose_inc.template segment<12>(cam_idx * 12);
            //tmp += jl_i.transpose() * (res + jp_i * p_inc);  //@Simon: to derive delta v in VarProj
/// @Simon: TRY 2 by using an updated Ju
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //auto p_inc_pose = pose_inc.template segment<8>(cam_idx * 11);
            // auto p_inc_intr = pose_inc.template segment<3>(cam_idx * 11 + 8);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
///
            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            //J_inc.template segment<4>(4 * i) += jp_i * p_inc;
            J_inc.template segment<4>(4 * i) += Jp * p_inc;
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        //H_ll.diagonal().array() += lambda_;
        //@Simon: why a -H_ll???
        //Vec3 inc = -H_ll.inverse() * tmp; //@Simon: is equal to -(Jv(uk + inc)^T Jv(uk+inc)).inverse() * Jv(uk+inc)^T * obs
        //Vec3 lm_update = H_ll.inverse() * tmp; //@Simon: to directly derive v* in VarProj
        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj
        // Add landmark jacobian cost change //@Simon: check and fix for varproj
        //lm_ptr_->p_w.array() *= Jl_col_scale.array().inverse();
        //lm_update.array() *= Jl_col_scale.array();
        //Vec3 inc = lm_update - lm_ptr_->p_w;// * Jl_col_scale.array().inverse();


        J_inc += storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_pOSE_.col(res_idx_pOSE_));
        //inc.array() *= Jl_col_scale.array();
        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale_pOSE.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();
        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj
        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj
        //lm_ptr_->p_w.normalize();
        //lm_ptr_->p_w.normalize();
    }

    void landmark_closed_form_pOSE_homogeneous(int alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_pOSE_homogeneous_);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            const auto& cam = cameras.at(cam_idx);
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
            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());

            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            //const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_pOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE_homogeneous jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);
            Vec4 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE_homogeneous(alpha,
                                                                                                obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....

            auto jp_i = storage_pOSE_homogeneous_.template block<4, 12>(4 * i, 0);
            auto r_i = storage_pOSE_homogeneous_.template block<4, 1>(4 * i, res_idx_pOSE_homogeneous_);
            auto obs_i = storage_pOSE_homogeneous_.template block<2,1>(4 * i + 2, obs_idx_pOSE_homogeneous_);

            auto jl_proj = jl_i * Proj;
            H_ll += jl_proj.transpose() * jl_proj;
//@Simon: initial version
            auto p_inc = pose_inc.template segment<11>(cam_idx * 11);
            //tmp += jl_i.transpose() * (res + jp_i * p_inc);  //@Simon: to derive delta v in VarProj
/// @Simon: TRY 2 by using an updated Ju
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //auto p_inc_pose = pose_inc.template segment<8>(cam_idx * 11);
            // auto p_inc_intr = pose_inc.template segment<3>(cam_idx * 11 + 8);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            tmp += jl_proj.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            J_inc.template segment<4>(4 * i) += (jp_i * Proj_pose) * p_inc;
            //J_inc.template segment<2>(2 * i) += jp_i * p_inc;
        }

        // TODO: store additionally "Hllinv" (inverted with lambda), so we don't
        // need lambda in the interface
        //H_ll.diagonal().array() += lambda_;
        //@Simon: why a -H_ll???
        //Vec3 inc = -H_ll.inverse() * tmp; //@Simon: is equal to -(Jv(uk + inc)^T Jv(uk+inc)).inverse() * Jv(uk+inc)^T * obs
        //Vec3 lm_update = H_ll.inverse() * tmp; //@Simon: to directly derive v* in VarProj
        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj
        // Add landmark jacobian cost change //@Simon: check and fix for varproj
        //lm_ptr_->p_w.array() *= Jl_col_scale.array().inverse();
        //lm_update.array() *= Jl_col_scale.array();
        //Vec3 inc = lm_update - lm_ptr_->p_w;// * Jl_col_scale.array().inverse();
        //std::cout << "in landmark_block    l2194 ok \n";

        J_inc += storage_pOSE_homogeneous_.block(0, lm_idx_pOSE_homogeneous_, num_rows_pOSE_homogeneous_, 4) * (Proj * inc);
        //std::cout << "in landmark_block    l2197 ok \n";
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_pOSE_homogeneous_.col(res_idx_pOSE_homogeneous_));
        //inc.array() *= Jl_col_scale.array();
        // Note: scale only after computing model cost change
        //inc.array() *= Jl_col_scale_pOSE.array();
        //@Simon: we directly get the new landmark (and not the increment)
        //lm_update.array() *= Jl_col_scale.array();
        VecX inc_lm = Proj * inc;
        //inc_lm.array() *=Jl_col_scale_pOSE_homogeneous.array();
        //lm_ptr_->p_w = lm_update; //@Simon: to directly derive v* in VarProj
        //std::cout << "in landmark_block    l2205 ok \n";
        lm_ptr_->p_w_homogeneous += inc_lm;
        //lm_ptr_->p_w_homogeneous += Proj * inc; //@Simon: to derive delta v in VarProj
        //lm_ptr_->p_w.normalize();
    }

    void landmark_closed_form_projective_space(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_);

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL jl_i;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);


            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_projective_space(
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            //@Simon: here, res is changed.... FIX IT.. in fact not so important as we don't use res.....



            auto jp_i = storage_.template block<2, 15>(2 * i, 0);
            //jl_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            //auto jL_i = storage_.template block<2, 3>(2 * i, lm_idx_);
            auto r_i = storage_.template block<2, 1>(2 * i, res_idx_);
            auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);

            H_ll += jl_i.transpose() * jl_i;
//@Simon: initial version
            auto p_inc = pose_inc.template segment<15>(cam_idx * 15);
            //tmp += jl_i.transpose() * (res + jp_i * p_inc);  //@Simon: to derive delta v in VarProj
///
/// @Simon: TRY 2 by using an updated Ju
            //auto p_inc = pose_inc.template segment<POSE_SIZE>(cam_idx * POSE_SIZE);
            //auto p_inc_pose = pose_inc.template segment<6>(cam_idx * POSE_SIZE);
            //auto p_inc_intr = pose_inc.template segment<3>(cam_idx * POSE_SIZE + 6);
            //tmp += jl_i.transpose() * (res + Jp * p_inc_pose + Ji * p_inc_intr); //@Simon: not in line with the supplementary of the paper, in line with equation page7
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
///

            // @Simon: in Revisiting VarPro, delta v = - J_v.pseudoinverse() * (res + J_u delta u)
            // id est:                      delta v = - (J_v^T J_v)^(-1) J_v^T (res + J_u delta u)
            //tmp += jl_i.transpose() * obs_i; //@Simon: to directly derive v* in VarProj
            J_inc.template segment<2>(2 * i) += jp_i * p_inc;
        }

        Vec3 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj

        J_inc += storage_.block(0, lm_idx_, num_rows_, 3) * inc;

        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_.col(res_idx_));

        lm_ptr_->p_w += inc; //@Simon: to derive delta v in VarProj
    }

    void landmark_closed_form_projective_space_homogeneous(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat4 H_ll = Mat4::Zero();
        Vec4 tmp = Vec4::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_);

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_projective_space_homogeneous jl_i;


            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_projective_space_homogeneous(
                    obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)


            auto jp_i = storage_.template block<2, 15>(2 * i, 0);
            auto r_i = storage_.template block<2, 1>(2 * i, res_idx_homogeneous_);
            auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_homogeneous_);

            H_ll += jl_i.transpose() * jl_i;
            auto p_inc = pose_inc.template segment<15>(cam_idx * 15);
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
            J_inc.template segment<2>(2 * i) += jp_i * p_inc;
        }

        //Mat4 I = Mat4::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I);
        Vec4 inc = - H_ll.template selfadjointView<Eigen::Upper>().llt().solve(Mat4::Identity()) * tmp;
        //Vec4 inc = - H_ll.householderQr().solve(I) * tmp;

        //Vec4 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj

        J_inc += storage_.block(0, lm_idx_, num_rows_, 4) * inc;

        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_.col(res_idx_homogeneous_));
        //lm_ptr_->p_w_homogeneous += inc;
    }

    void landmark_closed_form_projective_space_homogeneous_riemannian_manifold(const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            const auto& cam = cameras.at(cam_idx);
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
            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());

            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            //const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_projective_space_homogeneous jl_i;


            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_projective_space_homogeneous(
                    obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)


            auto jp_i = storage_.template block<2, 12>(2 * i, 0);
            auto r_i = storage_.template block<2, 1>(2 * i, res_idx_homogeneous_);
            auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_homogeneous_);

            auto jl_proj = jl_i * Proj;
            H_ll += jl_proj.transpose() * jl_proj;
            auto p_inc = pose_inc.template segment<11>(cam_idx * 11);
            tmp += jl_proj.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
            J_inc.template segment<2>(2 * i) += (jp_i*Proj_pose) * p_inc;
        }

        //Mat4 I = Mat4::Identity();
        //Hll_inv = Hll_inv.householderQr().solve(I);
        Vec3 inc = - H_ll.inverse().eval() * tmp;
        //Vec4 inc = - H_ll.template selfadjointView<Eigen::Upper>().llt().solve(Mat4::Identity()) * tmp;
        //Vec4 inc = - H_ll.householderQr().solve(I) * tmp;

        //Vec4 inc = - H_ll.inverse() * tmp; //@Simon: to derive delta v in VarProj
        J_inc += (storage_.block(0, lm_idx_, num_rows_, 4) * Proj) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_.col(res_idx_homogeneous_));
        //lm_ptr_->p_w_homogeneous += inc;
    }

    void landmark_closed_form_projective_space_homogeneous_nonlinear_initialization(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat4 H_ll = Mat4::Zero();
        Vec4 tmp = Vec4::Zero();
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_projective_space_homogeneous jl_i;

            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_projective_space_homogeneous(
                    obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)

            H_ll += jl_i.transpose() * jl_i;
            tmp += jl_i.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
        }

        Vec4 inc = - H_ll.template selfadjointView<Eigen::Upper>().llt().solve(Mat4::Identity()) * tmp;

        lm_ptr_->p_w_homogeneous += inc;
    }

    void landmark_closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold(const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();


        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

        //auto jl_proj = jl * Proj;
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            ///
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_projective_space_homogeneous jl_i;

            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_projective_space_homogeneous(
                    obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, &Jp, &Ji, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)
            auto jl_proj = jl_i * Proj;
            H_ll += jl_proj.transpose() * jl_proj;
            tmp += jl_proj.transpose() * res; //@Simon: in line with the supplementary of the paper, not with page7
        }

        Vec3 inc = - H_ll.inverse().eval() * tmp;

        lm_ptr_->p_w_homogeneous += Proj * inc;
    }


    void print_storage(const std::string& filename) const {
    std::ofstream f(filename);

    Eigen::IOFormat clean_fmt(4, 0, " ", "\n", "", "");

    f << "Storage (state: " << state_
      << " Jl_col_scale: " << Jl_col_scale.transpose() << "):\n"
      << storage_.format(clean_fmt) << std::endl;

    f.close();
  }

 protected:
  // Dense storage for pose Jacobians, padding, landmark Jacobians and
  // residuals [Jp (all jacobians in one column) | Jl | res] //@Simon: | obs]
  RowMatX storage_;
  RowMatX storage_homogeneous_;
  RowMatX storage_affine_;
  RowMatX storage_pOSE_;
  RowMatX storage_RpOSE_;
  RowMatX storage_RpOSE_ML_;
  RowMatX storage_expOSE_;
  RowMatX storage_pOSE_rOSE_;
  RowMatX storage_rOSE_;
  RowMatX storage_pOSE_homogeneous_;
  RowMatX storage_nullspace_;
  mutable Mat3 H_ll_inv_;
  //mutable Mat4 hll_inv_homogeneous_;

  Vec3 Jl_col_scale;
  Vec3 Jl_col_scale_pOSE;
    Vec3 Jl_col_scale_RpOSE;
    Vec3 Jl_col_scale_RpOSE_ML;
  Vec3 Jl_col_scale_expOSE;
  Vec4 Jl_col_scale_pOSE_homogeneous;
  Vec4 Jl_col_scale_homogeneous;
  Scalar lambda_ = 0;
  Scalar lambda_pOSE_ = 0;
    Scalar lambda_RpOSE_ = 0;
  Scalar lambda_expOSE_ = 0;

  Scalar lambda_lm_landmark_ = 0;

  std::vector<size_t> pose_idx_;
  size_t lm_idx_ = 0;
  size_t res_idx_ = 0;
  size_t obs_idx_ = 0;
  size_t lm_idx_homogeneous_ = 0;
  size_t res_idx_homogeneous_ = 0;
  size_t obs_idx_homogeneous_ = 0;
  size_t lm_idx_affine_ = 0;
  size_t res_idx_affine_ = 0;
  size_t obs_idx_affine_ = 0;

  size_t num_cols_ = 0;
  size_t num_rows_ = 0;
  size_t num_cols_homogeneous_ = 0;
  size_t num_cols_affine_ = 0;

  size_t num_rows_pOSE_ = 0;
  size_t num_cols_pOSE_ = 0;
  size_t lm_idx_pOSE_ = 0;
  size_t res_idx_pOSE_ = 0;
  size_t obs_idx_pOSE_ = 0;


  size_t num_rows_RpOSE_ = 0;
  size_t num_cols_RpOSE_ = 0;
  size_t lm_idx_RpOSE_ = 0;
  size_t res_idx_RpOSE_ = 0;
  size_t obs_idx_RpOSE_ = 0;

    size_t num_rows_RpOSE_ML_ = 0;
    size_t num_cols_RpOSE_ML_ = 0;
    size_t lm_idx_RpOSE_ML_ = 0;
    size_t res_idx_RpOSE_ML_ = 0;
    size_t obs_idx_RpOSE_ML_ = 0;

  size_t num_rows_expOSE_ = 0;
  size_t num_cols_expOSE_ = 0;
  size_t lm_idx_expOSE_ = 0;
  size_t res_idx_expOSE_ = 0;
  size_t obs_idx_expOSE_ = 0;


  size_t num_rows_pOSE_rOSE_ = 0;
  size_t num_cols_pOSE_rOSE_ = 0;
  size_t lm_idx_pOSE_rOSE_ = 0;
  size_t res_idx_pOSE_rOSE_ = 0;
  size_t obs_idx_pOSE_rOSE_ = 0;

  size_t num_rows_pOSE_homogeneous_ = 0;
  size_t num_cols_pOSE_homogeneous_ = 0;
  size_t lm_idx_pOSE_homogeneous_ = 0;
  size_t res_idx_pOSE_homogeneous_ = 0;
  size_t obs_idx_pOSE_homogeneous_ = 0;

  size_t num_rows_rOSE_ = 0;
  size_t num_cols_rOSE_ = 0;
  size_t lm_idx_rOSE_ = 0;
  size_t res_idx_rOSE_= 0;
  size_t obs_idx_rOSE_ = 0;

  size_t num_rows_nullspace_ = 0;
  size_t num_cols_nullspace_ = 0;
  size_t lm_idx_nullspace_ = 0;
  size_t res_idx_nullspace_ = 0;

  Options options_;

  State state_ = UNINITIALIZED;

  Landmark* lm_ptr_ = nullptr;
};

}  // namespace rootba
