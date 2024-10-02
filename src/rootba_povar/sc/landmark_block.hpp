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

#include "rootba_povar/bal/bal_bundle_adjustment_helper.hpp"
#include "rootba_povar/bal/bal_problem.hpp"
#include "rootba_povar/bal/bal_residual_options.hpp"
#include "rootba_povar/bal/solver_options.hpp"
#include "rootba_povar/cg/block_sparse_matrix.hpp"
#include "rootba_povar/util/assert.hpp"
#include "rootba_povar/util/format.hpp"

namespace rootba_povar {
template <typename, int>
class FactorSCBlock;

template <typename Scalar, int POSE_SIZE>
class LandmarkBlockSC {
 public:
  struct Options {
    // use_valid_projections_only: if true, set invalid projection's
    // residual and jacobian to 0; invalid means z <= 0
    bool use_valid_projections_only = true;

    // huber norm with given threshold, else squared norm
    BalResidualOptions residual_options;

    // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
    // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
    Scalar jacobi_scaling_eps = 1.0;

    // JACOBI or SCHUR_JACOBI
    SolverOptions::PreconditionerType preconditioner_type;
  };

  enum State { UNINITIALIZED = 0, ALLOCATED, NUMERICAL_FAILURE, LINEARIZED };

  inline bool is_numerical_failure() const {
    return state_ == NUMERICAL_FAILURE;
  }

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using Vec12 = Eigen::Matrix<Scalar, 12, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
  using Mat4 = Eigen::Matrix<Scalar,4,4>;
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

    lm_idx_homogeneous_ = 12;
    res_idx_homogeneous_ = lm_idx_homogeneous_ + 4;
    num_cols_homogeneous_ = res_idx_homogeneous_ + 1;

    lm_idx_nullspace_ = 11;
    num_cols_nullspace_ = lm_idx_nullspace_ + 3;
    num_rows_nullspace_ = pose_idx_.size() * 2;

    num_rows_pOSE_ = pose_idx_.size() * 4;
    lm_idx_pOSE_ = 12;
    res_idx_pOSE_ = lm_idx_pOSE_ + 3;
    num_cols_pOSE_ = res_idx_pOSE_ + 1;

    storage_.resize(num_rows_, num_cols_homogeneous_);
    storage_pOSE_.resize(num_rows_pOSE_, num_cols_pOSE_);
    storage_homogeneous_.resize(num_rows_pOSE_homogeneous_, num_cols_pOSE_homogeneous_);
    storage_nullspace_.resize(num_rows_nullspace_, num_cols_nullspace_);

    state_ = ALLOCATED;

    lm_ptr_ = &lm;
  }

    inline void linearize_landmark_pOSE(const Cameras& cameras, Scalar alpha) {
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

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE Jl;

            Vec4 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE(alpha,
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp,
                    &Jl);
            if (!options_.use_valid_projections_only || valid) {
                numerically_valid = numerically_valid && Jl.array().isFinite().all() &&
                                    Jp.array().isFinite().all() &&
                                    res.array().isFinite().all();

                const Scalar res_squared = res.squaredNorm();
                const auto [weighted_error, weight] =
                BalBundleAdjustmentHelper<Scalar>::compute_error_weight(
                        options_.residual_options, res_squared);
                const Scalar sqrt_weight = std::sqrt(weight);
                storage_pOSE_.template block<4, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_pOSE_.template block<4, 3>(obs_idx, lm_idx_pOSE_) = sqrt_weight * Jl;
                storage_pOSE_.template block<4, 1>(obs_idx, res_idx_pOSE_) = sqrt_weight * res;
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

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_projective_space Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_projective_space_homogeneous Jl;
            Vec2 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space_homogeneous(
                    obs.pos, lm_ptr_->p_w_homogeneous, cam.space_matrix, cam.intrinsics, true, res, &Jp,
                    &Jl);

            if (!options_.use_valid_projections_only || valid) {
                numerically_valid = numerically_valid && Jl.array().isFinite().all() &&
                                    Jp.array().isFinite().all() &&
                                    res.array().isFinite().all();

                const Scalar res_squared = res.squaredNorm();
                const auto [weighted_error, weight] =
                BalBundleAdjustmentHelper<Scalar>::compute_error_weight(
                        options_.residual_options, res_squared);
                const Scalar sqrt_weight = std::sqrt(weight);

                storage_homogeneous_.template block<2, 12>(obs_idx, pose_idx) = sqrt_weight * Jp;
                storage_homogeneous_.template block<2, 4>(obs_idx, lm_idx_homogeneous_) = sqrt_weight * Jl;
                storage_homogeneous_.template block<2, 1>(obs_idx, res_idx_homogeneous_) = sqrt_weight * res;
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


    inline void add_Jp_diag2_pOSE(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<12>(12 * cam_idx) +=
                    storage_pOSE_.template block<4, 12>(4 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
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

    inline void scale_Jp_cols_joint(const VecX& jacobian_scaling) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];

            storage_homogeneous_.template block<2, 12>(2 * i, 0) *=
                    jacobian_scaling.template segment<12>(12 * cam_idx)
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

  inline void set_landmark_damping(Scalar lambda) { lambda_ = lambda; }

  inline void set_landmark_damping_joint(Scalar lambda) { lambda_ = lambda; }

  inline size_t num_poses() const { return pose_idx_.size(); }

  inline const std::vector<size_t>& get_pose_idx() const { return pose_idx_; }

  inline auto get_Jl_homogeneous_riemannian_manifold_storage() const {
      return storage_nullspace_.template middleCols<3>(lm_idx_nullspace_);
  }

  inline auto get_Jl_pOSE() const {
      return storage_pOSE_.template middleCols<3>(lm_idx_pOSE_);
  }

  inline auto get_Jpi_pOSE(const size_t obs_idx) const {
      return storage_pOSE_.template block<4, 12>(4 * obs_idx, 0);
  }

  inline auto get_Jpi_projective_space_riemannian_manifold_storage(const size_t obs_idx, const Cameras& cameras) const {
      return storage_nullspace_.template block<2, 11>(2 * obs_idx, 0);
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
          H_ll_inv = H_ll.inverse();
          H_ll_inv_bl = H_ll_inv * (Jl.transpose() * storage_pOSE_.col(res_idx_pOSE_));
      }

      // Add pose-pose blocks and
      for (size_t i = 0; i < pose_idx_.size(); i++) {
          const size_t cam_idx_i = pose_idx_[i];
          auto jp_i = storage_pOSE_.template block<4, POSE_SIZE>(4 * i, 0);
          auto jl_i = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
          auto r_i = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);

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
              b.template segment<POSE_SIZE>(cam_idx_i * POSE_SIZE) +=
                      jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl); // in line with "Revisiting the Variable Projection Method for Separable Nonlinear Least Squares Problems" (Hong et al., CVPR 2017)
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
        auto Jl = storage_nullspace_.block(0, lm_idx_nullspace_, num_rows_nullspace_, 3);
        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
        {
            H_ll = Jl.transpose() * Jl;
            H_ll += Proj.transpose() * lambda_ * Proj;
            H_ll_inv = H_ll.inverse();
            H_ll_inv_bl = H_ll_inv * (Jl.transpose() * storage_homogeneous_.col(res_idx_homogeneous_));
        }

        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            const size_t cam_idx_i = pose_idx_[i];

            auto jp_i = storage_nullspace_.template block<2, 11>(2 * i, 0);
            auto jl_i = storage_nullspace_.template block<2, 3>(2 * i, lm_idx_nullspace_);
            auto r_i = storage_homogeneous_.template block<2, 1>(2 * i, res_idx_homogeneous_);

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
                b.template segment<11>(cam_idx_i * 11) +=
                        jp_i.transpose() * (r_i - jl_i * H_ll_inv_bl);
            }
        }
    }

    template <typename Derived>
    inline void get_Hll_inv_add_Hpp_b_joint(RowMatX& jp_t_jp,
                                      Eigen::MatrixBase<Derived>& Hll_inv,
                                      VecX& b, const Cameras& cameras,
                                      std::vector<std::mutex>& pose_mutex) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
        const auto jl_proj = storage_nullspace_.block(0,lm_idx_nullspace_,num_rows_nullspace_,3);

        Hll_inv = jl_proj.transpose() * jl_proj;
        Hll_inv += Proj.transpose() * lambda_ * Proj;
        Hll_inv = Hll_inv.inverse().eval();
        const Vec3 hll_inv_bl = Hll_inv * (jl_proj.transpose() * storage_homogeneous_.col(res_idx_homogeneous_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto& cam = cameras.at(cam_idx);

            const auto jp_proj = storage_nullspace_.template block<2,11>(2*i,0);
            const auto jl_pose = jl_proj.template block<2,3>(2*i,0);
            const auto r = storage_homogeneous_.template block<2, 1>(2 * i, res_idx_homogeneous_);

            const VecX tmp = jp_proj.transpose() * (r - jl_pose * hll_inv_bl);
            const RowMatX H_pp = jp_proj.transpose() * jp_proj;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<11>(cam_idx * 11) +=
                        tmp;
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
        const auto jl = storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3);
        Hll_inv = jl.transpose() * jl;
        Hll_inv = Hll_inv.inverse().eval();

        const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_pOSE_.col(res_idx_pOSE_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_pOSE_.template block<4, 12>(4 * i, 0);
            const auto jl_pose = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            const auto r = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);

            // fill b
            const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<12>(cam_idx * 12) +=
                        tmp;
                jp_t_jp.template block<12, 12>(12 * cam_idx, 0) +=
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
        Hll_inv = jl.transpose() * jl;
        Hll_inv.diagonal().array() += lambda_;
        Hll_inv = Hll_inv.inverse().eval();

        const Vec3 hll_inv_bl = Hll_inv * (jl.transpose() * storage_pOSE_.col(res_idx_pOSE_));

        for (size_t i = 0; i < pose_idx_.size(); ++i) {
            const size_t cam_idx = pose_idx_[i];
            const auto jp = storage_pOSE_.template block<4, 12>(4 * i, 0);
            const auto jl_pose = storage_pOSE_.template block<4, 3>(4 * i, lm_idx_pOSE_);
            const auto r = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);

            // fill b
            const VecX tmp = jp.transpose() * (r - jl_pose * hll_inv_bl);
            const RowMatX H_pp = jp.transpose() * jp;
            {
                std::scoped_lock lock(pose_mutex.at(cam_idx));
                b.template segment<12>(cam_idx * 12) +=
                        tmp;
                jp_t_jp.template block<12, 12>(12 * cam_idx, 0) +=
                        H_pp;
            }
        }
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
            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());

            auto jp_i = storage_homogeneous_.template block<2, 12>(2 * i, 0);
            auto jl_i = storage_homogeneous_.template block<2, 4>(2 * i, lm_idx_homogeneous_);
            auto r_i = storage_homogeneous_.template block<2, 1>(2 * i, res_idx_homogeneous_);
            auto jl_proj = jl_i * Proj;
            H_ll += jl_proj.transpose() * jl_proj;
            auto p_inc = pose_inc.template segment<11>(cam_idx_i * 11);
            tmp += jl_proj.transpose() * (r_i + jp_i * (Proj_pose * p_inc));
            J_inc.template segment<2>(2 * i) += jp_i * (Proj_pose * p_inc);
        }

        H_ll += Proj.transpose() * lambda_ * Proj;
        Vec3 inc = -H_ll.inverse() * tmp;
        VecX inc_proj = Proj * inc;
        // Add landmark jacobian cost change
        J_inc += storage_homogeneous_.block(0, lm_idx_homogeneous_, num_rows_, 4) * inc_proj;
        //@Simon: f(x+dx) = f(x) + g^T dx + 1/2 dx^T H dx
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_homogeneous_.col(res_idx_homogeneous_));
        // Note: scale only after computing model cost change
        inc_proj.array() *= Jl_col_scale_homogeneous.array();
        lm_ptr_->p_w_homogeneous += inc_proj;
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
            auto p_inc = pose_inc.template segment<12>(cam_idx_i * 12);
            tmp += jl_i.transpose() * (r_i + jp_i * (p_inc));
            J_inc.template segment<4>(4 * i) += jp_i * (p_inc);
        }

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

    inline void add_Jp_diag2_projective_space(VecX& res) const {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            res.template segment<12>(12 * cam_idx) +=
                    storage_homogeneous_.template block<2, 12>(2 * i, 0)
                            .colwise()
                            .squaredNorm();
        }
    }

    void back_substitute_pOSE(Scalar alpha, const VecX& pose_inc, Scalar& l_diff, const Cameras& cameras) {
        ROOTBA_ASSERT(state_ == LINEARIZED);

        Mat3 H_ll = Mat3::Zero();
        Vec3 tmp = Vec3::Zero();
        VecX J_inc;
        J_inc.setZero(num_rows_pOSE_);
        // Add pose-pose blocks and
        for (size_t i = 0; i < pose_idx_.size(); i++) {
            size_t cam_idx = pose_idx_[i];
            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);
            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE jl_i;

            Vec4 res;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE(alpha,
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, &Jp, &jl_i); /// @Simon: update Jl with the optimal pose parameters u* (cf Revisiting VarProj paper)

            auto jp_i = storage_pOSE_.template block<4, 12>(4 * i, 0);
            auto r_i = storage_pOSE_.template block<4, 1>(4 * i, res_idx_pOSE_);
            auto obs_i = storage_pOSE_.template block<2,1>(4 * i + 2, obs_idx_pOSE_);


            H_ll += jl_i.transpose() * jl_i;
            auto p_inc = pose_inc.template segment<12>(cam_idx * 12);

            tmp += jl_i.transpose() * res; // in line with the supplementary of "Revisiting the Variable Projection Method for Separable Nonlinear Least Squares Problems" (Hong et al., CVPR 2017)

            J_inc.template segment<4>(4 * i) += Jp * p_inc;
        }

        Vec3 inc = - H_ll.inverse() * tmp;

        J_inc += storage_pOSE_.block(0, lm_idx_pOSE_, num_rows_pOSE_, 3) * inc;
        l_diff -= J_inc.transpose() * (0.5 * J_inc + storage_pOSE_.col(res_idx_pOSE_));
        lm_ptr_->p_w += inc;
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
  RowMatX storage_pOSE_;
  RowMatX storage_nullspace_;

  Vec3 Jl_col_scale;
  Vec3 Jl_col_scale_pOSE;
  Vec4 Jl_col_scale_homogeneous;
  Scalar lambda_ = 0;

  std::vector<size_t> pose_idx_;

  size_t lm_idx_homogeneous_ = 0;
  size_t res_idx_homogeneous_ = 0;

  size_t num_rows_ = 0;
  size_t num_cols_homogeneous_ = 0;

  size_t num_rows_pOSE_ = 0;
  size_t num_cols_pOSE_ = 0;
  size_t lm_idx_pOSE_ = 0;
  size_t res_idx_pOSE_ = 0;
  size_t obs_idx_pOSE_ = 0;

  size_t num_rows_pOSE_homogeneous_ = 0;
  size_t num_cols_pOSE_homogeneous_ = 0;

  size_t num_rows_nullspace_ = 0;
  size_t num_cols_nullspace_ = 0;
  size_t lm_idx_nullspace_ = 0;

  Options options_;

  State state_ = UNINITIALIZED;

  Landmark* lm_ptr_ = nullptr;
};

}  // namespace rootba_povar
