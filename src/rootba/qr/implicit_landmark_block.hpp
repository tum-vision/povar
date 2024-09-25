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

template <typename Scalar, int POSE_SIZE>
class ImplicitLandmarkBlock {
 public:
  using Options = typename LandmarkBlock<Scalar>::Options;
  using State = typename LandmarkBlock<Scalar>::State;

  using Index = Eigen::Index;

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat2 = Eigen::Matrix<Scalar, 2, 2>;
  using Mat9 = Eigen::Matrix<Scalar, 9, 9>;

  using MatX3 = Eigen::Matrix<Scalar, Eigen::Dynamic, 3>;
  using MatX4 = Eigen::Matrix<Scalar, Eigen::Dynamic, 4>;
  using MatX9 = Eigen::Matrix<Scalar, Eigen::Dynamic, 9>;

  using RowMatXP =
      Eigen::Matrix<Scalar, Eigen::Dynamic, POSE_SIZE, Eigen::RowMajor>;

  using RowMat3 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;

  using HouseholderQR = Eigen::HouseholderQR<MatX3>;

  using Landmark = typename BalProblem<Scalar>::Landmark;
  using Camera = typename BalProblem<Scalar>::Camera;
  using Landmarks = typename BalProblem<Scalar>::Landmarks;
  using Cameras = typename BalProblem<Scalar>::Cameras;

  void allocate_landmark(Landmark& lm, const Options& options) {
    options_ = options;

    // verify assumptions
    ROOTBA_ASSERT_MSG(lm.obs.size() >= 2,
                      "implementation assumes at least 2 observations else Jl "
                      "doesn't have enough rows to be full rank");

    // remember global indcies of observing cameras
    pose_idx_.clear();
    pose_idx_.reserve(lm.obs.size());
    for (const auto& [cam_idx, obs] : lm.obs) {
      pose_idx_.push_back(cam_idx);
    }

    num_rows_ = pose_idx_.size() * 2;
    num_rows_damped_ = num_rows_ + 3;

    // allocate storage
    Jl_.resize(num_rows_damped_, 3);
    r_.resize(num_rows_);
    Jp_.resize(num_rows_, POSE_SIZE);
    QTr_.resize(num_rows_damped_);
    Jl_householder_ = HouseholderQR(num_rows_damped_, 3);

    state_ = State::ALLOCATED;
    lm_ptr_ = &lm;
  }

  inline void set_landmark_damping(const Scalar lambda) {
    // could be after backtracking or after first linearization
    ROOTBA_ASSERT(state_ == State::LINEARIZED || state_ == State::MARGINALIZED);
    lambda_ = lambda;
    // for now updating the lambda means that we have to redo qr
    state_ = State::LINEARIZED;
  };

  inline void linearize_landmark(const Cameras& cameras) {
    ROOTBA_ASSERT(state_ == State::ALLOCATED ||
                  state_ == State::NUMERICAL_FAILURE ||
                  state_ == State::LINEARIZED || state_ == State::MARGINALIZED);
    bool numerically_valid = true;

    // set damping rows in Jl and r to 0
    Jl_.template bottomRows<3>().setZero();

    for (Index i = 0; i < num_obs(); ++i) {
      size_t cam_idx = pose_idx_[i];

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

        Jpi_ref(i).template leftCols<6>() = sqrt_weight * Jp;
        Jpi_ref(i).template rightCols<3>() = sqrt_weight * Ji;
        Jli_ref(i) = sqrt_weight * Jl;
        ri_ref(i) = sqrt_weight * res;
      } else {
        Jpi_ref(i).setZero();
        Jli_ref(i).setZero();
        ri_ref(i).setZero();
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
                      state_ == State::LINEARIZED || state_ == State::MARGINALIZED);
        bool numerically_valid = true;

        // set damping rows in Jl and r to 0
        Jl_.template bottomRows<3>().setZero();

        for (Index i = 0; i < num_obs(); ++i) {
            size_t cam_idx = pose_idx_[i];

            const auto& obs = lm_ptr_->obs.at(cam_idx);
            const auto& cam = cameras.at(cam_idx);

            typename BalBundleAdjustmentHelper<Scalar>::MatRP_pOSE Jp;
            typename BalBundleAdjustmentHelper<Scalar>::MatRI_pOSE Ji;
            typename BalBundleAdjustmentHelper<Scalar>::MatRL_pOSE Jl;

            Vec2 res;
            Vec3 v_init;
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE(
                    obs.pos, lm_ptr_->p_w, cam.space_matrix, cam.intrinsics, true, res, false, &Jp, &Ji,
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

                Jpi_ref(i).template leftCols<12>() = sqrt_weight * Jp;
                Jpi_ref(i).template rightCols<3>() = sqrt_weight * Ji;
                Jli_ref(i) = sqrt_weight * Jl;
                ri_ref(i) = sqrt_weight * res;
            } else {
                Jpi_ref(i).setZero();
                Jli_ref(i).setZero();
                ri_ref(i).setZero();
            }
        }

        if (numerically_valid) {
            state_ = State::LINEARIZED;
        } else {
            state_ = State::NUMERICAL_FAILURE;
        }
    }


    inline void add_Jp_diag2(VecX& res,
                           std::vector<std::mutex>* pose_mutex) const {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    for (Index i = 0; i < num_obs(); ++i) {
      Index cam_idx = pose_idx_[i];
      auto dst = res.template segment<POSE_SIZE>(POSE_SIZE * cam_idx);
      if (pose_mutex) {
        std::scoped_lock lock(pose_mutex->at(cam_idx));
        dst.noalias() += Jpi_ref(i).colwise().squaredNorm();
      } else {
        dst.noalias() += Jpi_ref(i).colwise().squaredNorm();
      }
    }
  }

    inline void add_Jp_diag2_pOSE(VecX& res,
                             std::vector<std::mutex>* pose_mutex) const {
        ROOTBA_ASSERT(state_ == State::LINEARIZED);

        for (Index i = 0; i < num_obs(); ++i) {
            Index cam_idx = pose_idx_[i];
            auto dst = res.template segment<15>(15 * cam_idx);
            if (pose_mutex) {
                std::scoped_lock lock(pose_mutex->at(cam_idx));
                dst.noalias() += Jpi_ref(i).colwise().squaredNorm();
            } else {
                dst.noalias() += Jpi_ref(i).colwise().squaredNorm();
            }
        }
    }

  inline void scale_Jl_cols() {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
    // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
    Jl_col_scale_ =
        (options_.jacobi_scaling_eps + Jl_ref().colwise().norm().array())
            .inverse();

    Jl_ref() *= Jl_col_scale_.asDiagonal();
  }

  inline void scale_Jp_cols(const VecX& jacobian_scaling) {
    ROOTBA_ASSERT(state_ == State::LINEARIZED);

    for (Index i = 0; i < num_obs(); ++i) {
      Index cam_idx = pose_idx_[i];

      Jpi_ref(i) *=
          jacobian_scaling.template segment<POSE_SIZE>(POSE_SIZE * cam_idx)
              .asDiagonal();
    }
  }

  inline void perform_qr() {
    ROOTBA_ASSERT(state_ == State::LINEARIZED || state_ == State::MARGINALIZED);
    perform_qr_householder();
    state_ = State::MARGINALIZED;
  }

    inline void perform_qr_pOSE() {
        ROOTBA_ASSERT(state_ == State::LINEARIZED || state_ == State::MARGINALIZED);
        perform_qr_householder_pOSE();
        state_ = State::MARGINALIZED;
    }

  inline void add_Q2TJp_T_Q2TJp_blockdiag(
      BlockDiagonalAccumulator<Scalar>& accu,
      std::vector<std::mutex>* pose_mutex) const {
    add_Q2TJp_T_Q2TJp_blockdiag_sc(accu, pose_mutex);
    // add_Q2TJp_T_Q2TJp_blockdiag_qr(accu, pose_mutex);
  }

  inline void add_Q2TJp_T_Q2TJp_blockdiag_qr(
      BlockDiagonalAccumulator<Scalar>& accu,
      std::vector<std::mutex>* pose_mutex) const {
    ROOTBA_ASSERT(state_ == State::MARGINALIZED);

    // TODO: use scratch memory for temporary matrices

    // temporary matrices to hold QT_Jp (with damping rows) and Q2TJp_T_Q2TJp
    MatX9 QT_Jp(num_rows_damped_, POSE_SIZE);
    Mat9 Q2TJp_T_Q2TJp;

    for (Index i = 0; i < num_obs(); ++i) {
      // fill with zeros and Jpi
      QT_Jp.setZero();
      QT_Jp.template middleRows<2>(2 * i) = Jpi_ref(i);

      // compute QT_Jp in-place
      QT_Jp.applyOnTheLeft(Q_ref().transpose());

      // select only the Q2 part (no copy)
      auto Q2T_Jp = QT_Jp.bottomRows(num_rows_damped_ - 3);

      // compute diagonal block of RCS hessian
      Q2TJp_T_Q2TJp.noalias() = Q2T_Jp.transpose() * Q2T_Jp;

      // save to accumulator
      const size_t cam_idx = pose_idx_[i];
      if (pose_mutex) {
        std::scoped_lock lock(pose_mutex->at(cam_idx));
        accu.add(cam_idx, Q2TJp_T_Q2TJp);
      } else {
        accu.add(cam_idx, Q2TJp_T_Q2TJp);
      }
    }
  }

  inline void add_Q2TJp_T_Q2TJp_blockdiag_sc(
      BlockDiagonalAccumulator<Scalar>& accu,
      std::vector<std::mutex>* pose_mutex) const {
    ROOTBA_ASSERT(state_ == State::MARGINALIZED);

    // TODO: use scratch memory for temporary matrices

    Mat9 Q2TJp_T_Q2TJp;

    Eigen::Matrix<Scalar, 3, 3> R_inv;
    R_inv.setIdentity();
    Jl_householder_.matrixQR()
        .template topRows<3>()
        .template triangularView<Eigen::Upper>()
        .solveInPlace(R_inv);

    for (Index i = 0; i < num_obs(); ++i) {
      // compute diagonal block of RCS hessian

      Eigen::Matrix<Scalar, 2, 3> Jli_R_inv = Jli_ref(i) * R_inv;
      Mat2 I_minus_Jli_R_inv_R_inv_T_Jli_T = -Jli_R_inv * Jli_R_inv.transpose();
      I_minus_Jli_R_inv_R_inv_T_Jli_T.diagonal().array() += Scalar(1);

      Q2TJp_T_Q2TJp.noalias() =
          Jpi_ref(i).transpose() * I_minus_Jli_R_inv_R_inv_T_Jli_T * Jpi_ref(i);

      // save to accumulator
      const size_t cam_idx = pose_idx_[i];
      if (pose_mutex) {
        std::scoped_lock lock(pose_mutex->at(cam_idx));
        accu.add(cam_idx, Q2TJp_T_Q2TJp);
      } else {
        accu.add(cam_idx, Q2TJp_T_Q2TJp);
      }
    }
  }

  inline void add_Q2TJp_T_Q2Tr(VecX& res,
                               std::vector<std::mutex>* pose_mutex) const {
    ROOTBA_ASSERT(state_ == State::MARGINALIZED);

    // TODO: use scratch memory for temporary vector
    VecX y(num_rows_damped_);

    // Evaluate by matrix vector products:
    //   Q2TJp^T Q2Tr = Jp^T (Q2 Q2Tr)

    // get Q^T r and set top rows to 0 to get y = [0, Q2Tr^T]^T
    y = QTr_ref();
    y.template head<3>().setZero();

    // compute y = Q2 Q2Tr = Q [0, Q2Tr^T]^T
    y.applyOnTheLeft(Q_ref());

    // compute res += JpT Q2Q2Tr
    add_JpTy(res, y, pose_mutex);
  }

  inline size_t num_Q2T_rows() const { return num_rows_; }

  inline size_t num_reduced_cams() const { return pose_idx_.size(); }

  inline bool is_numerical_failure() const {
    return state_ == State::NUMERICAL_FAILURE;
  }

  inline void add_Q2TJp_T_Q2TJp_mult_x(
      VecX& res, const VecX& x_pose,
      std::vector<std::mutex>* pose_mutex) const {
    ROOTBA_ASSERT(state_ == State::MARGINALIZED);
    ROOTBA_ASSERT(res.size() == x_pose.size());

    // TODO: use scratch memory for temporary vector
    VecX y(num_rows_damped_);

    // Evaluate by matrix vector products:
    //   Q2TJp^T Q2TJp x = Jp^T (Q2 (Q2^T (Jp x)))

    // compute y = Jpx
    get_Jpx(y, x_pose);

    // compute Q^T Jpx and set top rows to 0 to get y = [0, Q2TJpx^T]^T
    y.applyOnTheLeft(Q_ref().transpose());
    y.template head<3>().setZero();

    // compute y = Q2 Q2TJpx = Q [0, Q2TJpx^T]^T
    y.applyOnTheLeft(Q_ref());

    // compute res += JpT Q2Q2Tr
    add_JpTy(res, y, pose_mutex);
  }

  void stage2(Scalar lambda, const VecX* jacobian_scaling,
              BlockDiagonalAccumulator<Scalar>* precond_block_diagonal,
              VecX& bref, std::vector<std::mutex>* pose_mutex) {
    // NOTE: One could consider also passing two separate lists of mutices, one
    // for computing the preconditioner and one for b. For now, we keep it
    // simple.

    // 1. scale jacobian
    if (jacobian_scaling) {
      scale_Jp_cols(*jacobian_scaling);
    }

    // 2. dampen landmarks
    set_landmark_damping(lambda);

    // 3. compute QR decomposition of Jl, store it in form of householder
    // vectors, and precompute QTr
    perform_qr();

    // 4. compute block diagonal preconditioner (SCHUR_JACOBI)
    if (precond_block_diagonal) {
      add_Q2TJp_T_Q2TJp_blockdiag(*precond_block_diagonal, pose_mutex);
    }

    // 5. compute rhs of reduced camera normal equations
    add_Q2TJp_T_Q2Tr(bref, pose_mutex);
  }

  void back_substitute(const VecX& pose_inc, Scalar& l_diff) {
    ROOTBA_ASSERT(state_ == State::MARGINALIZED);

    // TODO: use scratch memory for temporary vector
    VecX y(num_rows_damped_);

    // Evaluate by matrix vector products:
    //   - R1^-1 (Q1Tr + Q1TJp x) = R1^-1 (-(Q1Tr + (Q1^T (Jp x))))

    // compute y = Jpx
    get_Jpx(y, pose_inc);

    // compute Q^T Jpx; we have y[0:3] == Q1TJpx
    y.applyOnTheLeft(Q_ref().transpose());

    // compute -(Q1Tr + Q1TJpx)
    Vec3 inc = -(Q1Tr_ref() + y.template head<3>());

    // compute inc = R1^-1 (-(Q1Tr + Q1TJpx))
    R1_solve_in_place(inc);

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

    // Note: we compute model cost change w/o landmark damping. For now we use
    // the Jacobians before marginalization. If we later want to do the QR of Jl
    // in-place, we have to have to figure out how to undo the damping. Maybe we
    // can go back to Givens rotations, but this might make the matrix
    // multiplication in CG less efficient, which is the core bottleneck
    // Idea: For model cost computation, we can compute model cost change with
    // damping, and then compute it just for damping and subtract it again.

    // reuse scratch memory
    Eigen::Map<VecX> J_inc(y.data(), num_rows_);

    // compute Jp inc_p
    get_Jpx_no_damping(J_inc, pose_inc);

    // add Jl inc_l
    add_Jlx_no_damping(J_inc, inc);

    // TODO: use scratch memory for temporary vector in the following
    // computation (0.5 * J_inc + r_ref()), at least I think it needs to
    // allocate a temporary below... Alternatively we can compute it as
    // -(Jinc^T r) - 0.5 (Jinc^T Jinc) without need for additional memory (but
    // more multiplications).

    // compute - (Jinc)T (r + 0.5 (Jinc))
    l_diff -= J_inc.transpose() * (0.5 * J_inc + r_ref());

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

      std::cout.flush();
      LOG(FATAL) << "Numerical failure";
    }

    // Note: scale only after computing model cost change
    inc.array() *= Jl_col_scale_.array();
    lm_ptr_->p_w += inc;
  }

 protected:
  inline void perform_qr_householder() {
    // set damping in bottom three rows of Jl
    Jl_.template bottomRows<3>().diagonal().setConstant(std::sqrt(lambda_));

    // compute QR
    Jl_householder_.compute(Jl_);

    // precompute QTr
    QTr_.head(num_rows_) = r_;
    QTr_.template tail<3>().setZero();
    QTr_.applyOnTheLeft(Q_ref().transpose());
  }

    inline void perform_qr_householder_pOSE() {
        // set damping in bottom three rows of Jl
        Jl_.template bottomRows<3>().diagonal().setConstant(std::sqrt(lambda_));

        // compute QR
        Jl_householder_.compute(Jl_);

        // precompute QTr
        QTr_.head(num_rows_) = r_;
        QTr_.template tail<3>().setZero();
        QTr_.applyOnTheLeft(Q_ref().transpose());
    }

  // internal multiplication helper for Jp * x (w/o damping rows)
  template <class Derived>
  inline void get_Jpx_no_damping(Eigen::MatrixBase<Derived>& res,
                                 const VecX& x_pose) const {
    // assumption: res has length >= num_rows_

    // compute y = Jpx block-wise
    for (Index i = 0; i < num_obs(); ++i) {
      const size_t cam_idx = pose_idx_[i];
      res.template segment<2>(2 * i).noalias() =
          Jpi_ref(i) * x_pose.template segment<POSE_SIZE>(POSE_SIZE * cam_idx);
    }
  }

  // internal multiplication helper for Jp * x (w/ damping rows)
  inline void get_Jpx(VecX& res, const VecX& x_pose) const {
    // assumption: res has length >= num_rows_damped_

    // compute y = Jpx block-wise
    get_Jpx_no_damping(res, x_pose);

    // landmark damping rows are 0
    res.template tail<3>().setZero();
  }

  // intern multiplication helper for Jl * x (w/o damping rows)
  template <class Derived>
  inline void add_Jlx_no_damping(Eigen::MatrixBase<Derived>& res,
                                 const Vec3& x_lm) {
    // assumption: res has length >= num_rows_

    // compute y += Jlx for first 2*obs rows
    res.head(num_rows_).noalias() += Jl_ref() * x_lm;
  }

  // internal multiplication helper for JpT * y
  inline void add_JpTy(VecX& res, const VecX& y,
                       std::vector<std::mutex>* pose_mutex) const {
    // JpT is block-diagonal with 9x2 blocks (and 3 extra 0 columns on the right
    // for landmark damping) --> compute JpT y block-wise
    for (size_t i = 0; i < pose_idx_.size(); ++i) {
      size_t cam_idx = pose_idx_[i];
      auto dst = res.template segment<POSE_SIZE>(POSE_SIZE * cam_idx);
      if (pose_mutex) {
        std::scoped_lock lock(pose_mutex->at(cam_idx));
        dst.noalias() +=
            Jpi_ref(i).transpose() * y.template middleRows<2>(2 * i);
      } else {
        dst.noalias() +=
            Jpi_ref(i).transpose() * y.template middleRows<2>(2 * i);
      }
    }
  }

  // internal accessors to Jl (w/o damping rows)
  inline auto Jl_ref() { return Jl_.topRows(num_rows_); }
  inline auto Jl_ref() const { return Jl_.topRows(num_rows_); }
  inline auto Jli_ref(Index i) { return Jl_.template middleRows<2>(2 * i); }
  inline auto Jli_ref(Index i) const {
    return Jl_.template middleRows<2>(2 * i);
  }

  // internal accessors to r (w/o damping rows)
  inline auto r_ref() { return r_.head(num_rows_); }
  inline auto r_ref() const { return r_.head(num_rows_); }
  inline auto ri_ref(Index i) { return r_.template segment<2>(2 * i); }
  inline auto ri_ref(Index i) const { return r_.template segment<2>(2 * i); }

  // internal accessors to the non-zero blocks of Jp
  inline auto Jpi_ref(Index i) { return Jp_.template middleRows<2>(2 * i); }
  inline auto Jpi_ref(Index i) const {
    return Jp_.template middleRows<2>(2 * i);
  }

  // internal accessor to Q and R
  inline auto Q_ref() const { return Jl_householder_.householderQ(); }

  inline void R1_solve_in_place(Vec3& x) const {
    return Jl_householder_.matrixQR()
        .template topRows<3>()
        .template triangularView<Eigen::Upper>()
        .solveInPlace(x);
  }

  // internal accessor to QTr
  inline auto QTr_ref() { return QTr_; }
  inline auto QTr_ref() const { return QTr_; }
  inline auto Q1Tr_ref() { return QTr_.template head<3>(); }
  inline auto Q1Tr_ref() const { return QTr_.template head<3>(); }
  inline auto Q2Tr_ref() { return QTr_.tail(num_rows_damped_ - 3); }
  inline auto Q2Tr_ref() const { return QTr_.tail(num_rows_damped_ - 3); }

  // helper for loops over all observations
  Index num_obs() const { return Index(pose_idx_.size()); }

  // For now we allocate 3 extra rows for damping, as it simplifies computing
  // the Householder transformation.
  MatX3 Jl_;  // 2*obs+3 x 3

  // Residual vector w/o damping rows
  VecX r_;  // 2*obs

  // [Jp_i] blocks stacked vertically as a 2n x 9 row-major matrix
  // TODO: reconsider if it's better to have Jp row-major and stacked vertically
  // or col-major and stacked horizontally
  RowMatXP Jp_;  // 2*obs x 9

  // Note: for now we store the Householder state and QTr separately, so we
  // don't have to worry about undoing damping, etc, in the initial version.
  // Later we can do the QR in-place in Jl_ and maybe also store r as [Jl, r].
  HouseholderQR Jl_householder_;  // 2*obs+3 x 3

  // TODO: consider if it's worth pre-computing and storing this. We need it
  // just twice per outer iteration. Once to compute the RCS gradient, and once
  // in the backsubstitution.
  VecX QTr_;  // 2*obs+3

  // map local to global pose indices
  std::vector<size_t> pose_idx_;

  // size_t num_cols_;
  Index num_rows_;
  Index num_rows_damped_;

  Scalar lambda_ = 0;
  Vec3 Jl_col_scale_;

  Options options_;

  State state_ = State::UNINITIALIZED;

  Landmark* lm_ptr_ = nullptr;
};

}  // namespace rootba
