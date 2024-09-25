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

#include "rootba/cg/block_sparse_matrix.hpp"
#include "rootba/solver/linearizor_base.hpp"

namespace rootba {

template <typename Scalar, int POSE_SIZE>
class LinearizationQR;

template <class Scalar_>
class LinearizorQR : public LinearizorBase<Scalar_> {
 public:
  using Scalar = Scalar_;
  using Base = LinearizorBase<Scalar>;
  //constexpr static int POSE_SIZE = 9;
  //constexpr static int POSE_SIZE = 15;
  constexpr static int POSE_SIZE = 11;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

 public:  // public interface
  LinearizorQR(BalProblem<Scalar>& bal_problem, const SolverOptions& options,
               SolverSummary* summary = nullptr);

  ~LinearizorQR() override;

    void compute_plane_linearly() override;

  void linearize() override;
  void linearize_refine() override;
  void linearize_pOSE(int alpha) override;
    void linearize_RpOSE(double alpha) override;
    void linearize_RpOSE_refinement(double alpha) override;
    void linearize_RpOSE_ML() override;
    void linearize_metric_upgrade() override;
    void linearize_metric_upgrade_v2() override;
    void linearize_expOSE(int alpha, bool init) override;
  void linearize_pOSE_rOSE(int alpha) override;
  void linearize_rOSE(int alpha) override;
  void linearize_pOSE_homogeneous(int alpha) override;
  void linearize_affine_space() override;
  void linearize_projective_space() override;
  void linearize_projective_space_homogeneous() override;
    void linearize_projective_space_homogeneous_RpOSE() override;
  void linearize_projective_space_homogeneous_lm_landmark() override;

  void update_y_tilde_expose() override;
  void initialize_y_tilde_expose() override;
    void rpose_new_equilibrium() override;

    void estimate_alpha(VecX&& inc, Scalar lambda) override;

  VecX solve(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_joint(Scalar lambda, Scalar relative_error_change) override;
    VecX solve_joint_RpOSE(Scalar lambda, Scalar relative_error_change) override;

    VecX solve_refine(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_affine_space(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_pOSE(Scalar lambda, Scalar relative_error_change) override;
    VecX solve_RpOSE(Scalar lambda, Scalar relative_error_change) override;
    VecX solve_RpOSE_ML(Scalar lambda, Scalar relative_error_change) override;
    VecX solve_pOSE_poBA(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_metric_upgrade(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_direct_pOSE(Scalar lambda, Scalar relative_error_change) override;
    VecX solve_direct_RpOSE(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_direct_expOSE(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_expOSE(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_pOSE_rOSE(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_rOSE(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_pOSE_riemannian_manifold(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_pOSE_homogeneous(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_projective_space(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_projective_space_homogeneous(Scalar lambda, Scalar relative_error_change) override;
  VecX solve_projective_space_homogeneous_riemannian_manifold(Scalar lambda, Scalar relative_error_change) override;

  void non_linear_varpro_landmark(Scalar lambda) override;
  void apply_pose_update(VecX&& inc) override;
  void apply_pose_update_riemannian_manifold(VecX&& inc) override;
  void closed_form_projective_space_homogeneous_nonlinear_initialization() override;
  void closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold() override;

  void radial_estimate() override;

  Scalar apply(VecX&& inc) override;
  Scalar apply_joint(VecX&& inc) override;
    Scalar apply_joint_RpOSE(VecX&& inc) override;

  Scalar closed_form(VecX&& inc) override;
  Scalar closed_form_affine_space(VecX&& inc) override;
  Scalar closed_form_pOSE(int alpha, VecX&& inc) override;
    Scalar closed_form_RpOSE(double alpha, VecX&& inc) override;
    Scalar closed_form_RpOSE_ML(VecX&& inc) override;
    Scalar closed_form_RpOSE_refinement(double alpha, VecX&& inc) override;
  Scalar closed_form_pOSE_poBA(int alpha, VecX&& inc) override;
  Scalar closed_form_expOSE(int alpha, VecX&& inc) override;
  Scalar closed_form_pOSE_rOSE(int alpha, VecX&& inc) override;
  Scalar closed_form_rOSE(int alpha, VecX&& inc) override;
  Scalar closed_form_pOSE_riemannian_manifold(int alpha, VecX&& inc) override;
  Scalar closed_form_pOSE_homogeneous(int alpha, VecX&& inc) override;
  Scalar closed_form_projective_space(VecX&& inc) override;
  Scalar closed_form_projective_space_homogeneous(VecX&& inc) override;

 private:
  using Base::bal_problem_;
  using Base::it_summary_;
  using Base::options_;
  using Base::summary_;

  std::unique_ptr<LinearizationQR<Scalar, POSE_SIZE>> lqr_;

  // set during linearization, used in solve
  VecX pose_jacobian_scaling_;

  // diagonal blocks for JACOBI or SCHUR_JACOBI preconditioner
  IndexedBlocks<Scalar> precond_blocks_;

  // indicates if we call solve the first time since the last linearization
  // (first inner iteration for LM-backtracking); true after `linearize`, false
  // after `solve`;
  bool new_linearization_point_ = false;
};

}  // namespace rootba
