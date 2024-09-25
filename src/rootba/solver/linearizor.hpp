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

#include "rootba/bal/bal_problem.hpp"
#include "rootba/bal/residual_info.hpp"
#include "rootba/bal/solver_options.hpp"
#include "rootba/solver/solver_summary.hpp"

namespace rootba {

// TODO: can we find a better name than Linearizor? Linearization is already
// taken for the more low-level API in LinearizationQR and LinearizationSC.
template <typename Scalar_>
class Linearizor {
 public:
  using Scalar = Scalar_;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

 public:
  // factory method
  static std::unique_ptr<Linearizor<Scalar>> create(
      BalProblem<Scalar>& bal_problem, const SolverOptions& options,
      SolverSummary* summary = nullptr);
    static std::unique_ptr<Linearizor<Scalar>> create_pcg(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            SolverSummary* summary = nullptr);

  virtual ~Linearizor() = default;

  // start a new solver iteration and set (optional) summary for logging
  virtual void start_iteration(IterationSummary* it_summary = nullptr) = 0;

  // compute error (with logging; use after `start_iteration`)
  virtual void compute_error(ResidualInfo& ri) = 0;
  virtual void compute_error_affine_space(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_pOSE(ResidualInfo& ri, bool initialization_varproj) = 0;
    virtual void compute_error_RpOSE(ResidualInfo& ri, bool initialization_varproj) = 0;
    virtual void compute_error_RpOSE_ML(ResidualInfo& ri) = 0;
    virtual void compute_error_RpOSE_refinement(ResidualInfo& ri, double alpha) = 0;
    virtual void compute_error_metric_upgrade(ResidualInfo& ri) = 0;
    virtual void compute_error_expOSE(ResidualInfo& ri, bool initialization_varproj) = 0;
    virtual void compute_error_pOSE_rOSE(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_rOSE(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_pOSE_homogeneous(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_projective_space(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_projective_space_homogeneous(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_projective_space_homogeneous_RpOSE(ResidualInfo& ri, bool initialization_varproj) = 0;
    virtual void compute_error_projective_space_homogeneous_RpOSE_test_rotation(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_projective_space_homogeneous_lm_landmark(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_refine(ResidualInfo& ri) = 0;

  virtual void initialize_varproj_lm(bool initialization_varproj) = 0;

  virtual void initialize_varproj_lm_affine_space(bool initialization_varproj) = 0;
  virtual void initialize_varproj_lm_pOSE(int alpha, bool initialization_varproj) = 0;
    virtual void initialize_varproj_lm_RpOSE(double alpha, bool initialization_varproj) = 0;
    virtual void initialize_varproj_lm_expOSE(int alpha, bool initialization_varproj) = 0;
  virtual void initialize_varproj_lm_pOSE_rOSE(int alpha, bool initialization_varproj) = 0;
  virtual void initialize_varproj_lm_rOSE(int alpha, bool initialization_varproj) = 0;
  virtual void initialize_varproj_lm_pOSE_homogeneous(int alpha, bool initialization_varproj) = 0;

  virtual void radial_estimate() = 0;


  // called once for every new linearization point
  // TODO: add optional output parameter `ResidualInfo* ri`
  virtual void linearize() = 0;
  virtual void linearize_refine() = 0;
  virtual void linearize_affine_space() = 0;
  virtual void linearize_projective_space() = 0;
  virtual void linearize_pOSE(int alpha) = 0;
    virtual void linearize_RpOSE(double alpha) = 0;
    virtual void linearize_RpOSE_ML() = 0;
    virtual void linearize_RpOSE_refinement(double alpha) = 0;
    virtual void linearize_metric_upgrade() = 0;
    virtual void linearize_metric_upgrade_v2() = 0;
    virtual void linearize_expOSE(int alpha, bool init) = 0;
  virtual void linearize_pOSE_rOSE(int alpha) = 0;
  virtual void linearize_rOSE(int alpha) = 0;
  virtual void linearize_pOSE_homogeneous(int alpha) = 0;
  virtual void linearize_projective_space_homogeneous() = 0;
  virtual void linearize_projective_space_homogeneous_RpOSE() = 0;
  virtual void linearize_projective_space_homogeneous_lm_landmark() = 0;

  virtual void update_y_tilde_expose() = 0;
    virtual void initialize_y_tilde_expose() = 0;
    virtual void rpose_new_equilibrium() = 0;

    virtual void estimate_alpha(VecX&& inc, Scalar lambda) = 0;
  // maybe be called multiple times with different lambda for the same
  // linearization point (no call of `linearize` in between); returns camera
  // increment
  virtual VecX solve(Scalar lambda, Scalar relative_error_change) = 0;
  virtual VecX solve_joint(Scalar lambda, Scalar relative_error_change) = 0;
    virtual VecX solve_joint_RpOSE(Scalar lambda, Scalar relative_error_change) = 0;
  virtual VecX solve_refine(Scalar lambda, Scalar relative_error_change) = 0;

  virtual  VecX solve_affine_space(Scalar lambda, Scalar relative_error_change) = 0;
  virtual  VecX solve_pOSE(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_RpOSE(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_RpOSE_ML(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_pOSE_poBA(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_metric_upgrade(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_direct_pOSE(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_direct_RpOSE(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_direct_expOSE(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_expOSE(Scalar lambda, Scalar relative_error_change) = 0;
    virtual  VecX solve_pOSE_rOSE(Scalar lambda, Scalar relative_error_change) = 0;
  virtual  VecX solve_rOSE(Scalar lambda, Scalar relative_error_change) = 0;
  virtual  VecX solve_pOSE_riemannian_manifold(Scalar lambda, Scalar relative_error_change) = 0;
  virtual  VecX solve_pOSE_homogeneous(Scalar lambda, Scalar relative_error_change) = 0;
  virtual  VecX solve_projective_space(Scalar lambda, Scalar relative_error_change) = 0;
  virtual  VecX solve_projective_space_homogeneous(Scalar lambda, Scalar relative_error_change) = 0;
  virtual  VecX solve_projective_space_homogeneous_riemannian_manifold(Scalar lambda, Scalar relative_error_change) = 0;
  // apply camera increment (backsubstitute and update cameras)
  // returns model cost change l_diff
  virtual Scalar apply(VecX&& inc) = 0;
  virtual Scalar apply_joint(VecX&& inc) = 0;
    virtual Scalar apply_joint_RpOSE(VecX&& inc) = 0;

  virtual Scalar closed_form(VecX&& inc) = 0;

  virtual void compute_plane_linearly() = 0;

  virtual void non_linear_varpro_landmark(Scalar lambda) = 0;
  virtual void apply_pose_update(VecX&& inc) = 0;
  virtual void apply_pose_update_riemannian_manifold(VecX&& inc) = 0;

  virtual void closed_form_projective_space_homogeneous_nonlinear_initialization() = 0;
  virtual void closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold() = 0;
  virtual  Scalar closed_form_affine_space(VecX&& inc) = 0;
  virtual  Scalar closed_form_pOSE(int alpha, VecX&& inc) = 0;
    virtual  Scalar closed_form_RpOSE(double alpha, VecX&& inc) = 0;
    virtual  Scalar closed_form_RpOSE_ML(VecX&& inc) = 0;
    virtual  Scalar closed_form_RpOSE_refinement(double alpha, VecX&& inc) = 0;
    virtual  Scalar closed_form_pOSE_poBA(int alpha, VecX&& inc) = 0;
  virtual  Scalar closed_form_expOSE(int alpha, VecX&& inc) = 0;
  virtual  Scalar closed_form_pOSE_rOSE(int alpha, VecX&& inc) = 0;
  virtual  Scalar closed_form_rOSE(int alpha, VecX&& inc) = 0;
  virtual  Scalar closed_form_pOSE_riemannian_manifold(int alpha, VecX&& inc) = 0;
  virtual  Scalar closed_form_pOSE_homogeneous(int alpha, VecX&& inc) = 0;
  virtual  Scalar closed_form_projective_space(VecX&& inc) = 0;
  virtual  Scalar closed_form_projective_space_homogeneous(VecX&& inc) = 0;

    // finalize logging for a single solver iteration
  virtual void finish_iteration() = 0;
};

}  // namespace rootba
