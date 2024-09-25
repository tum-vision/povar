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
#include "rootba/solver/linearizor_qr.hpp"

#include "rootba/cg/conjugate_gradient.hpp"
#include "rootba/cg/preconditioner.hpp"
#include "rootba/qr/linearization_qr.hpp"
#include "rootba/util/time_utils.hpp"

// helper to deal with summary_ and it_summary_ pointers
#define IF_SET(POINTER_VAR) \
  if (POINTER_VAR) POINTER_VAR

namespace rootba {

template <class Scalar_>
LinearizorQR<Scalar_>::LinearizorQR(BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options,
                                    SolverSummary* summary)
    : LinearizorBase<Scalar>(bal_problem, options, summary) {
  // set options
  //typename LinearizationQR<Scalar, 9>::Options lqr_options;
    //typename LinearizationQR<Scalar, 15>::Options lqr_options;
    typename LinearizationQR<Scalar, 11>::Options lqr_options;
  lqr_options.lb_options.use_householder =
      options_.use_householder_marginalization;
  lqr_options.lb_options.use_valid_projections_only =
      options_.use_projection_validity_check();
  lqr_options.lb_options.jacobi_scaling_eps =
      Base::get_effective_jacobi_scaling_epsilon();

  lqr_options.reduction_alg = options_.reduction_alg;
  lqr_options.lb_options.residual_options = options_.residual;

  // create linearization object
  //lqr_ = std::make_unique<LinearizationQR<Scalar, 9>>(bal_problem, lqr_options);
    //lqr_ = std::make_unique<LinearizationQR<Scalar, 15>>(bal_problem, lqr_options);
    lqr_ = std::make_unique<LinearizationQR<Scalar, 11>>(bal_problem, lqr_options);
}

template <class Scalar_>
LinearizorQR<Scalar_>::~LinearizorQR() = default;

template <class Scalar_>
void LinearizorQR<Scalar_>::radial_estimate() {

    }

template <class Scalar_>
void LinearizorQR<Scalar_>::linearize() {
  // ////////////////////////////////////////////////////////////////////////
  // Stage 1: outside LM solver inner-loop
  // - linearization
  // - scale landmark Jacobians
  // - compute pose Jacobian scale
  // - marginalization of landmarks
  // - compute JACOBI preconditioner
  // ////////////////////////////////////////////////////////////////////////
  Timer timer_stage1;

  const bool use_jacobi_precond =
      options_.preconditioner_type == SolverOptions::PreconditionerType::JACOBI;

  VecX pose_jacobain_scale2;

  if (!options_.staged_execution) {
    Timer timer;
    CHECK(lqr_->linearize_problem())
        << "did not expect numerical failure during linearization";
    IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

    pose_jacobain_scale2 = lqr_->get_Jp_diag2();
    lqr_->scale_Jl_cols();
    IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds =
        timer.reset();

    if (use_jacobi_precond) {
      precond_blocks_ = lqr_->get_Jp_T_Jp_blockdiag();
      IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
          timer.reset();
    }

    lqr_->perform_qr();
    IF_SET(it_summary_)->perform_qr_time_in_seconds = timer.reset();
  } else {
    // TODO: for now we crash on numerical failure at this point; for better
    // error handling we should probably return from the solver properly with
    // FAILURE status and informative message; same above for 'unstaged'
    // version and after initial error computation

    pose_jacobain_scale2 =
        lqr_->get_stage1(use_jacobi_precond ? &precond_blocks_ : nullptr);
    CHECK(pose_jacobain_scale2.size() > 0)
        << "did not expect numerical failure during linearization";
  }

  // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
  // avoid accidental use of outdated info?

  // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
  // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
  pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                            pose_jacobain_scale2.array().sqrt())
                               .inverse();

  IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
  IF_SET(summary_)->num_jacobian_evaluations += 1;

  new_linearization_point_ = true;
}

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_refine() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // - marginalization of landmarks
        // - compute JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        const bool use_jacobi_precond =
                options_.preconditioner_type == SolverOptions::PreconditionerType::JACOBI;

        VecX pose_jacobain_scale2;

        if (!options_.staged_execution) {
            Timer timer;
            CHECK(lqr_->linearize_problem())
                            << "did not expect numerical failure during linearization";
            IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

            pose_jacobain_scale2 = lqr_->get_Jp_diag2();
            lqr_->scale_Jl_cols();
            IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds =
                                       timer.reset();

            if (use_jacobi_precond) {
                precond_blocks_ = lqr_->get_Jp_T_Jp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            lqr_->perform_qr();
            IF_SET(it_summary_)->perform_qr_time_in_seconds = timer.reset();
        } else {
            // TODO: for now we crash on numerical failure at this point; for better
            // error handling we should probably return from the solver properly with
            // FAILURE status and informative message; same above for 'unstaged'
            // version and after initial error computation

            pose_jacobain_scale2 =
                    lqr_->get_stage1(use_jacobi_precond ? &precond_blocks_ : nullptr);
            CHECK(pose_jacobain_scale2.size() > 0)
                            << "did not expect numerical failure during linearization";
        }

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_jacobain_scale2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));

        }
        inc.array() *= pose_jacobian_scaling_.array().inverse();
        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_affine_space(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_pOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_RpOSE(double alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_RpOSE_ML(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_RpOSE_refinement(double alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_pOSE_poBA(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_expOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_pOSE_rOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_rOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_pOSE_riemannian_manifold(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_pOSE_homogeneous(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }



    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_projective_space(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::closed_form_projective_space_homogeneous(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_pOSE(int alpha) {
// ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // - marginalization of landmarks
        // - compute JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        const bool use_jacobi_precond =
                options_.preconditioner_type == SolverOptions::PreconditionerType::JACOBI;

        VecX pose_jacobain_scale2;

        if (!options_.staged_execution) {
            Timer timer;
            CHECK(lqr_->linearize_problem_pOSE())
                            << "did not expect numerical failure during linearization";
            IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

            pose_jacobain_scale2 = lqr_->get_Jp_diag2_pOSE();
            //lqr_->scale_Jl_cols_pOSE();
            IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds =
                                       timer.reset();

            if (use_jacobi_precond) {
                precond_blocks_ = lqr_->get_Jp_T_Jp_blockdiag_pOSE();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            lqr_->perform_qr_pOSE();
            IF_SET(it_summary_)->perform_qr_time_in_seconds = timer.reset();
        } else {
            // TODO: for now we crash on numerical failure at this point; for better
            // error handling we should probably return from the solver properly with
            // FAILURE status and informative message; same above for 'unstaged'
            // version and after initial error computation

            pose_jacobain_scale2 =
                    lqr_->get_stage1(use_jacobi_precond ? &precond_blocks_ : nullptr);
            CHECK(pose_jacobain_scale2.size() > 0)
                            << "did not expect numerical failure during linearization";
        }

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_jacobain_scale2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;

    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_RpOSE(double alpha) {
// ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // - marginalization of landmarks
        // - compute JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        const bool use_jacobi_precond =
                options_.preconditioner_type == SolverOptions::PreconditionerType::JACOBI;

        VecX pose_jacobain_scale2;

        if (!options_.staged_execution) {
            Timer timer;
            CHECK(lqr_->linearize_problem_pOSE())
                            << "did not expect numerical failure during linearization";
            IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

            pose_jacobain_scale2 = lqr_->get_Jp_diag2_pOSE();
            //lqr_->scale_Jl_cols_pOSE();
            IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds =
                                       timer.reset();

            if (use_jacobi_precond) {
                precond_blocks_ = lqr_->get_Jp_T_Jp_blockdiag_pOSE();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            lqr_->perform_qr_pOSE();
            IF_SET(it_summary_)->perform_qr_time_in_seconds = timer.reset();
        } else {
            // TODO: for now we crash on numerical failure at this point; for better
            // error handling we should probably return from the solver properly with
            // FAILURE status and informative message; same above for 'unstaged'
            // version and after initial error computation

            pose_jacobain_scale2 =
                    lqr_->get_stage1(use_jacobi_precond ? &precond_blocks_ : nullptr);
            CHECK(pose_jacobain_scale2.size() > 0)
                            << "did not expect numerical failure during linearization";
        }

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_jacobain_scale2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;

    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_RpOSE_ML() {
// ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // - marginalization of landmarks
        // - compute JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        const bool use_jacobi_precond =
                options_.preconditioner_type == SolverOptions::PreconditionerType::JACOBI;

        VecX pose_jacobain_scale2;

        if (!options_.staged_execution) {
            Timer timer;
            CHECK(lqr_->linearize_problem_pOSE())
                            << "did not expect numerical failure during linearization";
            IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

            pose_jacobain_scale2 = lqr_->get_Jp_diag2_pOSE();
            //lqr_->scale_Jl_cols_pOSE();
            IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds =
                                       timer.reset();

            if (use_jacobi_precond) {
                precond_blocks_ = lqr_->get_Jp_T_Jp_blockdiag_pOSE();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            lqr_->perform_qr_pOSE();
            IF_SET(it_summary_)->perform_qr_time_in_seconds = timer.reset();
        } else {
            // TODO: for now we crash on numerical failure at this point; for better
            // error handling we should probably return from the solver properly with
            // FAILURE status and informative message; same above for 'unstaged'
            // version and after initial error computation

            pose_jacobain_scale2 =
                    lqr_->get_stage1(use_jacobi_precond ? &precond_blocks_ : nullptr);
            CHECK(pose_jacobain_scale2.size() > 0)
                            << "did not expect numerical failure during linearization";
        }

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_jacobain_scale2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;

    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_RpOSE_refinement(double alpha) {
// ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // - marginalization of landmarks
        // - compute JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        const bool use_jacobi_precond =
                options_.preconditioner_type == SolverOptions::PreconditionerType::JACOBI;

        VecX pose_jacobain_scale2;

        if (!options_.staged_execution) {
            Timer timer;
            CHECK(lqr_->linearize_problem_pOSE())
                            << "did not expect numerical failure during linearization";
            IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

            pose_jacobain_scale2 = lqr_->get_Jp_diag2_pOSE();
            //lqr_->scale_Jl_cols_pOSE();
            IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds =
                                       timer.reset();

            if (use_jacobi_precond) {
                precond_blocks_ = lqr_->get_Jp_T_Jp_blockdiag_pOSE();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            lqr_->perform_qr_pOSE();
            IF_SET(it_summary_)->perform_qr_time_in_seconds = timer.reset();
        } else {
            // TODO: for now we crash on numerical failure at this point; for better
            // error handling we should probably return from the solver properly with
            // FAILURE status and informative message; same above for 'unstaged'
            // version and after initial error computation

            pose_jacobain_scale2 =
                    lqr_->get_stage1(use_jacobi_precond ? &precond_blocks_ : nullptr);
            CHECK(pose_jacobain_scale2.size() > 0)
                            << "did not expect numerical failure during linearization";
        }

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_jacobain_scale2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;

    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_metric_upgrade() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_metric_upgrade_v2() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::estimate_alpha(VecX&& inc, Scalar lambda) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::compute_plane_linearly() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_expOSE(int alpha, bool init) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::update_y_tilde_expose() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }
    template <class Scalar_>
    void LinearizorQR<Scalar_>::initialize_y_tilde_expose() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::rpose_new_equilibrium() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }




    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_pOSE_rOSE(int alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_rOSE(int alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_pOSE_homogeneous(int alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::non_linear_varpro_landmark(Scalar lambda) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::closed_form_projective_space_homogeneous_nonlinear_initialization() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::apply_pose_update(VecX&& inc) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::apply_pose_update_riemannian_manifold(VecX&& inc) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_affine_space() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_projective_space() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_projective_space_homogeneous() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_projective_space_homogeneous_RpOSE() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    void LinearizorQR<Scalar_>::linearize_projective_space_homogeneous_lm_landmark() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;


    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_pOSE(
            Scalar lambda, Scalar relative_error_change) {
        VecX inc;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_RpOSE(
            Scalar lambda, Scalar relative_error_change) {
        VecX inc;

        return inc;
    }
    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_RpOSE_ML(
            Scalar lambda, Scalar relative_error_change) {
        VecX inc;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_pOSE_poBA(
            Scalar lambda, Scalar relative_error_change) {
        VecX inc;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_metric_upgrade(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_direct_pOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_direct_RpOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_direct_expOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_expOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }



    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_pOSE_rOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_rOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_pOSE_riemannian_manifold(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_pOSE_homogeneous(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_affine_space(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_projective_space(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_projective_space_homogeneous(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_projective_space_homogeneous_riemannian_manifold(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////



        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());


        return inc;
    }



    template <class Scalar_>
typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve(
    Scalar lambda, Scalar relative_error_change) {
  // ////////////////////////////////////////////////////////////////////////
  // Stage 2: inside LM solver inner-loop
  // - scale pose Jacobians (1st inner it)
  // - dampen
  // - compute RHS b
  // - compute SCHUR_JACOBI preconditioner
  // ////////////////////////////////////////////////////////////////////////
  Timer timer_stage2;

  // options
  const bool use_schur_jacobi_precond =
      options_.preconditioner_type ==
      SolverOptions::PreconditionerType::SCHUR_JACOBI;

  // dampen poses
  // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
  // stage2, since the diagonal is already added automatically there!
  lqr_->set_pose_damping(lambda);

  // RHS 'b' of normal equations
  VecX b;

  // compute "stage 2"
  if (!options_.staged_execution) {
    Timer timer;

    // scale pose jacobians only on the first inner iteration
    if (new_linearization_point_) {
      lqr_->scale_Jp_cols(pose_jacobian_scaling_);
      IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
    }

    // dampen landmarks
    lqr_->set_landmark_damping(lambda);
    IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

    // compute preconditioner
    if (use_schur_jacobi_precond) {
      precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
      IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
          timer.reset();
    }

    b = lqr_->get_Q2TJp_T_Q2Tr();
    IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
  } else {
    lqr_->get_stage2(
        lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
        use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
  }

  IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

  // ////////////////////////////////////////////////////////////////////////
  // Solving:
  // - invert preconditioner
  // - run PCG
  // ////////////////////////////////////////////////////////////////////////

  // create and invert the proconditioner
  std::unique_ptr<Preconditioner<Scalar>> precond;
  {
    Timer timer;
    const int num_cams = bal_problem_.num_cameras();
    const int pose_size = lqr_->POSE_SIZE;
    if (options_.preconditioner_type ==
        SolverOptions::PreconditionerType::JACOBI) {
      // jacobi preconditioner doesn't have damping diagonal yet, and isn't
      // scaled yet

      // TODO: scaling the preconditioner here after the JTJ multiplication is
      // a quickfix; ideally we want to scale before multiplying JTJ, but it
      // doesn't fit into the current staged execution pipeline; if we
      // switch to computing the scale just once in the first outer
      // iteration, we can compute it outside the stage1 and then have it
      // ready to be used during stage1;

      if (new_linearization_point_) {
        scale_jacobians(precond_blocks_, num_cams, pose_size,
                        pose_jacobian_scaling_);
      }

      // TODO: don't forget to update this damping diagonal for the
      // preconditioner when switching in the future to damping diagonal != Ones
      // in LinearizationQR
      VecX precond_damping_diagonal =
          VecX::Constant(num_cams * pose_size, lambda);

      precond.reset(new BlockDiagonalPreconditioner<Scalar>(
          num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
    } else if (options_.preconditioner_type ==
               SolverOptions::PreconditionerType::SCHUR_JACOBI) {
      // jacobian scaling and damping diagonal is already applied
      precond.reset(new BlockDiagonalPreconditioner<Scalar>(
          num_cams, pose_size, precond_blocks_, nullptr));
    } else {
      LOG(FATAL) << "unreachable";
    }
    // Note: We add the time here, b/c part of computing the preconditioner is
    // already done in stage 1 (and logged if run in "unstaged" mode).
    IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
        timer.reset();
  }

  // run pcg
  VecX inc = VecX::Zero(lqr_->num_cols_reduced());
  {
    Timer timer;
    typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
        Base::pcg(*lqr_, b, std::move(precond), inc);
    IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
    IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
    IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
    IF_SET(it_summary_)->linear_solver_type = "bal_qr";
    IF_SET(summary_)->num_linear_solves += 1;
  }

  // if we backtrack, we don't need to rescale jacobians / preconditioners in
  // the next `solve` call
  new_linearization_point_ = false;

  return inc;
}

    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_joint(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_joint_RpOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }



    template <class Scalar_>
    typename LinearizorQR<Scalar_>::VecX LinearizorQR<Scalar_>::solve_refine(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // - compute RHS b
        // - compute SCHUR_JACOBI preconditioner
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // options
        const bool use_schur_jacobi_precond =
                options_.preconditioner_type ==
                SolverOptions::PreconditionerType::SCHUR_JACOBI;

        // dampen poses
        // NOTE: we need to do this before computing SCHUR_JACOBI preconditioner in
        // stage2, since the diagonal is already added automatically there!
        lqr_->set_pose_damping(lambda);

        // RHS 'b' of normal equations
        VecX b;

        // compute "stage 2"
        if (!options_.staged_execution) {
            Timer timer;

            // scale pose jacobians only on the first inner iteration
            if (new_linearization_point_) {
                lqr_->scale_Jp_cols(pose_jacobian_scaling_);
                IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            }

            // dampen landmarks
            lqr_->set_landmark_damping(lambda);
            IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

            // compute preconditioner
            if (use_schur_jacobi_precond) {
                precond_blocks_ = lqr_->get_Q2TJp_T_Q2TJp_blockdiag();
                IF_SET(it_summary_)->compute_preconditioner_time_in_seconds =
                                           timer.reset();
            }

            b = lqr_->get_Q2TJp_T_Q2Tr();
            IF_SET(it_summary_)->compute_gradient_time_in_seconds = timer.reset();
        } else {
            lqr_->get_stage2(
                    lambda, new_linearization_point_ ? &pose_jacobian_scaling_ : nullptr,
                    use_schur_jacobi_precond ? &precond_blocks_ : nullptr, b);
        }

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // create and invert the proconditioner
        std::unique_ptr<Preconditioner<Scalar>> precond;
        {
            Timer timer;
            const int num_cams = bal_problem_.num_cameras();
            const int pose_size = lqr_->POSE_SIZE;
            if (options_.preconditioner_type ==
                SolverOptions::PreconditionerType::JACOBI) {
                // jacobi preconditioner doesn't have damping diagonal yet, and isn't
                // scaled yet

                // TODO: scaling the preconditioner here after the JTJ multiplication is
                // a quickfix; ideally we want to scale before multiplying JTJ, but it
                // doesn't fit into the current staged execution pipeline; if we
                // switch to computing the scale just once in the first outer
                // iteration, we can compute it outside the stage1 and then have it
                // ready to be used during stage1;

                if (new_linearization_point_) {
                    scale_jacobians(precond_blocks_, num_cams, pose_size,
                                    pose_jacobian_scaling_);
                }

                // TODO: don't forget to update this damping diagonal for the
                // preconditioner when switching in the future to damping diagonal != Ones
                // in LinearizationQR
                VecX precond_damping_diagonal =
                        VecX::Constant(num_cams * pose_size, lambda);

                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, &precond_damping_diagonal));
            } else if (options_.preconditioner_type ==
                       SolverOptions::PreconditionerType::SCHUR_JACOBI) {
                // jacobian scaling and damping diagonal is already applied
                precond.reset(new BlockDiagonalPreconditioner<Scalar>(
                        num_cams, pose_size, precond_blocks_, nullptr));
            } else {
                LOG(FATAL) << "unreachable";
            }
            // Note: We add the time here, b/c part of computing the preconditioner is
            // already done in stage 1 (and logged if run in "unstaged" mode).
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds +=
                                       timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lqr_->num_cols_reduced());
        {
            Timer timer;
            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(*lqr_, b, std::move(precond), inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_qr";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

template <class Scalar_>
Scalar_ LinearizorQR<Scalar_>::apply(VecX&& inc) {
  // backsubstitue landmarks and compute model cost difference
  Timer timer;
  Scalar l_diff = lqr_->back_substitute(inc);
  IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();

  // unscale pose increments
  inc.array() *= pose_jacobian_scaling_.array();

  // update cameras
  for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
    bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
    bal_problem_.cameras()[i].apply_inc_intrinsics(
        inc.template segment<3>(i * 9 + 6));
  }
  IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

  return l_diff;
}

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::apply_joint(VecX&& inc) {
        // backsubstitue landmarks and compute model cost difference
        Timer timer;
        Scalar l_diff = lqr_->back_substitute(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }
        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorQR<Scalar_>::apply_joint_RpOSE(VecX&& inc) {
        // backsubstitue landmarks and compute model cost difference
        Timer timer;
        Scalar l_diff = lqr_->back_substitute(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }
        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

#ifdef ROOTBA_INSTANTIATIONS_FLOAT
template class LinearizorQR<float>;
#endif

#ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template class LinearizorQR<double>;
#endif

}  // namespace rootba
