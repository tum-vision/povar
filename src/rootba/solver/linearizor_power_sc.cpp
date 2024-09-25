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
#include "rootba/solver/linearizor_power_sc.hpp"

#include "rootba/cg/conjugate_gradient.hpp"
#include "rootba/cg/preconditioner.hpp"
#include "rootba/sc/linearization_power_sc.hpp"
#include "rootba/util/time_utils.hpp"

// helper to deal with summary_ and it_summary_ pointers
#define IF_SET(POINTER_VAR) \
  if (POINTER_VAR) POINTER_VAR

namespace rootba {

template <class Scalar_>
LinearizorPowerSC<Scalar_>::LinearizorPowerSC(BalProblem<Scalar>& bal_problem,
                                              const SolverOptions& options,
                                              SolverSummary* summary)
    : LinearizorBase<Scalar>(bal_problem, options, summary) {
  // set options
  //typename LinearizationPowerSC<Scalar, 9>::Options lsc_options;
    //typename LinearizationPowerSC<Scalar, 14>::Options lsc_options;
    typename LinearizationPowerSC<Scalar, 11>::Options lsc_options;
  lsc_options.sc_options.use_householder =
      options_.use_householder_marginalization;
  lsc_options.sc_options.use_valid_projections_only =
      options_.use_projection_validity_check();
  lsc_options.sc_options.jacobi_scaling_eps =
      Base::get_effective_jacobi_scaling_epsilon();
  lsc_options.sc_options.residual_options = options_.residual;
  lsc_options.power_sc_iterations = options_.power_sc_iterations;

  // create linearization object
  //lsc_ = std::make_unique<LinearizationPowerSC<Scalar, 9>>(bal_problem,
  //                                                         lsc_options);
    //lsc_ = std::make_unique<LinearizationPowerSC<Scalar, 14>>(bal_problem,
    //                                                         lsc_options);
    lsc_ = std::make_unique<LinearizationPowerSC<Scalar, 11>>(bal_problem,
                                                              lsc_options);
}

template <class Scalar_>
LinearizorPowerSC<Scalar_>::~LinearizorPowerSC() = default;

template <class Scalar_>
void LinearizorPowerSC<Scalar_>::linearize() {
  // ////////////////////////////////////////////////////////////////////////
  // Stage 1: outside LM solver inner-loop
  // - linearization
  // - scale landmark Jacobians
  // - compute pose Jacobian scale
  // ////////////////////////////////////////////////////////////////////////
  Timer timer_stage1;

  VecX pose_damping_diagonal2;

  // TODO: would a staged version make sense here (consider stage 1 and 2)?

  Timer timer;
  CHECK(lsc_->linearize_problem())
      << "did not expect numerical failure during linearization";
  IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

  pose_damping_diagonal2 = lsc_->get_Jp_diag2();
  lsc_->scale_Jl_cols();
  IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

  // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
  // avoid accidental use of outdated info?

  // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
  // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
  pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                            pose_damping_diagonal2.array().sqrt())
                               .inverse();

  IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
  IF_SET(summary_)->num_jacobian_evaluations += 1;

  new_linearization_point_ = true;
}

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_affine_space(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_projective_space(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_pOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_RpOSE(double alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_RpOSE_ML(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_RpOSE_refinement(double alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_pOSE_poBA(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_expOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_pOSE_rOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_rOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_pOSE_riemannian_manifold(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_pOSE_homogeneous(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form_projective_space_homogeneous(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }
    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_pOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_RpOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_RpOSE_ML(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }



    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_pOSE_poBA(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_metric_upgrade(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_direct_pOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }
    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_direct_RpOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_direct_expOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_expOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_pOSE_rOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_rOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_pOSE_riemannian_manifold(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_pOSE_homogeneous(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::non_linear_varpro_landmark(Scalar lambda) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen

    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::closed_form_projective_space_homogeneous_nonlinear_initialization() {

    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold() {

    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::apply_pose_update(VecX&& inc) {

    }


    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::apply_pose_update_riemannian_manifold(VecX&& inc) {

    }


    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_affine_space(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_projective_space(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_projective_space_homogeneous(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_projective_space_homogeneous_riemannian_manifold(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        //lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_pOSE(int alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_RpOSE(double alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::radial_estimate() {

    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_RpOSE_ML() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }


    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_RpOSE_refinement(double alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }


    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_metric_upgrade() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_metric_upgrade_v2() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::estimate_alpha(VecX&& inc, Scalar lambda) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::update_y_tilde_expose() {

    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::initialize_y_tilde_expose() {

    }


    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::rpose_new_equilibrium() {

    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_expOSE(int alpha, bool init) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::compute_plane_linearly() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_pOSE_rOSE(int alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }


    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_rOSE(int alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_pOSE_homogeneous(int alpha) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }


    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_affine_space() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_projective_space() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_projective_space_homogeneous() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_projective_space_homogeneous_RpOSE() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_projective_space_homogeneous_lm_landmark() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }



    template <class Scalar_>
    void LinearizorPowerSC<Scalar_>::linearize_refine() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        CHECK(lsc_->linearize_problem())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2();
        lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

template <class Scalar_>
typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve(
    Scalar lambda, Scalar relative_error_change) {
  // ////////////////////////////////////////////////////////////////////////
  // Stage 2: inside LM solver inner-loop
  // - scale pose Jacobians (1st inner it)
  // - dampen
  // ////////////////////////////////////////////////////////////////////////
  Timer timer_stage2;

  // dampen poses
  lsc_->set_pose_damping(lambda);

  Timer timer;

  // scale pose jacobians only on the first inner iteration
  if (new_linearization_point_) {
    lsc_->scale_Jp_cols(pose_jacobian_scaling_);
    IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
  }

  // dampen landmarks
  lsc_->set_landmark_damping(lambda);
  IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

  IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

  // ////////////////////////////////////////////////////////////////////////
  // Solving:
  // - marginalize landmarks (SC)
  // - run PCG
  // ////////////////////////////////////////////////////////////////////////

  // compute Schur complement
  VecX b_p;
  {
    Timer timer;
    lsc_->prepare_Hb(b_p);
    IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
  }

  typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
  pso.q_tolerance = options_.eta;
  pso.r_tolerance = options_.r_tolerance;
  //pso.r_tolerance = 0.01;

  VecX inc;
  {
    Timer timer;
    const auto summary = lsc_->solve(b_p, inc, pso);
    IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
    IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
    IF_SET(it_summary_)->linear_solver_message = summary.message;
    IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
    IF_SET(summary_)->num_linear_solves += 1;
  }

  // if we backtrack, we don't need to rescale jacobians / preconditioners in
  // the next `solve` call
  new_linearization_point_ = false;

  return inc;
}

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_joint(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_joint_RpOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorPowerSC<Scalar_>::VecX LinearizorPowerSC<Scalar_>::solve_refine(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        lsc_->set_landmark_damping(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve(b_p, inc, pso);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_type = "bal_power_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerSC<Scalar_>::closed_form(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 6));
        }


        Scalar l_diff;// = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

template <class Scalar_>
Scalar_ LinearizorPowerSC<Scalar_>::apply(VecX&& inc) {
  // backsubstitue landmarks and compute model cost difference
  Timer timer;
  Scalar l_diff = lsc_->back_substitute(inc);
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
    Scalar_ LinearizorPowerSC<Scalar_>::apply_joint(VecX&& inc) {
        // backsubstitue landmarks and compute model cost difference
        Timer timer;
        Scalar l_diff = lsc_->back_substitute(inc);
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
    Scalar_ LinearizorPowerSC<Scalar_>::apply_joint_RpOSE(VecX&& inc) {
        // backsubstitue landmarks and compute model cost difference
        Timer timer;
        Scalar l_diff = lsc_->back_substitute(inc);
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
template class LinearizorPowerSC<float>;
#endif

#ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template class LinearizorPowerSC<double>;
#endif

}  // namespace rootba
