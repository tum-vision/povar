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
#include "rootba/solver/linearizor_sc.hpp"

#include "rootba/cg/conjugate_gradient.hpp"
#include "rootba/cg/preconditioner.hpp"
#include "rootba/sc/linearization_sc.hpp"
#include "rootba/util/time_utils.hpp"

#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

// helper to deal with summary_ and it_summary_ pointers
#define IF_SET(POINTER_VAR) \
  if (POINTER_VAR) POINTER_VAR

namespace rootba {

template <class Scalar_>
LinearizorSC<Scalar_>::LinearizorSC(BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options,
                                    SolverSummary* summary)
    : LinearizorBase<Scalar>(bal_problem, options, summary) {
  // set options
  //typename LinearizationSC<Scalar, 9>::Options lsc_options;
    //typename LinearizationSC<Scalar, 15>::Options lsc_options;
    typename LinearizationSC<Scalar, 11>::Options lsc_options;


    lsc_options.use_householder = options_.use_householder_marginalization;
  lsc_options.use_valid_projections_only =
      options_.use_projection_validity_check();
  lsc_options.jacobi_scaling_eps = Base::get_effective_jacobi_scaling_epsilon();
  lsc_options.residual_options = options_.residual;
  lsc_options.reduction_alg = options_.reduction_alg;

  // create linearization object
  //lsc_ = std::make_unique<LinearizationSC<Scalar, 9>>(bal_problem, lsc_options);
    //lsc_ = std::make_unique<LinearizationSC<Scalar, 15>>(bal_problem, lsc_options);
    lsc_ = std::make_unique<LinearizationSC<Scalar, 11>>(bal_problem, lsc_options);
}

template <class Scalar_>
LinearizorSC<Scalar_>::~LinearizorSC() = default;

template <class Scalar_>
void LinearizorSC<Scalar_>::linearize() {
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

  pose_damping_diagonal2 = lsc_->get_Jp_diag2_golden();
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
    void LinearizorSC<Scalar_>::linearize_refine() {
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
    Scalar_ LinearizorSC<Scalar_>::closed_form(VecX&& inc) {
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
    Scalar_ LinearizorSC<Scalar_>::closed_form_pOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));

        }
        inc.array() *= pose_jacobian_scaling_pOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_pOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorSC<Scalar_>::closed_form_RpOSE(double alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));

        }
        inc.array() *= pose_jacobian_scaling_pOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_pOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorSC<Scalar_>::closed_form_RpOSE_ML(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));

        }
        Scalar alpha = 1.0;
        inc.array() *= pose_jacobian_scaling_pOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_pOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorSC<Scalar_>::closed_form_RpOSE_refinement(double alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));

        }
        inc.array() *= pose_jacobian_scaling_pOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_pOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorSC<Scalar_>::closed_form_pOSE_poBA(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));

        }
        inc.array() *= pose_jacobian_scaling_pOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_pOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorSC<Scalar_>::closed_form_expOSE(int alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_expOSE_.array();
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_expOSE(inc.template segment<12>(i * 15));
        }
        inc.array() *= pose_jacobian_scaling_expOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_expOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorSC<Scalar_>::closed_form_pOSE_rOSE(int alpha, VecX&& inc) {
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
    Scalar_ LinearizorSC<Scalar_>::closed_form_rOSE(int alpha, VecX&& inc) {
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
    Scalar_ LinearizorSC<Scalar_>::closed_form_pOSE_riemannian_manifold(int alpha, VecX&& inc) {
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
    Scalar_ LinearizorSC<Scalar_>::closed_form_pOSE_homogeneous(int alpha, VecX&& inc) {
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
    Scalar_ LinearizorSC<Scalar_>::closed_form_affine_space(VecX&& inc) {
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
    Scalar_ LinearizorSC<Scalar_>::closed_form_projective_space(VecX&& inc) {
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
    Scalar_ LinearizorSC<Scalar_>::closed_form_projective_space_homogeneous(VecX&& inc) {
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
    void LinearizorSC<Scalar_>::update_y_tilde_expose(){
        lsc_->update_y_tilde_expose();
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::initialize_y_tilde_expose(){
        lsc_->initialize_y_tilde_expose();
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::rpose_new_equilibrium(){
        lsc_->initialize_y_tilde_expose();
    }




    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_pOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping_pOSE(lambda);
        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_pOSE(pose_jacobian_scaling_pOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_pOSE(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_pOSE());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_RpOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping_pOSE(lambda);
        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_pOSE(pose_jacobian_scaling_pOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_pOSE(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_pOSE());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_RpOSE_ML(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping_pOSE(lambda);
        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_pOSE(pose_jacobian_scaling_pOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_pOSE(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_pOSE());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_pOSE_poBA(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping_pOSE(lambda);
        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_pOSE(pose_jacobian_scaling_pOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_pOSE(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_pOSE());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_metric_upgrade(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
        using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
        using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        Timer timer_stage2;

        Timer timer;


        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        // compute Schur complement
        int num_cam = bal_problem_.cameras().size();
        int pose_time_num_cam = 9 * num_cam;
        Vec3 inc;
        MatX I_JTJ;
        I_JTJ.resize(9*num_cam,9*num_cam);
        I_JTJ.setZero();
        const auto J = storage_metric_.leftCols(3);
        const auto r = storage_metric_.middleCols(12,1);

        for (int i = 0; i < num_cam; i++) {
            I_JTJ.template block<9,9>(9*i,9*i) = MatX::Identity(9,9) - storage_metric_.template block<9,9>(9 * i,3);
        }
        //Vec3 b = J.transpose() * I_JTJ * r;
        Vec3 b = J.transpose() *I_JTJ* r;
        Mat3 JtJ = J.transpose() *I_JTJ* J;

        Mat3 D;
        D.setZero();
        D(0,0) = JtJ(0,0);
        D(1,1) = JtJ(1,1);
        D(2,2) = JtJ(2,2);

        //inc = - (JtJ + lambda *D).inverse() * b;
        inc = - (JtJ + lambda * MatX::Identity(3,3)).inverse() * b;

        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_direct_pOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping_pOSE(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_pOSE(pose_jacobian_scaling_pOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_pOSE(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;
            const auto summary = lsc_->solve_direct_pOSE(H_pp, b_p, inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_direct_RpOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping_pOSE(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_pOSE(pose_jacobian_scaling_pOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_pOSE(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;
            const auto summary = lsc_->solve_direct_pOSE(H_pp, b_p, inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_direct_expOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping_expOSE(lambda);

        Timer timer;
        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_expOSE(pose_jacobian_scaling_expOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }
        // dampen landmarks
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////
        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_expOSE(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_expOSE());
        {
            Timer timer;
            const auto summary = lsc_->solve_direct_expOSE(H_pp, b_p, inc);
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_expOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen poses
        lsc_->set_pose_damping_expOSE(lambda);

        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_expOSE(pose_jacobian_scaling_expOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        // dampen landmarks
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_expOSE(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_expOSE());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_pOSE_rOSE(
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
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_rOSE(
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
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_pOSE_riemannian_manifold(
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
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_pOSE_homogeneous(
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
        // @Simon: for VarProj, we set lambda_landmark = 0
        //lsc_->set_landmark_damping(lambda);
        //IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        // create and invert the proconditioner
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }



    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_affine_space(
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
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::milliseconds;
        // create and invert the proconditioner
        auto t1 = high_resolution_clock::now();
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }
        auto t2 = high_resolution_clock::now();
        auto ms1_int = duration_cast<milliseconds>(t2 - t1);

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_projective_space(
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
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::milliseconds;
        // create and invert the proconditioner
        auto t1 = high_resolution_clock::now();
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }
        auto t2 = high_resolution_clock::now();
        auto ms1_int = duration_cast<milliseconds>(t2 - t1);

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_projective_space_homogeneous(
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
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::milliseconds;
        // create and invert the proconditioner
        auto t1 = high_resolution_clock::now();
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }
        auto t2 = high_resolution_clock::now();
        auto ms1_int = duration_cast<milliseconds>(t2 - t1);

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_projective_space_homogeneous_riemannian_manifold(
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
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::milliseconds;
        // create and invert the proconditioner
        auto t1 = high_resolution_clock::now();
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }
        auto t2 = high_resolution_clock::now();
        auto ms1_int = duration_cast<milliseconds>(t2 - t1);

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::radial_estimate() {

    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_pOSE(int alpha) {
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
        CHECK(lsc_->linearize_problem_pOSE(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_pOSE();
        //lsc_->scale_Jl_cols_pOSE(); //@Simon: first, we try without scaled Jl (see if it is not an issue for preconditioners)
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();
        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_pOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();
        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_RpOSE(double alpha) {
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
        CHECK(lsc_->linearize_problem_pOSE(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_pOSE();
        //lsc_->scale_Jl_cols_pOSE(); //@Simon: first, we try without scaled Jl (see if it is not an issue for preconditioners)
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();
        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_pOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();
        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_RpOSE_ML() {
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
        Scalar alpha = 1.0;
        CHECK(lsc_->linearize_problem_pOSE(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_pOSE();
        //lsc_->scale_Jl_cols_pOSE(); //@Simon: first, we try without scaled Jl (see if it is not an issue for preconditioners)
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();
        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_pOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();
        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::estimate_alpha(VecX&& inc, Scalar lambda) {
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
        Scalar alpha = 1.0;
        CHECK(lsc_->linearize_problem_pOSE(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_pOSE();
        //lsc_->scale_Jl_cols_pOSE(); //@Simon: first, we try without scaled Jl (see if it is not an issue for preconditioners)
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();
        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_pOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();
        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_RpOSE_refinement(double alpha) {
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
        CHECK(lsc_->linearize_problem_pOSE(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_pOSE();
        //lsc_->scale_Jl_cols_pOSE(); //@Simon: first, we try without scaled Jl (see if it is not an issue for preconditioners)
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();
        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_pOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();
        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::compute_plane_linearly() {
        using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        using Mat64 = Eigen::Matrix<Scalar, 6, 4>;

        using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
        using Vec5 = Eigen::Matrix<Scalar, 5, 1>;
        using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
        MatX A_;
        A_.resize(6 * bal_problem_.cameras().size(), 4);
        VecX b_;
        b_.resize(6 * bal_problem_.cameras().size());
        for (int i = 0; i < bal_problem_.cameras().size(); i++) {
            Mat64 A;
            Vec6 b;
            A.setZero();
            b.setZero();
            Vec5 tmp = alpha(i,0,1);
            A.row(0) = (tmp.template head<4>()).transpose();
            b(0) = tmp(4);
            tmp = alpha(i,0,2);
            A.row(1) = tmp.template head<4>().transpose();
            b(1) = tmp(4);

            tmp = alpha(i,1,2);
            A.row(2) = tmp.template head<4>().transpose();
            b(2) = tmp(4);

            Vec5 tmp2 = alpha(i,1,1);
            Vec5 tmp1 = alpha(i,0,0);
            A.row(3) = (tmp1 - tmp2).template head<4>().transpose();
            b(3) = tmp1(4) - tmp2(4);

            Vec5 tmp3 = alpha(i,2,2);
            A.row(4) = (tmp1 - tmp3).template head<4>().transpose();
            b(4) = tmp1(4) - tmp3(4);
            A.row(5) = (tmp2 - tmp3).template head<4>().transpose();
            b(5) = tmp2(4) - tmp3(4);

            A_.template block<6,4>(6*i,0) = A;
            b_.template segment<6>(6*i) = b;
        }
        Vec4 plan_homogeneous = -A_.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_);
        bal_problem_.h_euclidean().plan_infinity = plan_homogeneous.template head<3>();
        //bal_problem_.h_euclidean().plan_infinity = -A_.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_);
        //bal_problem_.h_euclidean().plan_infinity = - A_.inverse() * b_;
    }

    template <class Scalar_>
    Eigen::Matrix<Scalar_, 5, 1> LinearizorSC<Scalar_>::alpha(int num_cameras, int i, int j) {
        Eigen::Matrix<Scalar, 5, 1> out;
        out.setZero();
        const auto& cam = bal_problem_.cameras().at(num_cameras);
        out(0) = cam.space_matrix_intrinsics(i,3) * cam.space_matrix_intrinsics(j,0) + cam.space_matrix_intrinsics(j,3) * cam.space_matrix_intrinsics(i,0);
        out(1) = cam.space_matrix_intrinsics(i,3) * cam.space_matrix_intrinsics(j,1) + cam.space_matrix_intrinsics(j,3) * cam.space_matrix_intrinsics(i,1);
        out(2) = cam.space_matrix_intrinsics(i,3) * cam.space_matrix_intrinsics(j,2) + cam.space_matrix_intrinsics(j,3) * cam.space_matrix_intrinsics(i,2);
        out(3) = cam.space_matrix_intrinsics(i,3) * cam.space_matrix_intrinsics(j,3);
        out(4) = Scalar(cam.space_matrix_intrinsics.row(i) *  cam.space_matrix_intrinsics.row(j).transpose());
        return out;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_metric_upgrade() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;
        VecX pose_damping_diagonal2;
        int alpha;
        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        storage_metric_.setZero(9*bal_problem_.cameras().size(),13);
        for (int i = 0; i < bal_problem_.cameras().size(); i++) {
            typename BalBundleAdjustmentHelper<Scalar_>::VecR_metric res;
            typename BalBundleAdjustmentHelper<Scalar_>::MatRH JH;
            typename BalBundleAdjustmentHelper<Scalar_>::MatR_alpha Jalpha;

            auto const cam = bal_problem_.cameras().at(i);
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade(
                    cam.PH, cam.PHHP, cam.space_matrix_intrinsics, cam.alpha, cam.intrinsics,true, res , &JH, &Jalpha);

            storage_metric_.template block<9,3>(9*i,0) = JH;
            storage_metric_.template block<9,9>(9*i,3) = Jalpha;
            storage_metric_.template block <9,1>(9*i,12) = res;
            //storage_metric_.template block <3,1>(9*i+3,12) = res.col(1);
            //storage_metric_.template block <3,1>(9*i+6,12) = res.col(2);

            //storage_metric_(9*i,12) = res(0.0);
            //storage_metric_(9*i,13) = res(1.0);
            //storage_metric_(9*i,14) = (res(2.0));
            //storage_metric_(9*i,15) = (res(0.1));
            //storage_metric_(9*i,16) = (res(1.1));
            //storage_metric_(9*i,17) = (res(2.1));
            //storage_metric_(9*i,18) = (res(0.2));
            //storage_metric_(9*i,19) = (res(1.2));
            //storage_metric_(9*i,20) = (res(2.2));
        }

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_metric_upgrade_v2() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;
        VecX pose_damping_diagonal2;
        int alpha;
        // TODO: would a staged version make sense here (consider stage 1 and 2)?

        Timer timer;
        storage_metric_.setZero(9*bal_problem_.cameras().size(),13);
        for (int i = 0; i < bal_problem_.cameras().size(); i++) {
            typename BalBundleAdjustmentHelper<Scalar_>::VecR_metric res;
            typename BalBundleAdjustmentHelper<Scalar_>::MatRH JH;
            typename BalBundleAdjustmentHelper<Scalar_>::MatR_alpha Jalpha;

            auto const cam = bal_problem_.cameras().at(i);
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade(
                    cam.PH, cam.PHHP, cam.space_matrix_intrinsics, cam.alpha, cam.intrinsics,true, res , &JH, &Jalpha);

            storage_metric_.template block<9,3>(9*i,0) = JH;
            storage_metric_.template block<9,9>(9*i,3) = Jalpha;
            storage_metric_.template block <9,1>(9*i,12) = res;
            //storage_metric_.template block <3,1>(9*i+3,12) = res.col(1);
            //storage_metric_.template block <3,1>(9*i+6,12) = res.col(2);

            //storage_metric_(9*i,12) = res(0.0);
            //storage_metric_(9*i,13) = res(1.0);
            //storage_metric_(9*i,14) = (res(2.0));
            //storage_metric_(9*i,15) = (res(0.1));
            //storage_metric_(9*i,16) = (res(1.1));
            //storage_metric_(9*i,17) = (res(2.1));
            //storage_metric_(9*i,18) = (res(0.2));
            //storage_metric_(9*i,19) = (res(1.2));
            //storage_metric_(9*i,20) = (res(2.2));
        }

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_expOSE(int alpha, bool init) {
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
        CHECK(lsc_->linearize_problem_expOSE(alpha, init))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_expOSE();
        //lsc_->scale_Jl_cols_pOSE(); //@Simon: first, we try without scaled Jl (see if it is not an issue for preconditioners)
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();
        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_expOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();
        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_pOSE_rOSE(int alpha) {
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
    void LinearizorSC<Scalar_>::linearize_rOSE(int alpha) {
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
    void LinearizorSC<Scalar_>::linearize_pOSE_homogeneous(int alpha) {
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
    void LinearizorSC<Scalar_>::closed_form_projective_space_homogeneous_nonlinear_initialization() {
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold() {
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::non_linear_varpro_landmark(Scalar lambda) {

    }


    template <class Scalar_>
    void LinearizorSC<Scalar_>::apply_pose_update(VecX&& inc) {

    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::apply_pose_update_riemannian_manifold(VecX&& inc) {

    }



    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_affine_space() {
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
    void LinearizorSC<Scalar_>::linearize_projective_space() {
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
    void LinearizorSC<Scalar_>::linearize_projective_space_homogeneous() {
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
        CHECK(lsc_->linearize_problem_projective_space_homogeneous())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2_projective_space();
        lsc_->scale_Jl_cols_homogeneous();
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
    void LinearizorSC<Scalar_>::linearize_projective_space_homogeneous_RpOSE() {
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
        CHECK(lsc_->linearize_problem_projective_space_homogeneous())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2_projective_space();
        lsc_->scale_Jl_cols_homogeneous();
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
    void LinearizorSC<Scalar_>::linearize_projective_space_homogeneous_lm_landmark() {
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
typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve(
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
  // - invert preconditioner
  // - run PCG
  // ////////////////////////////////////////////////////////////////////////

  // compute Schur complement
  const int num_cams = bal_problem_.num_cameras();
  const int pose_size = 9;
  BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
  VecX b_p;
  {
    Timer timer;
    lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
    IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
  }
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  // create and invert the proconditioner
  auto t1 = high_resolution_clock::now();
  CHECK(options_.preconditioner_type ==
        SolverOptions::PreconditionerType::SCHUR_JACOBI)
      << "not implemented";
  std::unique_ptr<Preconditioner<Scalar_>> precond;
  {
    Timer timer;
    precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
        num_cams, pose_size, H_pp.block_storage, nullptr));
    IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
  }
  auto t2 = high_resolution_clock::now();
  auto ms1_int = duration_cast<milliseconds>(t2 - t1);

  // run pcg
  VecX inc = VecX::Zero(lsc_->num_cols_reduced());
  {
    Timer timer;

    typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
        Base::pcg_golden(H_pp, b_p, std::move(precond), inc);

    IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
    IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
    IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
    IF_SET(it_summary_)->linear_solver_type = "bal_sc";
    IF_SET(summary_)->num_linear_solves += 1;
  }

  // if we backtrack, we don't need to rescale jacobians / preconditioners in
  // the next `solve` call
  new_linearization_point_ = false;

  return inc;
}

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_joint(
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
            lsc_->scale_Jp_cols_joint(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            lsc_->linearize_nullspace();
        }

        // dampen landmarks
        lsc_->set_landmark_damping_joint(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = 11; //lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_joint(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::milliseconds;
        // create and invert the proconditioner
        auto t1 = high_resolution_clock::now();
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }
        auto t2 = high_resolution_clock::now();
        auto ms1_int = duration_cast<milliseconds>(t2 - t1);

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_joint());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg_joint(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_joint_RpOSE(
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
            lsc_->scale_Jp_cols_joint(pose_jacobian_scaling_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            lsc_->linearize_nullspace();
        }

        // dampen landmarks
        lsc_->set_landmark_damping_joint(lambda);
        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = 11; //lsc_->POSE_SIZE;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_joint(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::milliseconds;
        // create and invert the proconditioner
        auto t1 = high_resolution_clock::now();
        CHECK(options_.preconditioner_type ==
              SolverOptions::PreconditionerType::SCHUR_JACOBI)
                        << "not implemented";
        std::unique_ptr<Preconditioner<Scalar_>> precond;
        {
            Timer timer;
            precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
                    num_cams, pose_size, H_pp.block_storage, nullptr));
            IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
        }
        auto t2 = high_resolution_clock::now();
        auto ms1_int = duration_cast<milliseconds>(t2 - t1);

        // run pcg
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_joint());
        {
            Timer timer;

            typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
                    Base::pcg_joint(H_pp, b_p, std::move(precond), inc);

            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
            IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
            IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
            IF_SET(it_summary_)->linear_solver_type = "bal_sc";
            IF_SET(summary_)->num_linear_solves += 1;
        }

        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }




    template <class Scalar_>
typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve_refine(
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
  // - invert preconditioner
  // - run PCG
  // ////////////////////////////////////////////////////////////////////////

  // compute Schur complement
  const int num_cams = bal_problem_.num_cameras();
  const int pose_size = lsc_->POSE_SIZE;
  BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
  VecX b_p;
  {
    Timer timer;
    lsc_->get_Hb(H_pp, b_p);  // in implicit: prepare_Hb(b_p)
    IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
  }
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  // create and invert the proconditioner
  auto t1 = high_resolution_clock::now();
  CHECK(options_.preconditioner_type ==
        SolverOptions::PreconditionerType::SCHUR_JACOBI)
      << "not implemented";
  std::unique_ptr<Preconditioner<Scalar_>> precond;
  {
    Timer timer;
    precond.reset(new BlockDiagonalPreconditioner<Scalar_>(
        num_cams, pose_size, H_pp.block_storage, nullptr));
    IF_SET(it_summary_)->compute_preconditioner_time_in_seconds = timer.reset();
  }
  auto t2 = high_resolution_clock::now();
  auto ms1_int = duration_cast<milliseconds>(t2 - t1);

  // run pcg
  VecX inc = VecX::Zero(lsc_->num_cols_reduced());
  {
    Timer timer;

    typename ConjugateGradientsSolver<Scalar>::Summary cg_summary =
        Base::pcg(H_pp, b_p, std::move(precond), inc);

    IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();
    IF_SET(it_summary_)->linear_solver_message = cg_summary.message;
    IF_SET(it_summary_)->linear_solver_iterations = cg_summary.num_iterations;
    IF_SET(it_summary_)->linear_solver_type = "bal_sc";
    IF_SET(summary_)->num_linear_solves += 1;
  }

  // if we backtrack, we don't need to rescale jacobians / preconditioners in
  // the next `solve` call
  new_linearization_point_ = false;

  return inc;
}


template <class Scalar_>
Scalar_ LinearizorSC<Scalar_>::apply(VecX&& inc) {
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
    Scalar_ LinearizorSC<Scalar_>::apply_joint(VecX&& inc) {
        // backsubstitue landmarks and compute model cost difference
        Timer timer;
        Scalar l_diff = lsc_->back_substitute_joint(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();

        // unscale pose increments
        //inc.array() *= pose_jacobian_scaling_.array();

        // update cameras
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            const auto& cam = bal_problem_.cameras().at(i);

            Eigen::Matrix<Scalar, 12, 1> camera_space_matrix;
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

            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD((camera_space_matrix).transpose());
            VecX inc_proj = Proj_pose * inc.template segment<11>(i*11);
            inc_proj.array() *= pose_jacobian_scaling_.array().template segment<12>(i*15);

            bal_problem_.cameras()[i].apply_inc_pose_projective_space(inc_proj);
            //inc_proj.array() *= pose_jacobian_scaling_.array().template segment<12>(i*15).inverse();
            //bal_problem_.cameras()[i].apply_inc_intrinsics(
            //        inc.template segment<3>(i * 9 + 6));
        }
        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorSC<Scalar_>::apply_joint_RpOSE(VecX&& inc) {
        // backsubstitue landmarks and compute model cost difference
        Timer timer;
        Scalar l_diff = lsc_->back_substitute_joint(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();

        // unscale pose increments
        //inc.array() *= pose_jacobian_scaling_.array();

        // update cameras
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            const auto& cam = bal_problem_.cameras().at(i);

            Eigen::Matrix<Scalar, 12, 1> camera_space_matrix;
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

            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD((camera_space_matrix).transpose());
            VecX inc_proj = Proj_pose * inc.template segment<11>(i*11);
            inc_proj.array() *= pose_jacobian_scaling_.array().template segment<12>(i*15);

            bal_problem_.cameras()[i].apply_inc_pose_projective_space(inc_proj);
            //inc_proj.array() *= pose_jacobian_scaling_.array().template segment<12>(i*15).inverse();
            //bal_problem_.cameras()[i].apply_inc_intrinsics(
            //        inc.template segment<3>(i * 9 + 6));
        }
        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }



#ifdef ROOTBA_INSTANTIATIONS_FLOAT
template class LinearizorSC<float>;
#endif

#ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template class LinearizorSC<double>;
#endif

}  // namespace rootba
