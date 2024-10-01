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
#include "rootba_povar/solver/linearizor_sc.hpp"

#include "rootba_povar/cg/conjugate_gradient.hpp"
#include "rootba_povar/cg/preconditioner.hpp"
#include "rootba_povar/sc/linearization_sc.hpp"
#include "rootba_povar/util/time_utils.hpp"

// helper to deal with summary_ and it_summary_ pointers
#define IF_SET(POINTER_VAR) \
  if (POINTER_VAR) POINTER_VAR

namespace rootba_povar {

template <class Scalar_>
LinearizorSC<Scalar_>::LinearizorSC(BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options,
                                    SolverSummary* summary)
    : LinearizorBase<Scalar>(bal_problem, options, summary) {
    typename LinearizationSC<Scalar, 12>::Options lsc_options;


  lsc_options.use_valid_projections_only =
      options_.use_projection_validity_check();
  lsc_options.jacobi_scaling_eps = Base::get_effective_jacobi_scaling_epsilon();
  lsc_options.residual_options = options_.residual;

  // create linearization object
    lsc_ = std::make_unique<LinearizationSC<Scalar, 12>>(bal_problem, lsc_options);
}

template <class Scalar_>
LinearizorSC<Scalar_>::~LinearizorSC() = default;

    template <class Scalar_>
    Scalar_ LinearizorSC<Scalar_>::apply(const SolverOptions& solver_options, Scalar_ alpha, VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_.array();

        // update cameras
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 12));

        }
        inc.array() *= pose_jacobian_scaling_pOSE_.array().inverse();
        Scalar l_diff = lsc_->back_substitute_pOSE(alpha, inc); // in line with Supp. of "Revisiting the Variable Projection Method for Separable Nonlinear Least Squares Problems" (Hong et al., CVPR 2017)
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    typename LinearizorSC<Scalar_>::VecX LinearizorSC<Scalar_>::solve(const SolverOptions& solver_options,
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

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - invert preconditioner
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        const int num_cams = bal_problem_.num_cameras();
        const int pose_size = 12;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_pOSE(H_pp, b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        VecX inc = VecX::Zero(lsc_->num_cols_reduced_pOSE());
        {
            Timer timer;
            if (solver_options.solver_type_step_1 == SolverOptions::SolverType::CHOLESKY) {
                {
                    const auto summary = lsc_->solve_direct_pOSE(H_pp, b_p, inc);
                    IF_SET(it_summary_)->linear_solver_type = "bal_sc";
                    IF_SET(it_summary_)->linear_solver_message = summary.message;
                    IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
                }
            }
            else if (solver_options.solver_type_step_1 == SolverOptions::SolverType::PCG) {
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
                {
                    typename ConjugateGradientsSolver<Scalar>::Summary summary =
                            Base::pcg(H_pp, b_p, std::move(precond), inc);
                    IF_SET(it_summary_)->linear_solver_type = "bal_sc";
                    IF_SET(it_summary_)->linear_solver_message = summary.message;
                    IF_SET(it_summary_)->linear_solver_iterations = summary.num_iterations;
                }
            }
            IF_SET(it_summary_)->solve_reduced_system_time_in_seconds = timer.elapsed();

            IF_SET(summary_)->num_linear_solves += 1;
        }


        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
        new_linearization_point_ = false;

        return inc;
    }

    template <class Scalar_>
    void LinearizorSC<Scalar_>::linearize_pOSE(Scalar_ alpha) {
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
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

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
    void LinearizorSC<Scalar_>::linearize_projective_space_homogeneous() {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 1: outside LM solver inner-loop
        // - linearization
        // - scale landmark Jacobians
        // - compute pose Jacobian scale
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage1;

        VecX pose_damping_diagonal2;

        Timer timer;
        CHECK(lsc_->linearize_problem_projective_space_homogeneous())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();

        pose_damping_diagonal2 = lsc_->get_Jp_diag2_projective_space();
        lsc_->scale_Jl_cols_homogeneous();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

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
        const int pose_size = 11;
        BlockSparseMatrix<Scalar_> H_pp(num_cams * pose_size, num_cams * pose_size);
        VecX b_p;
        {
            Timer timer;
            lsc_->get_Hb_joint(H_pp, b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        using std::chrono::duration;
        using std::chrono::duration_cast;
        using std::chrono::high_resolution_clock;
        using std::chrono::milliseconds;
        // create and invert the preconditioner
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
    Scalar_ LinearizorSC<Scalar_>::apply_joint(VecX&& inc) {
        // backsubstitue landmarks and compute model cost difference
        Timer timer;
        Scalar l_diff = lsc_->back_substitute_joint(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();

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
            inc_proj.array() *= pose_jacobian_scaling_.array().template segment<12>(i*12);

            bal_problem_.cameras()[i].apply_inc_pose_projective_space(inc_proj);
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

}  // namespace rootba_povar
