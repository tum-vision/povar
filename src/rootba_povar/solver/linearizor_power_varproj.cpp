//
// Created by Simon Weber on 10/07/2023.
//

#include "linearizor_power_varproj.hpp"
#include "rootba_povar/cg/conjugate_gradient.hpp"
#include "rootba_povar/cg/preconditioner.hpp"
#include "rootba_povar/sc/linearization_power_varproj.hpp"
#include "rootba_povar/util/time_utils.hpp"

#include "rootba_povar/cg/block_sparse_matrix.hpp"
#include <Eigen/Dense>


// helper to deal with summary_ and it_summary_ pointers
#define IF_SET(POINTER_VAR) \
  if (POINTER_VAR) POINTER_VAR

namespace rootba_povar {

    template <class Scalar_>
    LinearizorPowerVarproj<Scalar_>::LinearizorPowerVarproj(BalProblem<Scalar>& bal_problem,
                                                  const SolverOptions& options,
                                                  SolverSummary* summary)
            : LinearizorBase<Scalar>(bal_problem, options, summary) {
        // set options
        typename LinearizationPowerVarproj<Scalar, 12>::Options lsc_options;
        lsc_options.sc_options.use_valid_projections_only =
                options_.use_projection_validity_check();
        lsc_options.sc_options.jacobi_scaling_eps =
                Base::get_effective_jacobi_scaling_epsilon();
        lsc_options.sc_options.residual_options = options_.residual;
        lsc_options.power_sc_iterations = options_.power_sc_iterations;

        // create linearization object
        lsc_ = std::make_unique<LinearizationPowerVarproj<Scalar, 12>>(bal_problem,
                                                                       lsc_options);
    }

    template <class Scalar_>
    LinearizorPowerVarproj<Scalar_>::~LinearizorPowerVarproj() = default;


    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_pOSE(Scalar_ alpha) {
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
        lsc_->scale_Jl_cols_pOSE();
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
    void LinearizorPowerVarproj<Scalar_>::linearize_projective_space_homogeneous() {
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_joint(
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
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        VecX b_p;
        {
            Timer timer;
            lsc_->prepare_Hb_joint(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }

        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_joint(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve(const SolverOptions& solver_options,
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
            lsc_->scale_Jp_cols_pOSE(pose_jacobian_scaling_pOSE_);
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
        }

        if (solver_options.solver_type_step_1 == SolverOptions::SolverType::POWER_SCHUR_COMPLEMENT) {
            // dampen landmarks
            lsc_->set_landmark_damping(lambda);
        }
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
            if (solver_options.solver_type_step_1 == SolverOptions::SolverType::POWER_SCHUR_COMPLEMENT) {
                lsc_->prepare_Hb_pOSE_poBA(b_p);
            }
            else if (solver_options.solver_type_step_1 == SolverOptions::SolverType::POWER_VARPROJ) {
                lsc_->prepare_Hb_pOSE(b_p);
            }
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_pOSE(b_p, inc, pso);
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
    Scalar_ LinearizorPowerVarproj<Scalar_>::apply(const SolverOptions& solver_options, Scalar_ alpha, VecX&& inc) {
        Timer timer;
        Scalar l_diff;
        // unscale pose increments
        if (solver_options.solver_type_step_1 == SolverOptions::SolverType::POWER_VARPROJ) {
            inc.array() *= pose_jacobian_scaling_pOSE_.array();
            for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
                bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 12));
            }
            inc.array() *= pose_jacobian_scaling_pOSE_.array().inverse();
            l_diff = lsc_->back_substitute_pOSE(alpha, inc); // in line with Supp. of "Revisiting the Variable Projection Method for Separable Nonlinear Least Squares Problems" (Hong et al., CVPR 2017)
            IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();
            IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();
        }
        else if (solver_options.solver_type_step_1 == SolverOptions::SolverType::POWER_SCHUR_COMPLEMENT) {
            l_diff = lsc_->back_substitute_poBA(inc);
            IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();
            // unscale pose increments
            inc.array() *= pose_jacobian_scaling_pOSE_.array();
            // update cameras
            for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
                bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 12));
            }
            IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();
        }

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::apply_joint(VecX&& inc) {
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
    template class LinearizorPowerVarproj<float>;
#endif

#ifdef ROOTBA_INSTANTIATIONS_DOUBLE
    template class LinearizorPowerVarproj<double>;
#endif

}  // namespace rootba_povar
