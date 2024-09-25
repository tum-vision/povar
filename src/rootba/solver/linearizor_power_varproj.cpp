//
// Created by Simon on 10/07/2023.
//

#include "linearizor_power_varproj.hpp"
#include "rootba/cg/conjugate_gradient.hpp"
#include "rootba/cg/preconditioner.hpp"
#include "rootba/sc/linearization_power_varproj.hpp"
#include "rootba/util/time_utils.hpp"

#include "rootba/cg/block_sparse_matrix.hpp"
#include <Eigen/Dense>


// helper to deal with summary_ and it_summary_ pointers
#define IF_SET(POINTER_VAR) \
  if (POINTER_VAR) POINTER_VAR

namespace rootba {

    template <class Scalar_>
    LinearizorPowerVarproj<Scalar_>::LinearizorPowerVarproj(BalProblem<Scalar>& bal_problem,
                                                  const SolverOptions& options,
                                                  SolverSummary* summary)
            : LinearizorBase<Scalar>(bal_problem, options, summary) {
        // set options
        //typename LinearizationPowerVarproj<Scalar, 9>::Options lsc_options;
        //typename LinearizationPowerVarproj<Scalar, 15>::Options lsc_options;
        typename LinearizationPowerVarproj<Scalar, 11>::Options lsc_options;
        lsc_options.sc_options.use_householder =
                options_.use_householder_marginalization;
        lsc_options.sc_options.use_valid_projections_only =
                options_.use_projection_validity_check();
        lsc_options.sc_options.jacobi_scaling_eps =
                Base::get_effective_jacobi_scaling_epsilon();
        lsc_options.sc_options.residual_options = options_.residual;
        lsc_options.power_sc_iterations = options_.power_sc_iterations;

        // create linearization object
        //lsc_ = std::make_unique<LinearizationPowerVarproj<Scalar, 9>>(bal_problem,
        //                                                         lsc_options);
        //lsc_ = std::make_unique<LinearizationPowerVarproj<Scalar, 15>>(bal_problem,
        //                                                              lsc_options);
        lsc_ = std::make_unique<LinearizationPowerVarproj<Scalar, 11>>(bal_problem,
                                                                       lsc_options);
    }

    template <class Scalar_>
    LinearizorPowerVarproj<Scalar_>::~LinearizorPowerVarproj() = default;

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize() {
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
    void LinearizorPowerVarproj<Scalar_>::compute_plane_linearly() {

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
    Eigen::Matrix<Scalar_, 5, 1> LinearizorPowerVarproj<Scalar_>::alpha(int num_cameras, int i, int j) {
        Eigen::Matrix<Scalar, 5, 1> out;
        out.setZero();
        const auto& cam = bal_problem_.cameras().at(num_cameras);
        out(0) = cam.space_matrix_intrinsics(i,3) * cam.space_matrix_intrinsics(j,0) + cam.space_matrix_intrinsics(j,3) * cam.space_matrix_intrinsics(i,0);
        out(1) = cam.space_matrix_intrinsics(i,3) * cam.space_matrix_intrinsics(j,1) + cam.space_matrix_intrinsics(j,3) * cam.space_matrix_intrinsics(i,1);
        out(2) = cam.space_matrix_intrinsics(i,3) * cam.space_matrix_intrinsics(j,2) + cam.space_matrix_intrinsics(j,3) * cam.space_matrix_intrinsics(i,2);
        out(3) = 2 * cam.space_matrix_intrinsics(i,3) * cam.space_matrix_intrinsics(j,3);
        //out(4) = Scalar(cam.space_matrix_intrinsics.row(i) *  cam.space_matrix_intrinsics.row(j).transpose());
        out(4) = Scalar(cam.space_matrix_intrinsics.template block<1,3>(i,0) * cam.space_matrix_intrinsics.template block<1,3>(j,0).transpose());
        return out;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_affine_space() {
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
        CHECK(lsc_->linearize_problem_affine_space())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_affine_space();
        //lsc_->scale_Jl_cols();
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_affine_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                  pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_metric_upgrade() {
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
        storage_metric_.setZero(9*bal_problem_.cameras().size(),13);
        for (int i = 0; i < bal_problem_.cameras().size(); i++) {
            typename BalBundleAdjustmentHelper<Scalar_>::VecR_metric res;
            typename BalBundleAdjustmentHelper<Scalar_>::MatRH JH;
            typename BalBundleAdjustmentHelper<Scalar_>::MatR_alpha Jalpha;

            auto const cam = bal_problem_.cameras().at(i);
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade(
                    cam.PH, cam.PHHP, cam.space_matrix_intrinsics, cam.alpha, cam.intrinsics,true, res , &JH, &Jalpha);
            storage_metric_.template block<9,3>(9*i,0) = JH;
            //storage_metric_.template block<9,1>(9*i,3) = Jalpha; //for v2
            storage_metric_.template block<9,9>(9*i,3) = Jalpha;
            storage_metric_.template block <9,1>(9*i,12) = res;
            //storage_metric_.template block <3,1>(9*i+3,12) = res.col(1);
            //storage_metric_.template block <3,1>(9*i+6,12) = res.col(2);

        }

        //IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        //pose_damping_diagonal2 = lsc_->get_Jp_diag2_pOSE();
        //@Simon: scale Jl_cols
        //lsc_->scale_Jl_cols_pOSE();
        //
        //IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        //pose_jacobian_scaling_pOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
        //                               pose_damping_diagonal2.array().sqrt())
        //        .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_metric_upgrade_v2() {

        Timer timer_stage1;
        VecX pose_damping_diagonal2;

        Timer timer;
        Scalar_ f = bal_problem_.cameras().at(0).intrinsics.getParam()[0];;
        storage_metric_.setZero(9*bal_problem_.cameras().size(),14);
        for (int i = 0; i < bal_problem_.cameras().size(); i++) {
            typename BalBundleAdjustmentHelper<Scalar_>::VecR_metric res;
            typename BalBundleAdjustmentHelper<Scalar_>::MatRH JH;
            typename BalBundleAdjustmentHelper<Scalar_>::MatR_alpha Jalpha2;
            typename BalBundleAdjustmentHelper<Scalar_>::MatR_alpha_v2 Jalpha;

            auto const cam = bal_problem_.cameras().at(i);
            //@sImon: initial version
            const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade_v2(cam.PH, cam.PHHP, cam.space_matrix_intrinsics, cam.alpha, cam.intrinsics, true, res , &JH, &Jalpha2, &Jalpha);
            //@Simon: with handcrafted JH
            //const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade_v3(bal_problem_.h_euclidean().plan_infinity,
            //        cam.PH, cam.PHHP, cam.space_matrix_intrinsics, cam.alpha, cam.intrinsics, true, res , &JH, &Jalpha2, &Jalpha);
            //@Simon: in line with Pollefeys' paper
            //const bool valid = BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade_v3_pollefeys(f, bal_problem_.h_euclidean().plan_infinity,
            //                                                                                  cam.PH, cam.PHHP, cam.space_matrix_intrinsics, cam.alpha, cam.intrinsics, true, res , &JH, &Jalpha2, &Jalpha);




            storage_metric_.template block<9,3>(9*i,0) = JH;
            storage_metric_.template block<9,1>(9*i,3) = Jalpha; //for v2
            storage_metric_.template block<9,9>(9*i,4) = Jalpha2;
            storage_metric_.template block<9,1>(9*i,13) = res;
            //storage_metric_.template block <3,1>(9*i,13) = res.col(0);
            //storage_metric_.template block <3,1>(9*i+3,13) = res.col(1);
            //storage_metric_.template block <3,1>(9*i+6,13) = res.col(2);

        }

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }


    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_pOSE(int alpha) {
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
        //@Simon: scale Jl_cols
        lsc_->scale_Jl_cols_pOSE();
        //
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
    void LinearizorPowerVarproj<Scalar_>::linearize_RpOSE(double alpha) {
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
        CHECK(lsc_->linearize_problem_RpOSE(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_RpOSE();

        //@Simon: scale Jl_cols
        lsc_->scale_Jl_cols_RpOSE();
        //
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_RpOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_RpOSE_refinement(double alpha) {
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
        CHECK(lsc_->linearize_problem_RpOSE_refinement(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_RpOSE();

        //@Simon: scale Jl_cols
        lsc_->scale_Jl_cols_RpOSE();
        //
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_RpOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                        pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_RpOSE_ML() {
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
        CHECK(lsc_->linearize_problem_RpOSE_ML())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_RpOSE_ML();

        //@Simon: scale Jl_cols
        lsc_->scale_Jl_cols_RpOSE_ML();
        //
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_RpOSE_ML_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                        pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_expOSE(int alpha, bool init) {
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
        //@Simon: scale Jl_cols
        //lsc_->scale_Jl_cols_expOSE();
        //
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
    void LinearizorPowerVarproj<Scalar_>::linearize_pOSE_rOSE(int alpha) {
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
        //@Simon: here we derive jacobians of H and alpha, and store them in storage_metric_:

        CHECK(lsc_->linearize_problem_pOSE_rOSE(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_pOSE_rOSE();
        //@Simon: scale Jl_cols
        //lsc_->scale_Jl_cols_pOSE();
        //
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_pOSE_rOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_rOSE(int alpha) {
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
        CHECK(lsc_->linearize_problem_rOSE(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_rOSE();
        //@Simon: scale Jl_cols
        //lsc_->scale_Jl_cols_pOSE();
        //
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_rOSE_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_pOSE_homogeneous(int alpha) {
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
        CHECK(lsc_->linearize_problem_pOSE_homogeneous(alpha))
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_pOSE_homogeneous();
        //@Simon: scale Jl_cols
        //lsc_->scale_Jl_cols_pOSE(); @Simon: FIX IT
        //lsc_->scale_Jl_cols_pOSE_homogeneous();
        //
        IF_SET(it_summary_)->scale_landmark_jacobian_time_in_seconds = timer.reset();

        // TODO: maybe we should reset pose_jacobian_scaling_ at iteration end to
        // avoid accidental use of outdated info?

        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
        pose_jacobian_scaling_pOSE_homogeneous_ = (Base::get_effective_jacobi_scaling_epsilon() +
                                       pose_damping_diagonal2.array().sqrt())
                .inverse();

        IF_SET(it_summary_)->stage1_time_in_seconds = timer_stage1.elapsed();
        IF_SET(summary_)->num_jacobian_evaluations += 1;

        new_linearization_point_ = true;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::linearize_projective_space() {
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
        CHECK(lsc_->linearize_problem_projective_space())
                        << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_projective_space();
        //lsc_->scale_Jl_cols();
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
//@Simon: without storage_nullspace:
        CHECK(lsc_->linearize_problem_projective_space_homogeneous())
                        << "did not expect numerical failure during linearization";
//@Simon: with storage_nullspace:
        //CHECK(lsc_->linearize_problem_projective_space_homogeneous_storage())
        //                << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_projective_space();
        //@Simon: scale Jl_cols homogeneous
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
    void LinearizorPowerVarproj<Scalar_>::linearize_projective_space_homogeneous_RpOSE() {
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
//@Simon: without storage_nullspace:
        CHECK(lsc_->linearize_problem_projective_space_homogeneous_RpOSE())
                        << "did not expect numerical failure during linearization";
//@Simon: with storage_nullspace:
        //CHECK(lsc_->linearize_problem_projective_space_homogeneous_storage())
        //                << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_projective_space_RpOSE();
        //@Simon: scale Jl_cols homogeneous
        lsc_->scale_Jl_cols_homogeneous_RpOSE();
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
    void LinearizorPowerVarproj<Scalar_>::linearize_projective_space_homogeneous_lm_landmark() {
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
//@Simon: without storage_nullspace:
        CHECK(lsc_->linearize_problem_projective_space_homogeneous())
                        << "did not expect numerical failure during linearization";
//@Simon: with storage_nullspace:
        //CHECK(lsc_->linearize_problem_projective_space_homogeneous_storage())
        //                << "did not expect numerical failure during linearization";
        IF_SET(it_summary_)->jacobian_evaluation_time_in_seconds = timer.reset();
        pose_damping_diagonal2 = lsc_->get_Jp_diag2_projective_space();
        //@Simon: scale Jl_cols homogeneous
        lsc_->scale_Jl_cols_homogeneous();
        //
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
    void LinearizorPowerVarproj<Scalar_>::linearize_refine() {
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
        CHECK(lsc_->linearize_problem_refine())
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_refine(
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
            lsc_->prepare_Hb_refine(b_p);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve(
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
        //pso.r_tolerance = 0.01;

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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_joint_RpOSE(
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
            lsc_->prepare_Hb_joint_RpOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_joint_RpOSE(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_affine_space(
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
            lsc_->scale_Jp_cols_affine(pose_jacobian_scaling_affine_);
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
            lsc_->prepare_Hb_affine_space(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_affine_space(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_pOSE(
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
            lsc_->prepare_Hb_pOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_RpOSE(
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
            lsc_->scale_Jp_cols_RpOSE(pose_jacobian_scaling_RpOSE_);
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
            lsc_->prepare_Hb_RpOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_RpOSE(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_RpOSE_ML(
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
            lsc_->scale_Jp_cols_RpOSE_ML(pose_jacobian_scaling_RpOSE_ML_);
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
            lsc_->prepare_Hb_RpOSE_ML(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_RpOSE_ML(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_pOSE_poBA(
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
            lsc_->prepare_Hb_pOSE_poBA(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_metric_upgrade(
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



        IF_SET(it_summary_)->landmark_damping_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->stage2_time_in_seconds = timer_stage2.elapsed();

        // ////////////////////////////////////////////////////////////////////////
        // Solving:
        // - marginalize landmarks (SC)
        // - run PCG
        // ////////////////////////////////////////////////////////////////////////

        // compute Schur complement
        int num_cam = bal_problem_.cameras().size();
        int pose_time_num_cam = 9 * num_cam;
        Vec3 inc;
        MatX I_JTJ;
        I_JTJ.resize(9*num_cam,9*num_cam);
        I_JTJ.setZero();

        const auto J = storage_metric_.leftCols(3);
        const auto r = storage_metric_.rightCols(1);

        //const auto Jalpha = storage_metric_.middleCols(3);

        //@Simon: for VarPro
        for (int i = 0; i < num_cam; i++) {
            I_JTJ.template block<9,9>(9*i,9*i) = MatX::Identity(9,9) - storage_metric_.template block<9,9>(9 * i,4);// / (1.0 + lambda);
        }


        //@Simon: for joint estimation:
        //for (int i = 0; i < num_cam; i++) {
        //    I_JTJ.template block<9,9>(9*i,9*i) = MatX::Identity(9,9) - storage_metric_.template block<9,1>(9 * i,3) * ((1.0 + lambda) * storage_metric_.template block<9,1>(9 * i,3).transpose()*storage_metric_.template block<9,1>(9 * i,3)).inverse() * storage_metric_.template block<9,1>(9 * i,3).transpose();// / (1.0 + lambda);
        //    //I_JTJ.template block<9,9>(9*i,9*i) /= 1.0 + lambda;
        //}


        //Vec3 b = J.transpose() * I_JTJ * r;// / Scalar_(1.0 + lambda);
        Vec3 b = J.transpose() * r;
        Mat3 JtJ = J.transpose() *I_JTJ* J;// / Scalar_(1.0 + lambda);
        Mat3 Diag = J.transpose()*J;

        std::cout << "b.norm() = " << b.norm() << "\n";
        std::cout << "J_H.norm() = " << J.norm() << "\n";
        std::cout << "I_JTJ.norm() = " << I_JTJ.norm() << "\n";

        Mat3 D;
        D.setZero();
        D(0,0) = Diag(0,0);
        D(1,1) = Diag(1,1);
        D(2,2) = Diag(2,2);

        std::cout << "lambda * D.norm() = " << (lambda * D).norm() << "\n";


        inc = - (JtJ + lambda * D).inverse() * b;
        //inc = - (JtJ).inverse() * b;
        //inc = - (JtJ + lambda * MatX::Identity(3,3)).inverse() * b;
        //inc = - (JtJ + lambda * MatX::Identity(3,3)).colPivHouseholderQr().solve(b);

        std::cout << "in solve     inc = " << inc.norm() << "\n";

        new_linearization_point_ = false;

        return inc;
    }


    template <class Scalar_>
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_direct_pOSE(
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
            lsc_->prepare_Hb_pOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_direct_RpOSE(
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
            lsc_->scale_Jp_cols_RpOSE(pose_jacobian_scaling_RpOSE_);
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
            lsc_->prepare_Hb_RpOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_RpOSE(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_direct_expOSE(
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
            lsc_->prepare_Hb_expOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_expOSE(
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
            lsc_->scale_Jp_cols_expOSE(pose_jacobian_scaling_expOSE_);
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
            lsc_->prepare_Hb_expOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_expOSE(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_pOSE_rOSE(
            Scalar lambda, Scalar relative_error_change) {
        // ////////////////////////////////////////////////////////////////////////
        // Stage 2: inside LM solver inner-loop
        // - scale pose Jacobians (1st inner it)
        // - dampen
        // ////////////////////////////////////////////////////////////////////////
        Timer timer_stage2;

        // dampen pose
        lsc_->set_pose_damping(lambda);
        Timer timer;

        // scale pose jacobians only on the first inner iteration
        if (new_linearization_point_) {
            lsc_->scale_Jp_cols_pOSE_rOSE(pose_jacobian_scaling_pOSE_rOSE_);
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
            lsc_->prepare_Hb_pOSE_rOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_pOSE_rOSE(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_rOSE(
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
            lsc_->scale_Jp_cols_rOSE(pose_jacobian_scaling_rOSE_);
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
            lsc_->prepare_Hb_rOSE(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_rOSE(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_pOSE_riemannian_manifold(
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
            lsc_->prepare_Hb_pOSE_riemannian_manifold(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_pOSE_riemannian_manifold(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_pOSE_homogeneous(
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
            lsc_->scale_Jp_cols_pOSE_homogeneous(pose_jacobian_scaling_pOSE_homogeneous_);
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
            lsc_->prepare_Hb_pOSE_homogeneous(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_pOSE_homogeneous(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_projective_space(
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
            lsc_->prepare_Hb_projective_space_homogeneous(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_projective_space_homogeneous(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_projective_space_homogeneous(
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
            //lsc_->prepare_Hb_projective_space(b_p);
            //@Simon: with Riemannian manifold optimization:
            lsc_->prepare_Hb_projective_space_homogeneous_riemannian_manifold(b_p);
            //@Simon: without Riemannian manifold optimization:
            //lsc_->prepare_Hb_projective_space_homogeneous(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_projective_space_homogeneous(b_p, inc, pso);
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
    typename LinearizorPowerVarproj<Scalar_>::VecX LinearizorPowerVarproj<Scalar_>::solve_projective_space_homogeneous_riemannian_manifold(
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
            lsc_->scale_Jp_cols(pose_jacobian_scaling_); //@Simon: ISSUE here. TRY: either create storage_nullspace here, or scale the projected Jacobian
            IF_SET(it_summary_)->scale_pose_jacobian_time_in_seconds = timer.reset();
            lsc_->linearize_nullspace();
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
            //lsc_->prepare_Hb_projective_space(b_p);
            //@Simon: with Riemannian manifold optimization:
            lsc_->prepare_Hb_projective_space_homogeneous_riemannian_manifold(b_p);
            //@Simon: without Riemannian manifold optimization:
            //lsc_->prepare_Hb_projective_space_homogeneous(b_p);
            IF_SET(it_summary_)->prepare_time_in_seconds = timer.elapsed();
        }
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions pso;
        pso.q_tolerance = options_.eta;
        pso.r_tolerance = options_.r_tolerance;
        //pso.r_tolerance = 0.01;

        VecX inc;
        {
            Timer timer;
            const auto summary = lsc_->solve_projective_space_homogeneous_riemannian_manifold_storage(b_p, inc, pso);
            //const auto summary = lsc_->solve_projective_space_homogeneous_riemannian_manifold(b_p, inc, pso);
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
    void LinearizorPowerVarproj<Scalar_>::estimate_alpha(VecX&& inc, Scalar lambda) {
        using Vec9 = Eigen::Matrix<Scalar, 9, 1>;
        for (int i = 0; i < bal_problem_.cameras().size(); i++) {
            //std::cout << "in estimate alpha l2145 \n";

            auto cam = bal_problem_.cameras().at(i);


            auto jh_i = storage_metric_.template block<9,3>(9*i,0);
            auto jalpha_i = storage_metric_.template block<9,1>(9*i,3);
            auto res_init = storage_metric_.template block<9,1>(9*i,13);

            typename BalBundleAdjustmentHelper<Scalar>::MatR_alpha_v2 Jalpha;
            typename BalBundleAdjustmentHelper<Scalar>::MatR_alpha Jalpha2;
            typename BalBundleAdjustmentHelper<Scalar>::MatRH JH;

            typename BalBundleAdjustmentHelper<Scalar>::VecR_metric res;

            //auto obs_i = storage_.template block<2,1>(2 * i, obs_idx_);

            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_jacobian_metric_upgrade(cam.PH, cam.PHHP,
                                                                                                cam.space_matrix_intrinsics,
                                                                                                cam.alpha, cam.intrinsics,
                                                                                                true, res , &JH,
                                                                                                &Jalpha2, &Jalpha);

//            const bool valid = BalBundleAdjustmentHelper<Scalar>::update_jacobian_metric_upgrade_v3(bal_problem_.h_euclidean().plan_infinity,
//                                                                                                    cam.PH, cam.PHHP,
//                                                                                                 cam.space_matrix_intrinsics,
//                                                                                                 cam.alpha, cam.intrinsics,
//                                                                                                 true, res , &JH,
//                                                                                                 &Jalpha2, &Jalpha);

            Vec9 ri = res;

            //@Simon: for joint optimization:
            //Scalar_ inc_alpha = (- Scalar_(jalpha_i.transpose() * res_init) - Scalar_(jalpha_i.transpose() * jh_i * inc)) /Scalar_((1+lambda)*jalpha_i.transpose() * jalpha_i);
            //Scalar_ inc_alpha = - Scalar_(jalpha_i.transpose() * (res_init + jh_i * inc)) / ((1.0 + lambda) * Scalar_(jalpha_i.transpose() * jalpha_i));

            //@Simon: for VarProj
            //const auto Ji = storage_metric_.template block<9,3>(9*i,0);
            //const auto Jalphai = storage_metric_.template block<9,1>(9*i,3);
            //const auto ri = storage_metric_.template block<9,1>(9*i,13);
            //Scalar_ inc_alpha = (- Scalar_(Jalpha.transpose() * ri) - Scalar_(Jalpha.transpose() * JH * inc))/Scalar_(Jalpha.transpose() * Jalpha);
            Scalar_ inc_alpha = (- Scalar_(Jalpha.transpose() * ri))/Scalar_(Jalpha.transpose() * Jalpha);

            bal_problem_.cameras().at(i).alpha += inc_alpha;
        }
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form(VecX&& inc) {
        Timer timer;

        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();

        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
//        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
//            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
//            bal_problem_.cameras()[i].apply_inc_intrinsics(
//                    inc.template segment<3>(i * 9 + 6));
//        }
        //@Simon: Update cameras in the camera matrix space, and not in SE(3) space:
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 9 + 8));
        }

        inc.array() *= pose_jacobian_scaling_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_affine_space(VecX&& inc) {
        Timer timer;
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_affine_.array();
        // update cameras //@Simon: for VarProj, we update cameras before deriving closed_form
//        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
//            bal_problem_.cameras()[i].apply_inc_pose(inc.template segment<6>(i * 9));
//            bal_problem_.cameras()[i].apply_inc_intrinsics(
//                    inc.template segment<3>(i * 9 + 6));
//        }
        //@Simon: Update cameras in the camera matrix space, and not in SE(3) space:

        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_affine_space(inc.template segment<8>(i * 11));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 11 + 8));
        }
        inc.array() *= pose_jacobian_scaling_affine_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_affine_space(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_pOSE(int alpha, VecX&& inc) {
        Timer timer;
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_.array();

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
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_RpOSE(double alpha, VecX&& inc) {
        Timer timer;
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_RpOSE_.array();

        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_RpOSE(inc.template segment<8>(i * 11));

        }

        inc.array() *= pose_jacobian_scaling_RpOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_RpOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_RpOSE_refinement(double alpha, VecX&& inc) {
        Timer timer;
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_RpOSE_.array();

        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_RpOSE(inc.template segment<8>(i * 11));

        }

        inc.array() *= pose_jacobian_scaling_RpOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_RpOSE_refinement(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_RpOSE_ML(VecX&& inc) {
        Timer timer;
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_RpOSE_ML_.array();

        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_RpOSE(inc.template segment<8>(i * 11));

        }
        inc.array() *= pose_jacobian_scaling_RpOSE_ML_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_RpOSE_ML(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();

        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_pOSE_poBA(int alpha, VecX&& inc) {
        Timer timer;
        Scalar l_diff = lsc_->back_substitute_poBA(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_.array();
        // update cameras
        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));
        }
        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();
        return l_diff;
    }


    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_expOSE(int alpha, VecX&& inc) {
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
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_pOSE_rOSE(int alpha, VecX&& inc) {
        Timer timer;
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_pOSE_rOSE_.array();

        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));

        }

        inc.array() *= pose_jacobian_scaling_pOSE_rOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_pOSE_rOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_rOSE(int alpha, VecX&& inc) {
        Timer timer;
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_rOSE_.array();

        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc.template segment<12>(i * 15));

        }


        inc.array() *= pose_jacobian_scaling_rOSE_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_rOSE(alpha, inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_pOSE_riemannian_manifold(int alpha, VecX&& inc) {
        Timer timer;
        VecX inc_;
        inc_.resize(bal_problem_.cameras().size() * 12);

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
            inc_proj.array() *= pose_jacobian_scaling_pOSE_.array().template segment<12>(i*15);
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc_proj);
            inc_proj.array() *= pose_jacobian_scaling_pOSE_.array().template segment<12>(i*15).inverse();
            inc_.template segment<12>(i*12) = inc_proj;
        }


        Scalar l_diff = lsc_->landmark_closed_form_pOSE_riemannian_manifold(alpha, inc_);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_pOSE_homogeneous(int alpha, VecX&& inc) {
        Timer timer;
        // unscale pose increments

//        }
        //@Simon: Update cameras in the camera matrix space, and not in SE(3) space:

        VecX inc_;
        inc_.resize(bal_problem_.cameras().size() * 12);
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

            inc_proj.array() *= pose_jacobian_scaling_pOSE_homogeneous_.array().template segment<12>(i*15);
            bal_problem_.cameras()[i].apply_inc_pose_pOSE(inc_proj);
            inc_proj.array() *= pose_jacobian_scaling_pOSE_homogeneous_.array().template segment<12>(i*15).inverse();

            inc_.template segment<12>(i*12) = inc_proj;
        }

        //inc.array() *= pose_jacobian_scaling_pOSE_homogeneous_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_pOSE_homogeneous(alpha, inc_);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_projective_space(VecX&& inc) {
        Timer timer;
        // unscale pose increments
        inc.array() *= pose_jacobian_scaling_.array();
        //@Simon: Update cameras in the camera matrix space, and not in SE(3) space:

        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_projective_space(inc.template segment<12>(i * 15));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 15 + 12));
        }

        inc.array() *= pose_jacobian_scaling_.array().inverse();
        Scalar l_diff = lsc_->landmark_closed_form_projective_space_homogeneous(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::update_y_tilde_expose(){
        lsc_->update_y_tilde_expose();
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::initialize_y_tilde_expose(){
        lsc_->initialize_y_tilde_expose();
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::rpose_new_equilibrium(){
        lsc_->rpose_new_equilibrium();
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::apply_pose_update_riemannian_manifold(VecX&& inc) {
        //@Simon: TODO: Fix the scaling: seems to be the reason of the inaccurate results
        //inc.array() *= pose_jacobian_scaling_.array();



        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {

            //const size_t cam_idx = pose_indices[i];
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
            inc_proj.array() *= pose_jacobian_scaling_.array().template segment<12>(i*15).inverse();

        }

    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::apply_pose_update(VecX&& inc) {
        //inc.array() *= pose_jacobian_scaling_.array();

        for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
            bal_problem_.cameras()[i].apply_inc_pose_projective_space(inc.template segment<12>(i * 15));
            bal_problem_.cameras()[i].apply_inc_intrinsics(
                    inc.template segment<3>(i * 15 + 12));
        }
        //inc.array() *= pose_jacobian_scaling_.array().inverse();
    }


    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::non_linear_varpro_landmark(Scalar lambda) {
        Timer timer;
        //@Simon: TODO: put the apply_inc part before optimize_lm_landmark
        // unscale pose increments
        //inc.array() *= pose_jacobian_scaling_.array();
//
        //for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
        //    bal_problem_.cameras()[i].apply_inc_pose_projective_space(inc.template segment<12>(i * 15));
        //    bal_problem_.cameras()[i].apply_inc_intrinsics(
        //            inc.template segment<3>(i * 15 + 12));
        //}
        //inc.array() *= pose_jacobian_scaling_.array().inverse();

        // Dampen landmarks
        lsc_->set_landmark_damping(lambda);

        VecX delta_lm;

        //@Simon: with riemannian manifold projection
        lsc_->solve_lm_landmark_riemannian_manifold(delta_lm);
        //@Simon: without riemannian manifold projection
        //lsc_->solve_lm_landmark(delta_lm);


        // if we backtrack, we don't need to rescale jacobians / preconditioners in
        // the next `solve` call
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_projective_space_homogeneous(VecX&& inc) {
        Timer timer;
        // unscale pose increments
        //inc.array() *= pose_jacobian_scaling_.array();

        //for (size_t i = 0; i < bal_problem_.cameras().size(); i++) {
        //    bal_problem_.cameras()[i].apply_inc_pose_projective_space(inc.template segment<12>(i * 15));
        //    bal_problem_.cameras()[i].apply_inc_intrinsics(
        //            inc.template segment<3>(i * 15 + 12));
        //}
        //inc.array() *= pose_jacobian_scaling_.array().inverse();

        Scalar l_diff = lsc_->landmark_closed_form_projective_space_homogeneous_riemannian_manifold(inc);
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

        return l_diff;
    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::closed_form_projective_space_homogeneous_nonlinear_initialization() {
        Timer timer;
        // unscale pose increments

        lsc_->landmark_closed_form_projective_space_homogeneous_nonlinear_initialization();
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

    }

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold() {
        Timer timer;
        // unscale pose increments

        lsc_->landmark_closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold();
        IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

    }

    //template <class Scalar_>
    //Scalar_ LinearizorPowerVarproj<Scalar_>::closed_form_projective_space_homogeneous_lm_landmark(VecX&& inc) {
    //    Timer timer;


    //    Scalar l_diff = lsc_->landmark_closed_form_projective_space_homogeneous_lm_landmark(inc);
    //    IF_SET(it_summary_)->back_substitution_time_in_seconds = timer.reset();


    //    IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();

    //    return l_diff;
    //}

    template <class Scalar_>
    void LinearizorPowerVarproj<Scalar_>::radial_estimate() {
        for (int i = 0; i < bal_problem_.cameras().size(); i++) {
            auto cam = bal_problem_.cameras().at(i);
            using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
            using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
            using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
            using Vec7 = Eigen::Matrix<Scalar, 7, 1>;
            MatX A(cam.linked_lm.size(),6);
            VecX b(cam.linked_lm.size());
            std::unordered_set<int>::const_iterator it = cam.linked_lm.begin();
            int l = 0;
            for (; it != cam.linked_lm.end(); it++) {
                int lm_i = *it;
                auto lm_cam = bal_problem_.landmarks().at(lm_i);
                const auto& obs = lm_cam.obs.at(i);
                //Scalar lambda = 1.0;
                Scalar lambda = Scalar_(cam.space_matrix.row(0) * lm_cam.p_w.homogeneous()) / obs.pos(0);
                A.setZero();
                b.setZero();
                A(l, 0) = lm_cam.p_w(0);
                A(l, 1) = lm_cam.p_w(1);
                A(l, 2) = lm_cam.p_w(2);
                A(l, 3) = 1;
                A(l, 4) = -(obs.pos(0) * obs.pos(0) + obs.pos(1) * obs.pos(1)) * lambda;
                A(l, 5) = -std::pow((obs.pos(0) * obs.pos(0) + obs.pos(1) * obs.pos(1)),2) * lambda;
                b(l) = lambda;
                l += 1;

            }
            Vec6 radial = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
            cam.space_matrix(2,0) = radial(0);
            cam.space_matrix(2,1) = radial(1);
            cam.space_matrix(2,2) = radial(2);
            cam.space_matrix(2,3) = radial(3);

            cam.intrinsics = basalt::BalCamera<Scalar_>(Vec3(cam.intrinsics.getParam()[0], radial(4), radial(5)));

        }

    }


    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::apply(VecX&& inc) {
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
    Scalar_ LinearizorPowerVarproj<Scalar_>::apply_joint(VecX&& inc) {
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

            //bal_problem_.cameras()[i].space_matrix.normalize();

        }
        IF_SET(it_summary_)->update_cameras_time_in_seconds = timer.elapsed();
        return l_diff;
    }

    template <class Scalar_>
    Scalar_ LinearizorPowerVarproj<Scalar_>::apply_joint_RpOSE(VecX&& inc) {
        // backsubstitue landmarks and compute model cost difference
        using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
        Timer timer;
        Scalar l_diff = lsc_->back_substitute_joint_RpOSE(inc);
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
            VecX inc_proj = Proj_pose * inc.template segment<11>(i*13);
            inc_proj.array() *= pose_jacobian_scaling_.array().template segment<12>(i*15);

            bal_problem_.cameras()[i].apply_inc_pose_projective_space(inc_proj);

            Vec3 inc_intrinsics; // = Vec3(0,inc.template segment<2>(i*13 +11));
            inc_intrinsics(0) = 0;
            inc_intrinsics(1) = inc(i*13 + 11);
            inc_intrinsics(2) = inc(i*13 + 12);

            inc_intrinsics.array() *= pose_jacobian_scaling_.array().template segment<3>(i*15+12);
            bal_problem_.cameras()[i].apply_inc_pose_projective_space_intrinsics(inc_intrinsics);

            //bal_problem_.cameras()[i].space_matrix.normalize();

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

}  // namespace rootba
