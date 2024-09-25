//
// Created by Simon on 10/07/2023.
//

#pragma once

#include "rootba/solver/linearizor_base.hpp"

namespace rootba {

    template <typename Scalar, int POSE_SIZE>
    class LinearizationPowerVarproj;

    template <class Scalar_>
    class LinearizorPowerVarproj : public LinearizorBase<Scalar_> {
    public:
        using Scalar = Scalar_;
        using Base = LinearizorBase<Scalar>;
        //constexpr static int POSE_SIZE = 9;
        //constexpr static int POSE_SIZE = 15;
        constexpr static int POSE_SIZE = 11;
        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    public:  // public interface
        LinearizorPowerVarproj(BalProblem<Scalar>& bal_problem,
                          const SolverOptions& options,
                          SolverSummary* summary = nullptr);

        ~LinearizorPowerVarproj() override;

        void linearize() override;
        void linearize_refine() override;
        void linearize_affine_space() override;
        void linearize_pOSE(int alpha) override;
        void linearize_RpOSE(double alpha) override;
        void linearize_RpOSE_refinement(double alpha) override;
        void linearize_RpOSE_ML() override;
        void linearize_expOSE(int alpha, bool init) override;
        void linearize_metric_upgrade() override;
        void linearize_metric_upgrade_v2() override;
        void linearize_pOSE_rOSE(int alpha) override;
        void linearize_rOSE(int alpha) override;
        void linearize_pOSE_homogeneous(int alpha) override;
        void linearize_projective_space() override;
        void linearize_projective_space_homogeneous() override;
        void linearize_projective_space_homogeneous_RpOSE() override;
        void linearize_projective_space_homogeneous_lm_landmark() override;


        void radial_estimate() override;

        void estimate_alpha(VecX&& inc, Scalar lambda) override;

        void update_y_tilde_expose() override;
        void initialize_y_tilde_expose() override;
        void rpose_new_equilibrium() override;
        Eigen::Matrix<Scalar, 5, 1> alpha(int num_cameras, int i, int j);
        void compute_plane_linearly() override;

        //void initialize_nonlinear_varpro() override;
        void non_linear_varpro_landmark(Scalar lambda) override;
        void apply_pose_update(VecX&& inc) override;
        void apply_pose_update_riemannian_manifold(VecX&& inc) override;
        void closed_form_projective_space_homogeneous_nonlinear_initialization() override;
        void closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold() override;

        VecX solve(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_metric_upgrade(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_joint(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_joint_RpOSE(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_refine(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_affine_space(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_pOSE(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_RpOSE(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_RpOSE_ML(Scalar lambda, Scalar relative_error_change) override;
        VecX solve_pOSE_poBA(Scalar lambda, Scalar relative_error_change) override;
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

        Scalar apply(VecX&& inc) override;
        Scalar apply_joint(VecX&& inc) override;
        Scalar apply_joint_RpOSE(VecX&& inc) override;

        Scalar closed_form(VecX&& inc) override;
        Scalar closed_form_affine_space(VecX&& inc) override;
        Scalar closed_form_pOSE(int alpha, VecX&& inc) override;
        Scalar closed_form_RpOSE(double alpha, VecX&& inc) override;
        Scalar closed_form_RpOSE_refinement(double alpha, VecX&& inc) override;
        Scalar closed_form_RpOSE_ML(VecX&& inc) override;
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

        std::unique_ptr<LinearizationPowerVarproj<Scalar, POSE_SIZE>> lsc_;

        // set during linearization, used in solve
        VecX pose_jacobian_scaling_;
        VecX pose_jacobian_scaling_pOSE_;
        VecX pose_jacobian_scaling_RpOSE_;
        VecX pose_jacobian_scaling_RpOSE_ML_;
        VecX pose_jacobian_scaling_expOSE_;
        VecX pose_jacobian_scaling_pOSE_rOSE_;
        VecX pose_jacobian_scaling_rOSE_;
        VecX pose_jacobian_scaling_pOSE_homogeneous_;
        VecX pose_jacobian_scaling_affine_;

        // indicates if we call solve the first time since the last linearization
        // (first inner iteration for LM-backtracking); true after `linearize`, false
        // after `solve`;
        bool new_linearization_point_ = false;

        //@Simon: for metric upgrade
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> storage_metric_;
    };

}  // namespace rootba
