//
// Created by Simon Weber on 10/07/2023.
//

#pragma once

#include "rootba_povar/solver/linearizor_base.hpp"

namespace rootba_povar {

    template <typename Scalar, int POSE_SIZE>
    class LinearizationPowerVarproj;

    template <class Scalar_>
    class LinearizorPowerVarproj : public LinearizorBase<Scalar_> {
    public:
        using Scalar = Scalar_;
        using Base = LinearizorBase<Scalar>;
        constexpr static int POSE_SIZE = 12;
        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    public:  // public interface
        LinearizorPowerVarproj(BalProblem<Scalar>& bal_problem,
                          const SolverOptions& options,
                          SolverSummary* summary = nullptr);

        ~LinearizorPowerVarproj() override;


        void linearize_pOSE(Scalar alpha) override;
        void linearize_projective_space_homogeneous() override;



        VecX solve_joint(Scalar lambda, Scalar relative_error_change) override;
        VecX solve(const SolverOptions& solver_options, Scalar lambda, Scalar relative_error_change) override;

        Scalar apply_joint(VecX&& inc) override;
        Scalar apply(const SolverOptions& solver_options, Scalar alpha, VecX&& inc) override;

    private:
        using Base::bal_problem_;
        using Base::it_summary_;
        using Base::options_;
        using Base::summary_;

        std::unique_ptr<LinearizationPowerVarproj<Scalar, POSE_SIZE>> lsc_;

        // set during linearization, used in solve
        VecX pose_jacobian_scaling_;
        VecX pose_jacobian_scaling_pOSE_;

        // indicates if we call solve the first time since the last linearization
        // (first inner iteration for LM-backtracking); true after `linearize`, false
        // after `solve`;
        bool new_linearization_point_ = false;
    };

}  // namespace rootba_povar
