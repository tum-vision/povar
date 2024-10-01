//
// Created by Simon on 10/07/2023.
//

#ifndef ROOTBA_LINEARIZATION_POWER_VARPROJ_HPP
#define ROOTBA_LINEARIZATION_POWER_VARPROJ_HPP

#pragma once

#include <Eigen/Dense>
#include <basalt_custom/utils/sophus_utils.hpp>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "rootba_povar/bal/bal_problem.hpp"
#include "rootba_povar/cg/conjugate_gradient.hpp"
#include "rootba_povar/sc/landmark_block.hpp"
#include "rootba_povar/sc/linearization_varproj.hpp"
#include "rootba_povar/util/assert.hpp"


namespace rootba_povar {

    template <typename Scalar_, int POSE_SIZE_>
    class LinearizationPowerVarproj : private LinearizationVarProj<Scalar_, POSE_SIZE_> {
    public:
        using Scalar = Scalar_;
        static constexpr int POSE_SIZE = POSE_SIZE_;
        using Base = LinearizationVarProj<Scalar_, 12>;


        using Vec2 = Eigen::Matrix<Scalar_, 2, 1>;
        using Vec4 = Eigen::Matrix<Scalar_, 4, 1>;
        using Vec12 = Eigen::Matrix<Scalar, 12, 1>;
        using VecX = Eigen::Matrix<Scalar_, Eigen::Dynamic, 1>;

        using MatX = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;
        using RowMatX =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using PerSolveOptions =
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions;

        struct Options {
            typename LandmarkBlockSC<Scalar, POSE_SIZE>::Options sc_options;
            size_t power_sc_iterations = 20;
        };

        struct Summary {
            enum TerminationType {
                LINEAR_SOLVER_NO_CONVERGENCE,
                LINEAR_SOLVER_SUCCESS,
                LINEAR_SOLVER_FAILURE
            };

            TerminationType termination_type;
            std::string message;
            int num_iterations = 0;
        };

        LinearizationPowerVarproj(BalProblem<Scalar>& bal_problem, const Options& options)
                : Base(bal_problem, options.sc_options, false),
                  m_(options.power_sc_iterations),
                  bal_problem_(bal_problem),
                  b_inv_pOSE_(12 * num_cameras_, 12),
                  b_inv_joint_(11 * num_cameras_, 11),
                  hll_inv_pOSE_(3 * landmark_blocks_.size(), 3),
                  hll_inv_joint_(3 * landmark_blocks_.size(), 3){
            ROOTBA_ASSERT(pose_mutex_.size() == num_cameras_);
        }

        void prepare_Hb_joint(VecX& b_p) {
            b_inv_joint_.setZero(11 * num_cameras_, 11);
            b_p.setZero(num_cameras_ * 11);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_joint_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_joint(b_inv_joint_, hll_inv, b_p, bal_problem_.cameras(), pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_joint_.template block<11, 11>(11 * r, 0);
                        const auto& cam = bal_problem_.cameras().at(r);

                        Vec12 camera_space_matrix;
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

                        auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                        b_inv += Proj_pose.transpose() * pose_damping_diagonal_ * Proj_pose;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(11, 11));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_pOSE(VecX& b_p) {
            b_inv_pOSE_.setZero(12 * num_cameras_, 12);
            b_p.setZero(num_cameras_ * 12);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_pOSE_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_pOSE(b_inv_pOSE_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_pOSE_.template block<12, 12>(12 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(12, 12));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_pOSE_poBA(VecX& b_p) {
            b_inv_pOSE_.setZero(12 * num_cameras_, 12);
            b_p.setZero(num_cameras_ * 12);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_pOSE_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_pOSE_poBA(b_inv_pOSE_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_pOSE_.template block<12, 12>(12 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(12, 12));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }


        Summary solve_pOSE(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 12 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_pOSE(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_pOSE(right_mul_e0_pOSE(tmp));
                    accum += tmp;

                    // Check if converged
                    const Scalar iter_norm =
                            pso.q_tolerance > 0 || pso.r_tolerance > 0 ? tmp.norm() : 0;
                    if (pso.q_tolerance > 0) {
                        const Scalar zeta = i * iter_norm / accum.norm();
                        if (zeta < pso.q_tolerance) {
                            summary.termination_type = Summary::LINEAR_SOLVER_SUCCESS;
                            summary.num_iterations = i;
                            std::stringstream ss;
                            ss << "Iteration: " << summary.num_iterations
                               << " Convergence. zeta = " << zeta << " < " << pso.q_tolerance;
                            summary.message = ss.str();
                            return summary;
                        }
                    }
                    if (pso.r_tolerance > 0 && iter_norm / norm_0 < pso.r_tolerance) {
                        summary.termination_type = Summary::LINEAR_SOLVER_SUCCESS;
                        summary.num_iterations = i;
                        std::stringstream ss;
                        ss << "Iteration: " << summary.num_iterations
                           << " Convergence. |r| = " << iter_norm / norm_0 << " < "
                           << pso.r_tolerance;
                        summary.message = ss.str();
                        return summary;
                    }
                }
            }

            summary.termination_type = Summary::LINEAR_SOLVER_NO_CONVERGENCE;
            summary.num_iterations = m_;
            summary.message = "Maximum number of iterations reached.";
            return summary;
        }


        Summary solve_joint(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 11 * num_cameras_);

            Summary summary;

            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_joint(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_joint(right_mul_e0_joint(tmp));
                    accum += tmp;

                    // Check if converged
                    const Scalar iter_norm =
                            pso.q_tolerance > 0 || pso.r_tolerance > 0 ? tmp.norm() : 0;
                    if (pso.q_tolerance > 0) {
                        const Scalar zeta = i * iter_norm / accum.norm();
                        if (zeta < pso.q_tolerance) {
                            summary.termination_type = Summary::LINEAR_SOLVER_SUCCESS;
                            summary.num_iterations = i;
                            std::stringstream ss;
                            ss << "Iteration: " << summary.num_iterations
                               << " Convergence. zeta = " << zeta << " < " << pso.q_tolerance;
                            summary.message = ss.str();
                            return summary;
                        }
                    }
                    if (pso.r_tolerance > 0 && iter_norm / norm_0 < pso.r_tolerance) {
                        summary.termination_type = Summary::LINEAR_SOLVER_SUCCESS;
                        summary.num_iterations = i;
                        std::stringstream ss;
                        ss << "Iteration: " << summary.num_iterations
                           << " Convergence. |r| = " << iter_norm / norm_0 << " < "
                           << pso.r_tolerance;
                        summary.message = ss.str();
                        return summary;
                    }
                }
            }

            summary.termination_type = Summary::LINEAR_SOLVER_NO_CONVERGENCE;
            summary.num_iterations = m_;
            summary.message = "Maximum number of iterations reached.";
            return summary;
        }

        Scalar back_substitute_pOSE(Scalar alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 12));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].back_substitute_pOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        // make selected base class methods public
        using Base::back_substitute_poBA;
        using Base::back_substitute_joint;
        using Base::get_Jp_diag2_pOSE;
        using Base::get_Jp_diag2_projective_space;
        using Base::num_cols_reduced;
        using Base::scale_Jl_cols_pOSE;
        using Base::scale_Jl_cols_homogeneous;
        using Base::scale_Jp_cols_joint;
        using Base::scale_Jp_cols_pOSE;
        using Base::set_landmark_damping;
        using Base::set_landmark_damping_joint;
        using Base::set_pose_damping;
        using Base::linearize_problem_pOSE;
        using Base::linearize_problem_projective_space_homogeneous;
        using Base::linearize_nullspace;


        inline VecX right_mul_b_inv_pOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 12);

            VecX res(num_cameras_ * 12);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_pOSE_.template block<12, 12>(12 * r, 0);
                    const auto v = x.template segment<12>(12 * r);
                    res.template segment<12>(12 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_joint(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_joint_.template block<11, 11>(11 * r, 0);
                    const auto v = x.template segment<11>(11 * r);
                    res.template segment<11>(11 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }



        inline VecX right_mul_e0_pOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 12);

            VecX res = VecX::Zero(num_cameras_ * 12);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();
                    VecX jp_x(num_obs * 4);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_pOSE(i);
                        const auto v =
                                x.template segment<12>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<4>(i * 4) = u * v;
                    }

                    const auto hll_inv = hll_inv_pOSE_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl_pOSE();
                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_pOSE(i);
                        const auto v = tmp.template segment<4>(i * 4);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<12>(pose_idx * 12) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_joint(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res = VecX::Zero(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();

                    VecX jp_x(num_obs * 2);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    //const auto& pose_indices = lb.get_pose_idx();
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space_riemannian_manifold_storage(i, bal_problem_.cameras()); //@debug: u is ok
                        const auto v =
                                x.template segment<11>(pose_indices.at(i) * 11);

                        jp_x.template segment<2>(i * 2) = u * v;
                    }
                    const auto hll_inv = hll_inv_joint_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl_homogeneous_riemannian_manifold_storage();
                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));
                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space_riemannian_manifold_storage(i, bal_problem_.cameras());
                        const auto v = tmp.template segment<2>(i * 2);
                        const size_t pose_idx = pose_indices.at(i);

                        {

                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<11>(pose_idx * 11) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

    protected:
        using Base::landmark_blocks_;
        using Base::num_cameras_;
        using Base::pose_damping_diagonal_;
        using Base::pose_mutex_;

        BalProblem<Scalar>& bal_problem_;

        const size_t m_;

        RowMatX b_inv_joint_;
        RowMatX b_inv_pOSE_;
        RowMatX hll_inv_joint_;
        RowMatX hll_inv_pOSE_;
    };

}  // namespace rootba_povar

#endif //ROOTBA_LINEARIZATION_POWER_VARPROJ_HPP
