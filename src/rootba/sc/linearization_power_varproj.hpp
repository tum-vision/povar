//
// Created by Simon on 10/07/2023.
//

#ifndef ROOTBA_LINEARIZATION_POWER_VARPROJ_HPP
#define ROOTBA_LINEARIZATION_POWER_VARPROJ_HPP

#pragma once

#include <Eigen/Dense>
#include <basalt/utils/sophus_utils.hpp>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "rootba/bal/bal_problem.hpp"
#include "rootba/cg/conjugate_gradient.hpp"
#include "rootba/sc/landmark_block.hpp"
#include "rootba/sc/linearization_varproj.hpp"
#include "rootba/util/assert.hpp"

#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

namespace rootba {

    template <typename Scalar_, int POSE_SIZE_>
    class LinearizationPowerVarproj : private LinearizationVarProj<Scalar_, POSE_SIZE_> {
    public:
        using Scalar = Scalar_;
        static constexpr int POSE_SIZE = POSE_SIZE_;
        //using Base = LinearizationVarProj<Scalar_, POSE_SIZE_>;
        using Base = LinearizationVarProj<Scalar_, 11>;
        //using Base = LinearizationVarProj<Scalar_, 15>;


        using Vec2 = Eigen::Matrix<Scalar_, 2, 1>;
        using Vec4 = Eigen::Matrix<Scalar_, 4, 1>;
        using VecX = Eigen::Matrix<Scalar_, Eigen::Dynamic, 1>;
        using MatX = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;
        using Mat36 = Eigen::Matrix<Scalar_, 3, 6>;
        using RowMatX =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using PerSolveOptions =
        typename ConjugateGradientsSolver<Scalar>::PerSolveOptions;
        using SparseMat = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

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
                  b_inv_(9 * num_cameras_, 9),
                  b_inv_joint_(11 * num_cameras_, 11),
                  b_inv_joint_RpOSE_(13 * num_cameras_, 13),
                  b_inv_projective_(15 * num_cameras_, 15),
                  b_inv_projective_riemannian_manifold_(11 * num_cameras_, 11),
                  b_inv_pOSE_(15 * num_cameras_, 15),
                  b_inv_expOSE_(15 * num_cameras_, 15),
                  b_inv_pOSE_rOSE_(15 * num_cameras_, 15),
                  b_inv_rOSE_(15 * num_cameras_, 15),
                  b_inv_RpOSE_(11 * num_cameras_, 11),
                  b_inv_RpOSE_ML_(11 * num_cameras_, 11),
                  b_inv_pOSE_riemannian_manifold_(11 * num_cameras_, 11),
                  b_inv_pOSE_homogeneous_(11 * num_cameras_, 11),
                  //b_inv_(POSE_SIZE * num_cameras_, POSE_SIZE),
                  hll_inv_(3 * landmark_blocks_.size(), 3),
                  hll_inv_joint_(3 * landmark_blocks_.size(), 3),
                  hll_inv_joint_RpOSE_(3 * landmark_blocks_.size(), 3),
                  hll_inv_pOSE_(3 * landmark_blocks_.size(), 3),
                  hll_inv_RpOSE_(3 * landmark_blocks_.size(), 3),
                  hll_inv_RpOSE_ML_(3 * landmark_blocks_.size(), 3),
                  hll_inv_expOSE_(3 * landmark_blocks_.size(), 3),
                  hll_inv_pOSE_rOSE_(3 * landmark_blocks_.size(), 3),
                  hll_inv_rOSE_(3 * landmark_blocks_.size(), 3),
                  hll_inv_pOSE_riemannian_manifold_(3 * landmark_blocks_.size(), 3),
                  hll_inv_pOSE_homogeneous_(3 * landmark_blocks_.size(), 3),
                  hll_inv_homogeneous_riemannian_manifold_(3 * landmark_blocks_.size(), 3),
                  hll_inv_homogeneous_(4 * landmark_blocks_.size(), 4){
            ROOTBA_ASSERT(pose_mutex_.size() == num_cameras_);
        }

        void solve_lm_landmark(VecX& delta_lm) {
            {
                delta_lm.setZero(4 * landmark_blocks_.size());
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        //auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
                        lb.get_lm_landmark(delta_lm, r);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }
        }

        void solve_lm_landmark_riemannian_manifold(VecX& delta_lm) {
            {
                delta_lm.setZero(4 * landmark_blocks_.size());
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        //auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
                        lb.get_lm_landmark_riemannian_manifold(delta_lm, r);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb(VecX& b_p) {
            b_inv_.setZero(9 * num_cameras_, 9);
            b_p.setZero(num_cameras_ * 9);
            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b(b_inv_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_.template block<9, 9>(9 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(9, 9));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
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

        void prepare_Hb_joint_RpOSE(VecX& b_p) {
            b_inv_joint_RpOSE_.setZero(13 * num_cameras_, 13);
            b_p.setZero(num_cameras_ * 13);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_joint_RpOSE_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_joint_RpOSE(b_inv_joint_RpOSE_, hll_inv, b_p, bal_problem_.cameras(), pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_joint_RpOSE_.template block<13, 13>(13 * r, 0);
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
                        b_inv.template block<11,11>(0,0) += Proj_pose.transpose() * pose_damping_diagonal_ * Proj_pose;
                        b_inv(11,11) += pose_damping_diagonal_ * cam.intrinsics.getParam()[1] * cam.intrinsics.getParam()[1];
                        b_inv(12,12) += pose_damping_diagonal_ * cam.intrinsics.getParam()[2] * cam.intrinsics.getParam()[2];
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(13, 13));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_affine_space(VecX& b_p) {
            b_inv_.setZero(11 * num_cameras_, 11);
            b_p.setZero(num_cameras_ * 11);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_affine_space(b_inv_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_.template block<11, 11>(11 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(11, 11));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_pOSE(VecX& b_p) {
            b_inv_pOSE_.setZero(15 * num_cameras_, 15);
            b_p.setZero(num_cameras_ * 15);

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
                                b_inv_pOSE_.template block<15, 15>(15 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(15, 15));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_RpOSE(VecX& b_p) {
            b_inv_RpOSE_.setZero(11 * num_cameras_, 11);
            b_p.setZero(num_cameras_ * 11);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_RpOSE_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_RpOSE(b_inv_RpOSE_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_RpOSE_.template block<11, 11>(11 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(11, 11));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_RpOSE_ML(VecX& b_p) {
            b_inv_RpOSE_ML_.setZero(11 * num_cameras_, 11);
            b_p.setZero(num_cameras_ * 11);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_RpOSE_ML_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_RpOSE_ML(b_inv_RpOSE_ML_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_RpOSE_ML_.template block<11, 11>(11 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(11, 11));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }


        void prepare_Hb_pOSE_poBA(VecX& b_p) {
            b_inv_pOSE_.setZero(15 * num_cameras_, 15);
            b_p.setZero(num_cameras_ * 15);

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
                                b_inv_pOSE_.template block<15, 15>(15 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(15, 15));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_expOSE(VecX& b_p) {
            b_inv_expOSE_.setZero(15 * num_cameras_, 15);
            b_p.setZero(num_cameras_ * 15);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_expOSE_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_expOSE(b_inv_expOSE_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_expOSE_.template block<15, 15>(15 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(15, 15));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_pOSE_rOSE(VecX& b_p) {
            b_inv_pOSE_rOSE_.setZero(15 * num_cameras_, 15);
            b_p.setZero(num_cameras_ * 15);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_pOSE_rOSE_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_pOSE_rOSE(b_inv_pOSE_rOSE_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_pOSE_rOSE_.template block<15, 15>(15 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(15, 15));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_rOSE(VecX& b_p) {
            b_inv_rOSE_.setZero(15 * num_cameras_, 15);
            b_p.setZero(num_cameras_ * 15);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_rOSE_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_rOSE(b_inv_rOSE_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_rOSE_.template block<15, 15>(15 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(15, 15));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_pOSE_riemannian_manifold(VecX& b_p) {
            b_inv_pOSE_riemannian_manifold_.setZero(11 * num_cameras_, 11);
            b_p.setZero(num_cameras_ * 11);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_pOSE_riemannian_manifold_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_pOSE_riemannian_manifold(b_inv_pOSE_riemannian_manifold_, hll_inv, b_p,bal_problem_.cameras(), pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_pOSE_riemannian_manifold_.template block<11, 11>(11 * r, 0);
                        const auto& cam = bal_problem_.cameras().at(r);

                        //const auto jp = storage_.template block<2, 12>(2 * obs_idx, 0);

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
                        //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
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

        void prepare_Hb_pOSE_homogeneous(VecX& b_p) {
            b_inv_pOSE_homogeneous_.setZero(11 * num_cameras_, 11);
            b_p.setZero(num_cameras_ * 11);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_pOSE_homogeneous_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_pOSE_homogeneous(b_inv_pOSE_homogeneous_, hll_inv, b_p, bal_problem_.cameras(), pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_pOSE_homogeneous_.template block<11, 11>(11 * r, 0);
                        const auto& cam = bal_problem_.cameras().at(r);

                        //const auto jp = storage_.template block<2, 12>(2 * obs_idx, 0);

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
                        //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
                        auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                        b_inv += Proj_pose.transpose() * pose_damping_diagonal_ * Proj_pose;
                        //b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(11, 11));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_projective_space(VecX& b_p) {
            b_inv_projective_.setZero(15 * num_cameras_, 15);
            b_p.setZero(num_cameras_ * 15);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_projective_space(b_inv_projective_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_projective_.template block<15, 15>(15 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(15, 15));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_projective_space_homogeneous(VecX& b_p) {
            b_inv_projective_.setZero(15 * num_cameras_, 15);
            b_p.setZero(num_cameras_ * 15);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_homogeneous_.template block<4, 4>(4 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_projective_space_homogeneous(b_inv_projective_, hll_inv, b_p, pose_mutex_);
                     }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_projective_.template block<15, 15>(15 * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(15, 15));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_projective_space_homogeneous_riemannian_manifold(VecX& b_p) {
            b_inv_projective_riemannian_manifold_.setZero(11 * num_cameras_, 11);
            b_p.setZero(num_cameras_ * 11);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_homogeneous_riemannian_manifold_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_projective_space_homogeneous_riemannian_manifold(b_inv_projective_riemannian_manifold_, hll_inv, b_p, bal_problem_.cameras(), pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_projective_riemannian_manifold_.template block<11, 11>(11 * r, 0);
                        //@Simon: try to add Proj_pose.transpose() * pose_damping_diagonal_ * Proj_pose:

                        //@Simon: TODO: check and fix if necessary
                        //const size_t cam_idx = r;
                        const auto& cam = bal_problem_.cameras().at(r);

                        //const auto jp = storage_.template block<2, 12>(2 * obs_idx, 0);

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
                        //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
                        auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                        b_inv += Proj_pose.transpose() * pose_damping_diagonal_ * Proj_pose;
                        //@Simon: initial version:
                        //b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(11, 11));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        void prepare_Hb_refine(VecX& b_p) {
            b_inv_.setZero(POSE_SIZE * num_cameras_, POSE_SIZE);
            b_p.setZero(num_cameras_ * POSE_SIZE);

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_.at(r);
                        auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
                        lb.get_Hll_inv_add_Hpp_b_refine(b_inv_, hll_inv, b_p, pose_mutex_);
                    }
                };

                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);
            }

            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        auto b_inv =
                                b_inv_.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * r, 0);
                        b_inv.diagonal().array() += pose_damping_diagonal_;
                        b_inv = b_inv.template selfadjointView<Eigen::Upper>().llt().solve(
                                MatX::Identity(POSE_SIZE, POSE_SIZE));
                    }
                };

                tbb::blocked_range<size_t> range(0, num_cameras_);
                tbb::parallel_for(range, body);
            }
        }

        Summary solve(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 9 * num_cameras_);

            Summary summary;

            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv(right_mul_e0(tmp));
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

        Summary solve_affine_space(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 11 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_affine_space(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_affine_space(right_mul_e0_affine_space(tmp));
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

        Summary solve_projective_space(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 15 * num_cameras_);

            Summary summary;

            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_projective_space(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_projective_space(right_mul_e0_projective_space(tmp));
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

        Summary solve_pOSE(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 15 * num_cameras_);

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

        Summary solve_RpOSE(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 11 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_RpOSE(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_RpOSE(right_mul_e0_RpOSE(tmp));
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

        Summary solve_RpOSE_ML(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 11 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            //std::cout << "in solve    b_p = " << b_p.norm() << "\n";
            accum = right_mul_b_inv_RpOSE_ML(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    //std::cout << "right_mul_e0_RpOSE_ML(tmp).norm() = " << right_mul_e0_RpOSE_ML(tmp).norm() << "\n";
                    tmp = right_mul_b_inv_RpOSE_ML(right_mul_e0_RpOSE_ML(tmp));
                    accum += tmp;
                    //std::cout << "tmp.norm() = " << tmp.norm() << "\n";

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

        Summary solve_expOSE(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 15 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_expOSE(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_expOSE(right_mul_e0_expOSE(tmp));
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

        Summary solve_pOSE_rOSE(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 15 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_pOSE_rOSE(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_pOSE_rOSE(right_mul_e0_pOSE_rOSE(tmp));
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


        Summary solve_rOSE(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 15 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_rOSE(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_rOSE(right_mul_e0_rOSE(tmp));
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

        Summary solve_pOSE_riemannian_manifold(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 11 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_pOSE_riemannian_manifold(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_pOSE_riemannian_manifold(right_mul_e0_pOSE_riemannian_manifold(tmp));
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

        Summary solve_pOSE_homogeneous(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 11 * num_cameras_);

            Summary summary;
            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_pOSE_homogeneous(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_pOSE_homogeneous(right_mul_e0_pOSE_homogeneous(tmp));
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

        Summary solve_projective_space_homogeneous(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 15 * num_cameras_);

            Summary summary;

            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_projective_space(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_projective_space(right_mul_e0_projective_space_homogeneous(tmp));
                    accum += tmp;
                    //
                    //

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

        Summary solve_projective_space_homogeneous_riemannian_manifold(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 11 * num_cameras_);

            Summary summary;

            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_projective_space_riemannian_manifold(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_projective_space_riemannian_manifold(right_mul_e0_projective_space_homogeneous_riemannian_manifold(tmp));
                    accum += tmp;
                    //
                    //

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

        Summary solve_projective_space_homogeneous_riemannian_manifold_storage(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 11 * num_cameras_);

            Summary summary;

            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_projective_space_riemannian_manifold(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_projective_space_riemannian_manifold(right_mul_e0_projective_space_homogeneous_riemannian_manifold_storage(tmp));
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

        Summary solve_joint_RpOSE(const VecX& b_p, VecX& accum, PerSolveOptions& pso) const {
            ROOTBA_ASSERT(static_cast<size_t>(b_p.size()) == 13 * num_cameras_);

            Summary summary;

            // - (B^-1 * E_0)^i * B^-1 * g
            accum = right_mul_b_inv_joint_RpOSE(-b_p);
            if (m_ > 0) {
                const Scalar norm_0 = pso.r_tolerance > 0 ? accum.norm() : 0;

                VecX tmp = accum;
                for (size_t i = 1; i <= m_; ++i) {
                    tmp = right_mul_b_inv_joint_RpOSE(right_mul_e0_joint_RpOSE(tmp));
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



        Scalar landmark_closed_form(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * POSE_SIZE));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_affine_space(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_affine_space(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_pOSE(int alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_pOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_RpOSE(double alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_RpOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_RpOSE_refinement(double alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_RpOSE_refinement(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_RpOSE_ML(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_RpOSE_ML(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }



        void update_y_tilde_expose() {

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].update_y_tilde_expose(bal_problem_.cameras());
                }
                //return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void initialize_y_tilde_expose() {

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].update_y_tilde_expose_initialize(bal_problem_.cameras());
                }
                //return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void rpose_new_equilibrium() {

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].rpose_new_equilibrium(bal_problem_.cameras());
                }
                //return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }


        Scalar landmark_closed_form_expOSE(int alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_expOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_pOSE_rOSE(int alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_pOSE_rOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_rOSE(int alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_rOSE(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }


        Scalar landmark_closed_form_pOSE_riemannian_manifold(int alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 12));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_pOSE_riemannian_manifold(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_pOSE_homogeneous(int alpha, const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 12));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_pOSE_homogeneous(alpha, pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_projective_space(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_projective_space(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_projective_space_homogeneous_riemannian_manifold(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_projective_space_homogeneous_riemannian_manifold(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar landmark_closed_form_projective_space_homogeneous(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_projective_space_homogeneous(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        void landmark_closed_form_projective_space_homogeneous_nonlinear_initialization() {

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_projective_space_homogeneous_nonlinear_initialization(bal_problem_.cameras());
                }
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void landmark_closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold() {

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold(bal_problem_.cameras());
                }
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        Scalar landmark_closed_form_projective_space_homogeneous_lm_landmark(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_projective_space_homogeneous(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }


        void print_block(const std::string& filename, size_t block_idx) {
            landmark_blocks_[block_idx].printStorage(filename);
        }

        // For debugging only
        inline auto get_b_inv(const size_t cam_idx) const {
            return b_inv_.template block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * cam_idx, 0);
        }

        // For debugging only
        VecX right_multiply(const VecX& x) const {
            const auto jacobi = Base::get_jacobi();

            VecX res(POSE_SIZE * num_cameras_);
            for (size_t i = 0; i < num_cameras_; ++i) {
                const auto u = jacobi.block_diagonal.at(std::make_pair(i, i));
                const auto v = x.template segment<POSE_SIZE>(POSE_SIZE * i);
                res.template segment<POSE_SIZE>(POSE_SIZE * i) = u * v;
            }

            res -= right_mul_e0(x);
            return res;
        }

        // make selected base class methods public
        using Base::back_substitute;
        using Base::back_substitute_poBA;
        using Base::back_substitute_joint;
        using Base::back_substitute_joint_RpOSE;
        using Base::compute_Jp_scale_and_scale_Jp_cols;
        using Base::get_Jp_diag2;
        using Base::get_Jp_diag2_pOSE;
        using Base::get_Jp_diag2_RpOSE;
        using Base::get_Jp_diag2_RpOSE_ML;
        using Base::get_Jp_diag2_expOSE;
        using Base::get_Jp_diag2_pOSE_rOSE;
        using Base::get_Jp_diag2_rOSE;
        using Base::get_Jp_diag2_pOSE_homogeneous;
        using Base::get_Jp_diag2_affine_space;
        using Base::get_Jp_diag2_projective_space;
        using Base::get_Jp_diag2_projective_space_RpOSE;
        using Base::linearize_problem;
        using Base::linearize_problem_refine;
        using Base::num_cols_reduced;
        using Base::scale_Jl_cols;
        using Base::scale_Jl_cols_pOSE;
        using Base::scale_Jl_cols_RpOSE;
        using Base::scale_Jl_cols_RpOSE_ML;
        using Base::scale_Jl_cols_expOSE;
        using Base::scale_Jl_cols_pOSE_homogeneous;
        using Base::scale_Jl_cols_homogeneous;
        using Base::scale_Jl_cols_homogeneous_RpOSE;
        using Base::scale_Jp_cols;
        using Base::scale_Jp_cols_joint;
        using Base::scale_Jp_cols_affine;
        using Base::scale_Jp_cols_pOSE;
        using Base::scale_Jp_cols_RpOSE;
        using Base::scale_Jp_cols_RpOSE_ML;
        using Base::scale_Jp_cols_expOSE;
        using Base::scale_Jp_cols_pOSE_rOSE;
        using Base::scale_Jp_cols_rOSE;
        using Base::scale_Jp_cols_pOSE_homogeneous;
        using Base::set_landmark_damping;
        using Base::set_landmark_damping_joint;
        using Base::set_pose_damping;
        using Base::linearize_problem_affine_space;
        using Base::linearize_problem_pOSE;
        using Base::linearize_problem_RpOSE;
        using Base::linearize_problem_RpOSE_ML;
        using Base::linearize_problem_RpOSE_refinement;
        using Base::linearize_problem_expOSE;
        using Base::linearize_problem_pOSE_rOSE;
        using Base::linearize_problem_rOSE;
        using Base::linearize_problem_pOSE_homogeneous;
        using Base::linearize_problem_projective_space;
        using Base::linearize_problem_projective_space_homogeneous;
        using Base::linearize_problem_projective_space_homogeneous_RpOSE;
        using Base::linearize_problem_projective_space_homogeneous_storage;
        using Base::linearize_nullspace;
        using Vec12 = Eigen::Matrix<Scalar, 12, 1>;

        inline VecX right_mul_b_inv(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 9);

            VecX res(num_cameras_ * 9);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_.template block<9, 9>(9 * r, 0);
                    const auto v = x.template segment<9>(9 * r);
                    res.template segment<9>(9 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_affine_space(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_.template block<11, 11>(11 * r, 0);
                    const auto v = x.template segment<11>(11 * r);
                    res.template segment<11>(11 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_pOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res(num_cameras_ * 15);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_pOSE_.template block<15, 15>(15 * r, 0);
                    const auto v = x.template segment<15>(15 * r);
                    res.template segment<15>(15 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_RpOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_RpOSE_.template block<11, 11>(11 * r, 0);
                    const auto v = x.template segment<11>(11 * r);
                    res.template segment<11>(11 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_RpOSE_ML(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_RpOSE_ML_.template block<11, 11>(11 * r, 0);
                    //std::cout << "in right_mul   u = " << u.norm() << "\n";
                    const auto v = x.template segment<11>(11 * r);
                    res.template segment<11>(11 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }


        inline VecX right_mul_b_inv_expOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res(num_cameras_ * 15);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_expOSE_.template block<15, 15>(15 * r, 0);
                    const auto v = x.template segment<15>(15 * r);
                    res.template segment<15>(15 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }


        inline VecX right_mul_b_inv_pOSE_rOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res(num_cameras_ * 15);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_pOSE_rOSE_.template block<15, 15>(15 * r, 0);
                    const auto v = x.template segment<15>(15 * r);
                    res.template segment<15>(15 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }


        inline VecX right_mul_b_inv_rOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res(num_cameras_ * 15);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_rOSE_.template block<15, 15>(15 * r, 0);
                    const auto v = x.template segment<15>(15 * r);
                    res.template segment<15>(15 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_pOSE_riemannian_manifold(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_pOSE_riemannian_manifold_.template block<11, 11>(11 * r, 0);
                    const auto v = x.template segment<11>(11 * r);
                    res.template segment<11>(11 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_pOSE_homogeneous(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_pOSE_homogeneous_.template block<11, 11>(11 * r, 0);
                    const auto v = x.template segment<11>(11 * r);
                    res.template segment<11>(11 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_projective_space(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res(num_cameras_ * 15);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_projective_.template block<15, 15>(15 * r, 0);
                    const auto v = x.template segment<15>(15 * r);
                    res.template segment<15>(15 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_b_inv_projective_space_riemannian_manifold(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_projective_riemannian_manifold_.template block<11, 11>(11 * r, 0);
                    const auto v = x.template segment<11>(11 * r);
                    res.template segment<11>(11 * r) = u * v;
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

        inline VecX right_mul_b_inv_joint_RpOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 13);

            VecX res(num_cameras_ * 13);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto u =
                            b_inv_joint_RpOSE_.template block<13, 13>(13 * r, 0);
                    const auto v = x.template segment<13>(13 * r);
                    res.template segment<13>(13 * r) = u * v;
                }
            };

            tbb::blocked_range<size_t> range(0, num_cameras_);
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 9);

            VecX res = VecX::Zero(num_cameras_ * 9);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();

                    VecX jp_x(num_obs * 2);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi(i);
                        const auto v =
                                x.template segment<9>(pose_indices.at(i) * 9);
                        jp_x.template segment<2>(i * 2) = u * v;
                    }

                    const auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl();

                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi(i);
                        const auto v = tmp.template segment<2>(i * 2);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<9>(pose_idx * 9) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_affine_space(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res = VecX::Zero(num_cameras_ * 11);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();

                    VecX jp_x(num_obs * 2);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_affine_space(i);
                        const auto v =
                                x.template segment<11>(pose_indices.at(i) * 11);
                        jp_x.template segment<2>(i * 2) = u * v;
                    }

                    const auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);

                    const auto jl = lb.get_Jl_affine_space();

                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_affine_space(i);
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

        inline VecX right_mul_e0_pOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res = VecX::Zero(num_cameras_ * 15);
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
                                x.template segment<15>(pose_indices.at(i) * POSE_SIZE);
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
                            res.template segment<15>(pose_idx * 15) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_RpOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res = VecX::Zero(num_cameras_ * 11);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();
                    VecX jp_x(num_obs * 3);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_RpOSE(i);
                        const auto v =
                                x.template segment<11>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<3>(i * 3) = u * v;
                    }

                    const auto hll_inv = hll_inv_RpOSE_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl_RpOSE();
                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_RpOSE(i);
                        const auto v = tmp.template segment<3>(i * 3);
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

        inline VecX right_mul_e0_RpOSE_ML(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res = VecX::Zero(num_cameras_ * 11);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();
                    VecX jp_x(num_obs);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_RpOSE_ML(i);

                        const auto v =
                                x.template segment<11>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<1>(i) = u * v;
                    }

                    const auto hll_inv = hll_inv_RpOSE_ML_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl_RpOSE_ML();
                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_RpOSE_ML(i);
                        const auto v = tmp.template segment<1>(i);
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

        inline VecX right_mul_e0_expOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res = VecX::Zero(num_cameras_ * 15);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();
                    VecX jp_x(num_obs * 3);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_expOSE(i);
                        const auto v =
                                x.template segment<15>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<3>(i * 3) = u * v;
                    }

                    const auto hll_inv = hll_inv_expOSE_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl_expOSE();
                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_expOSE(i);
                        const auto v = tmp.template segment<3>(i * 3);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<15>(pose_idx * 15) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_pOSE_rOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res = VecX::Zero(num_cameras_ * 15);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();
                    VecX jp_x(num_obs * 5);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_pOSE_rOSE(i);
                        const auto v =
                                x.template segment<15>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<5>(i * 5) = u * v;
                    }

                    const auto hll_inv = hll_inv_pOSE_rOSE_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl_pOSE_rOSE();
                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_pOSE_rOSE(i);
                        const auto v = tmp.template segment<5>(i * 5);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<15>(pose_idx * 15) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_rOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res = VecX::Zero(num_cameras_ * 15);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();
                    VecX jp_x(num_obs * 3);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_rOSE(i);
                        const auto v =
                                x.template segment<15>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<3>(i * 3) = u * v;
                    }

                    const auto hll_inv = hll_inv_rOSE_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl_rOSE();
                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_rOSE(i);
                        const auto v = tmp.template segment<3>(i * 3);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<15>(pose_idx * 15) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }


        inline VecX right_mul_e0_pOSE_riemannian_manifold(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res = VecX::Zero(num_cameras_ * 11);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();
                    VecX jp_x(num_obs * 4);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_pOSE_riemannian_manifold(i);
                        const size_t cam_idx = pose_indices[i];
                        const auto& cam = bal_problem_.cameras().at(cam_idx);
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
                        const auto v =
                                x.template segment<11>(pose_indices.at(i) * 11);
                        jp_x.template segment<4>(i * 4) = (u*Proj_pose) * v;
                    }

                    const auto hll_inv = hll_inv_pOSE_riemannian_manifold_.template block<3, 3>(3 * r, 0);
                    //auto lm_ptr_ = lb.get_lm_ptr();
                    //auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());

                    const auto jl = lb.get_Jl_pOSE();

                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_pOSE_riemannian_manifold(i);
                        const auto v = tmp.template segment<4>(i * 4);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            const size_t cam_idx = pose_indices[i];
                            const auto& cam = bal_problem_.cameras().at(cam_idx);
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
                            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
                            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<11>(pose_idx * 11) +=
                                    (u * Proj_pose).transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_pOSE_homogeneous(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 11);

            VecX res = VecX::Zero(num_cameras_ * 11);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();
                    VecX jp_x(num_obs * 4);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_pOSE_homogeneous(i);
                        const size_t cam_idx = pose_indices[i];
                        const auto& cam = bal_problem_.cameras().at(cam_idx);

                        //const auto jp = storage_.template block<2, 12>(2 * obs_idx, 0);

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
                        //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
                        auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                        const auto v =
                                x.template segment<11>(pose_indices.at(i) * 11);
                        jp_x.template segment<4>(i * 4) = (u * Proj_pose) * v;
                    }
                    const auto hll_inv = hll_inv_pOSE_homogeneous_.template block<3, 3>(3 * r, 0);
                    auto lm_ptr_ = lb.get_lm_ptr();
                    auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
                    const auto jl = lb.get_Jl_pOSE_homogeneous();
                    const VecX tmp = (jl * Proj) * (hll_inv * ((jl * Proj).transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_pOSE_homogeneous(i);
                        const auto v = tmp.template segment<4>(i * 4);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            const size_t cam_idx = pose_indices[i];
                            const auto& cam = bal_problem_.cameras().at(cam_idx);
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
                            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
                            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<11>(pose_idx * 11) +=
                                    (u * Proj_pose).transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_projective_space(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res = VecX::Zero(num_cameras_ * 15);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();

                    VecX jp_x(num_obs * 2);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space(i);
                        const auto v =
                                x.template segment<15>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<2>(i * 2) = u * v;
                    }

                    const auto hll_inv = hll_inv_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl();

                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space(i);
                        const auto v = tmp.template segment<2>(i * 2);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<15>(pose_idx * 15) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_projective_space_homogeneous(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 15);

            VecX res = VecX::Zero(num_cameras_ * 15);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();

                    VecX jp_x(num_obs * 2);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space(i); //@debug: u is ok
                        const auto v =
                                x.template segment<15>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<2>(i * 2) = u * v;
                    }

                    const auto hll_inv = hll_inv_homogeneous_.template block<4, 4>(4 * r, 0);
                    const auto jl = lb.get_Jl_homogeneous();

                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));

                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space(i);
                        const auto v = tmp.template segment<2>(i * 2);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<15>(pose_idx * 15) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_projective_space_homogeneous_riemannian_manifold(const VecX& x) const {
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
                        const auto u = lb.get_Jpi_projective_space_riemannian_manifold(i, bal_problem_.cameras()); //@debug: u is ok
                        const size_t cam_idx = pose_indices[i];
                        const auto& cam = bal_problem_.cameras().at(cam_idx);

                        //const auto jp = storage_.template block<2, 12>(2 * obs_idx, 0);

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
                        //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
                        auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                        const auto v =
                                x.template segment<11>(pose_indices.at(i) * 11);
                        //const auto v =
                        //        x.template segment<11>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<2>(i * 2) = (u * Proj_pose) * v;
                    }
                    const auto hll_inv = hll_inv_homogeneous_riemannian_manifold_.template block<3, 3>(3 * r, 0);
                    auto lm_ptr_ = lb.get_lm_ptr();
                    auto Proj = BalBundleAdjustmentHelper<Scalar>::kernel_COD((lm_ptr_->p_w_homogeneous).transpose());
                    const auto jl = lb.get_Jl_homogeneous_riemannian_manifold();
                    const VecX tmp = (jl * Proj) * (hll_inv * ((jl * Proj).transpose() * jp_x));
                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space_riemannian_manifold(i, bal_problem_.cameras());
                        const auto v = tmp.template segment<2>(i * 2);
                        const size_t pose_idx = pose_indices.at(i);

                        {
                            const size_t cam_idx = pose_indices[i];
                            const auto& cam = bal_problem_.cameras().at(cam_idx);
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
                            //camera_space_matrix << cam.space_matrix.row(0), cam.space_matrix.row(1), cam.space_matrix.row(2);
                            auto Proj_pose = BalBundleAdjustmentHelper<Scalar>::kernel_COD(camera_space_matrix.transpose());
                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<11>(pose_idx * 11) +=
                                    (u * Proj_pose).transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }

        inline VecX right_mul_e0_projective_space_homogeneous_riemannian_manifold_storage(const VecX& x) const {
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
                        //const auto v =
                        //        x.template segment<11>(pose_indices.at(i) * POSE_SIZE);
                        jp_x.template segment<2>(i * 2) = u * v;
                    }
                    const auto hll_inv = hll_inv_homogeneous_riemannian_manifold_.template block<3, 3>(3 * r, 0);
                    //auto lm_ptr_ = lb.get_lm_ptr();
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

        inline VecX right_mul_e0_joint_RpOSE(const VecX& x) const {
            ROOTBA_ASSERT(static_cast<size_t>(x.size()) == num_cameras_ * 13);

            VecX res = VecX::Zero(num_cameras_ * 13);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_.at(r);

                    const auto& pose_indices = lb.get_pose_idx();
                    const size_t num_obs = pose_indices.size();

                    VecX jp_x(num_obs * 2);
                    // TODO@demmeln(LOW, Niko): create Jp_x method for lmb
                    //const auto& pose_indices = lb.get_pose_idx();
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space_riemannian_manifold_storage_RpOSE(i, bal_problem_.cameras()); //@debug: u is ok
                        const auto v =
                                x.template segment<13>(pose_indices.at(i) * 13);

                        jp_x.template segment<2>(i * 2) = u * v;
                    }
                    const auto hll_inv = hll_inv_joint_RpOSE_.template block<3, 3>(3 * r, 0);
                    const auto jl = lb.get_Jl_homogeneous_riemannian_manifold_storage_RpOSE();
                    const VecX tmp = jl * (hll_inv * (jl.transpose() * jp_x));
                    // TODO@demmeln(LOW, Niko): create JpT_x method for lmb
                    for (size_t i = 0; i < num_obs; ++i) {
                        const auto u = lb.get_Jpi_projective_space_riemannian_manifold_storage_RpOSE(i, bal_problem_.cameras());
                        const auto v = tmp.template segment<2>(i * 2);
                        const size_t pose_idx = pose_indices.at(i);

                        {

                            std::scoped_lock lock(pose_mutex_.at(pose_idx));
                            res.template segment<13>(pose_idx * 13) +=
                                    u.transpose() * v;
                        }
                    }
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            return res;
        }


        MatX get_M() const {

            SparseMat Jp = get_sparse_Jp();
            SparseMat Jl = get_sparse_Jl();
            SparseMat Hpp_inv = get_sparse_Hpp_inv();
            SparseMat Hll_inv = get_sparse_Hll_inv();

            MatX R = Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;

            return R;
        }

        MatX get_inverted_matrix(int & num_iteration) const {

            SparseMat Jp = get_sparse_Jp();
            SparseMat Jl = get_sparse_Jl();

            SparseMat Hpp_inv = get_sparse_Hpp_inv();
            SparseMat Hll_inv = get_sparse_Hll_inv();

            MatX accm = Eigen::Matrix<Scalar_,-1,-1>::Identity(Hpp_inv.rows(),Hpp_inv.cols());
            MatX tmp = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols());

            MatX sum = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols()) - Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;
            for (size_t i = 1; i <= num_iteration; ++i) {
                tmp = Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp * tmp;
                accm += tmp;
            }

            return sum.inverse() - accm;

        }

        MatX get_R(int & num_iteration) const {

            SparseMat Jp = get_sparse_Jp();
            SparseMat Jl = get_sparse_Jl();

            SparseMat Hpp_inv = get_sparse_Hpp_inv();
            SparseMat Hll_inv = get_sparse_Hll_inv();

            MatX accm = Eigen::Matrix<Scalar_,-1,-1>::Identity(Hpp_inv.rows(),Hpp_inv.cols());
            MatX tmp = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols());

            MatX sum = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols()) - Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;
            for (size_t i = 1; i <= num_iteration; ++i) {
                tmp = Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp * tmp;
                accm += tmp;
            }

            return (sum.inverse() - accm)*Hpp_inv;

        }

        MatX get_power_S(int & num_iteration) const {

            SparseMat Jp = get_sparse_Jp();
            SparseMat Jl = get_sparse_Jl();

            SparseMat Hpp_inv = get_sparse_Hpp_inv();
            SparseMat Hll_inv = get_sparse_Hll_inv();

            MatX accm = Eigen::Matrix<Scalar_,-1,-1>::Identity(Hpp_inv.rows(),Hpp_inv.cols());
            MatX tmp = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols());

            //MatX sum = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols()) - Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;
            for (size_t i = 1; i <= num_iteration; ++i) {
                tmp = Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp * tmp;
                accm += tmp;
            }

            return accm * Hpp_inv;

        }

        MatX get_S() const {

            SparseMat Jp = get_sparse_Jp();
            SparseMat Jl = get_sparse_Jl();

            SparseMat Hpp_inv = get_sparse_Hpp_inv();
            SparseMat Hll_inv = get_sparse_Hll_inv();

            MatX accm = Eigen::Matrix<Scalar_,-1,-1>::Identity(Hpp_inv.rows(),Hpp_inv.cols());
            MatX tmp = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols());

            MatX sum = Eigen::Matrix<Scalar_,-1,-1>::Identity(accm.rows(),accm.cols()) - Hpp_inv * Jp.transpose() * Jl * Hll_inv * Jl.transpose() * Jp;

            return sum.inverse() * Hpp_inv;

        }

        // For debugging only
        // TODO: refactor to avoid duplicating with similar methods in implicitSC
        SparseMat get_sparse_Jp() const {
            std::vector<size_t> lm_obs_indices;
            size_t num_obs = 0;
            lm_obs_indices.reserve(landmark_blocks_.size());
            for (const auto& lm_block : landmark_blocks_) {
                lm_obs_indices.push_back(num_obs);
                num_obs += lm_block.num_poses();
            }

            std::vector<Eigen::Triplet<Scalar>> triplets;
            triplets.reserve(num_obs * 2 * POSE_SIZE);

            for (size_t lm_idx = 0; lm_idx < landmark_blocks_.size(); ++lm_idx) {
                const auto& lb = landmark_blocks_[lm_idx];
                const size_t row_idx = lm_obs_indices[lm_idx] * 2;

                const auto& pose_indices = lb.get_pose_idx();
                for (size_t j = 0; j < pose_indices.size(); ++j) {
                    const auto block = lb.get_Jpi(j);
                    const size_t col_idx = pose_indices[j];

                    const size_t row_offset = row_idx + j * 2;
                    const size_t col_offset = col_idx * POSE_SIZE;

                    for (Eigen::Index row = 0; row < 2; ++row) {
                        for (Eigen::Index col = 0; col < POSE_SIZE; ++col) {
                            triplets.emplace_back(row + row_offset, col + col_offset,
                                                  block(row, col));
                        }
                    }
                }
            }

            // build sparse matrix
            SparseMat res(num_obs * 2, num_cameras_ * POSE_SIZE);
            if (!triplets.empty()) {
                res.setFromTriplets(triplets.begin(), triplets.end());
            }

            return res;
        }

        // For debugging only
        // TODO: refactor to avoid duplicating with similar methods in implicitSC
        SparseMat get_sparse_Jl() const {
            std::vector<size_t> lm_obs_indices;
            size_t num_obs = 0;
            lm_obs_indices.reserve(landmark_blocks_.size());
            for (const auto& lm_block : landmark_blocks_) {
                lm_obs_indices.push_back(num_obs);
                num_obs += lm_block.num_poses();
            }

            std::vector<Eigen::Triplet<Scalar>> triplets;
            triplets.reserve(num_obs * 2 * 3);

            for (size_t lm_idx = 0; lm_idx < landmark_blocks_.size(); ++lm_idx) {
                const auto& lb = landmark_blocks_[lm_idx];


                const size_t row_idx = lm_obs_indices[lm_idx] * 2;
                const size_t col_offset = lm_idx * 3;

                const size_t num_lm_obs = lb.get_pose_idx().size();
                for (size_t j = 0; j < num_lm_obs; ++j) {
                    const auto block = lb.get_Jli(j);
                    const size_t row_offset = row_idx + j * 2;

                    for (Eigen::Index row = 0; row < 2; ++row) {
                        for (Eigen::Index col = 0; col < 3; ++col) {
                            triplets.emplace_back(row + row_offset, col + col_offset,
                                                  block(row, col));
                        }
                    }
                }
            }

            // build sparse matrix
            SparseMat res(num_obs * 2, landmark_blocks_.size() * 3);
            if (!triplets.empty()) {
                res.setFromTriplets(triplets.begin(), triplets.end());
            }

            return res;
        }



        // TODO: refactor to avoid duplicating with similar methods in implicitSC
        SparseMat get_sparse_Hll_inv() const {
            BlockSparseMatrix<Scalar, 3> Hll_inv(3 * landmark_blocks_.size(),
                                                 3 * landmark_blocks_.size());
            for (size_t i = 0; i < landmark_blocks_.size(); ++i) {
                //const auto& lb = landmark_blocks_[i];
                Eigen::Matrix<Scalar, 3, 3> m = hll_inv_.template block<3, 3>(3 * i, 0);
                //Eigen::Matrix<Scalar, 3, 3> m = lb.get_Hll_inv();
                Hll_inv.add(i, i, std::move(m));
            }
            return Hll_inv.get_sparse_matrix();
        }

        /*// TODO: refactor to avoid duplicating with similar methods in implicitSC
        SparseMat get_sparse_Hll_inv() const {
          BlockSparseMatrix<Scalar, 3> Hll_inv(3 * landmark_blocks_.size(),
                                               3 * landmark_blocks_.size());
          for (size_t i = 0; i < landmark_blocks_.size(); ++i) {
            const auto& lb = landmark_blocks_[i];
            Eigen::Matrix<Scalar, 3, 3> m = hll_inv_.template block<3, 3>(3 * i, 0);
            //Eigen::Matrix<Scalar, 3, 3> m = lb.get_Hll_inv();
            Hll_inv.add(i, i, std::move(m));
          }

          return Hll_inv.get_sparse_matrix();
          //return hll_inv_.get_sparse_matrix();
        }*/

        // TODO: refactor to avoid duplicating with similar methods in implicitSC
        SparseMat get_sparse_Hpp_inv() const {
            BlockDiagonalAccumulator<Scalar> Hpp;
            //RowMatX Hpp;
            {
                auto body = [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks_[r];
                        lb.add_Hpp(Hpp, pose_mutex_);
                    }
                };
                tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
                tbb::parallel_for(range, body);

                if (pose_damping_diagonal_ > 0) {
                    Hpp.add_diag(
                            num_cameras_, POSE_SIZE,
                            VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_));
                }
            }

            BlockSparseMatrix<Scalar, POSE_SIZE> Hpp_inv(num_cameras_ * POSE_SIZE,
                                                         num_cameras_ * POSE_SIZE);
            {
                auto body = [&](const typename IndexedBlocks<Scalar>::range_type& range) {
                    for (const auto& [idx, block] : range) {
                        Eigen::Matrix<Scalar, POSE_SIZE, POSE_SIZE> inv_block =
                                block.inverse();
                        Hpp_inv.add(idx.first, idx.second, std::move(inv_block));
                    }
                };
                tbb::parallel_for(Hpp.block_diagonal.range(), body);
            }

            return Hpp_inv.get_sparse_matrix();
        }

    protected:
        using Base::landmark_blocks_;
        //using Base::landmark_blocks_affine_;
        using Base::num_cameras_;
        using Base::pose_damping_diagonal_;
        using Base::pose_mutex_;

        BalProblem<Scalar>& bal_problem_;

        const size_t m_;
        // TODO@demmeln(LOW, Tin): Don't call it b_inv. b is already used for RCS
        // gradient. I know in Simon's notes it's called B, but in this code base
        // `Hpp_inv_` probably makes more sense.
        RowMatX b_inv_;
        RowMatX b_inv_joint_;
        RowMatX b_inv_joint_RpOSE_;
        RowMatX b_inv_projective_;
        RowMatX b_inv_projective_riemannian_manifold_;
        RowMatX b_inv_pOSE_;
        RowMatX b_inv_RpOSE_;
        RowMatX b_inv_RpOSE_ML_;
        RowMatX b_inv_expOSE_;
        RowMatX b_inv_pOSE_rOSE_;
        RowMatX b_inv_rOSE_;
        RowMatX b_inv_pOSE_riemannian_manifold_;
        RowMatX b_inv_pOSE_homogeneous_;
        RowMatX hll_inv_;
        RowMatX hll_inv_joint_;
        RowMatX hll_inv_joint_RpOSE_;
        RowMatX hll_inv_homogeneous_;
        RowMatX hll_inv_homogeneous_riemannian_manifold_;
        RowMatX hll_inv_pOSE_;
        RowMatX hll_inv_RpOSE_;
        RowMatX hll_inv_RpOSE_ML_;
        RowMatX hll_inv_expOSE_;
        RowMatX hll_inv_pOSE_rOSE_;
        RowMatX hll_inv_rOSE_;
        RowMatX hll_inv_pOSE_riemannian_manifold_;
        RowMatX hll_inv_pOSE_homogeneous_;
    };

}  // namespace rootba

#endif //ROOTBA_LINEARIZATION_POWER_VARPROJ_HPP
