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
#include "rootba_povar/bal/bal_bundle_adjustment_helper.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

#include <cmath>


#include <Eigen/Dense>
#include <Eigen/QR>

namespace rootba_povar {

template <typename Scalar>
std::tuple<Scalar, Scalar>
BalBundleAdjustmentHelper<Scalar>::compute_error_weight(
    const BalResidualOptions& options, Scalar res_squared) {

  // Note: Definition of cost is 0.5 ||r(x)||^2 to be in line with ceres

  switch (options.robust_norm) {
    case BalResidualOptions::RobustNorm::HUBER: {
      const Scalar thresh = options.huber_parameter;
      const Scalar huber_weight =
          res_squared < thresh * thresh ? 1.0 : thresh / std::sqrt(res_squared);
      const Scalar error =
          0.5 * (2 - huber_weight) * huber_weight * res_squared;
      return {error, huber_weight};
    }
    case BalResidualOptions::RobustNorm::CAUCHY: {
        return {log(1.0 + res_squared), 1.0};
    }
    case BalResidualOptions::RobustNorm::NONE:
      return {0.5 * res_squared, 1.0};
    default:
      LOG(FATAL) << "unreachable";
  }
}
    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm_pOSE(Scalar alpha,
            BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const auto& lm = bal_problem.landmarks().at(r);
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    bal_problem.landmarks().at(r).p_w += initialize_varproj_pOSE(alpha, obs.pos, cam.space_matrix);
                }
            }
        };
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }


    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::setzeros_varproj_lm(
            BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const auto& lm = bal_problem.landmarks().at(r);
                bal_problem.landmarks().at(r).p_w.setZero();
            }
        };

        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::compute_error_pOSE(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    VecR_pOSE res;
                    const bool projection_valid =
                            linearize_point_pOSE(options.alpha, obs.pos, lm.p_w, cam.space_matrix, cam.intrinsics,
                                                 ignore_validity_check, res);
                    const bool numerically_valid = res.array().isFinite().all();

                    const Scalar res_squared = res.squaredNorm();
                    const auto [weighted_error, weight] =
                    compute_error_weight(options.residual, res_squared);
                    error_accu.add(numerically_valid, projection_valid,
                                   std::sqrt(res_squared), weighted_error);
                }
            }

            return error_accu;
        };

        // go over all host frames
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        ResidualInfoAccu error_accu = tbb::parallel_reduce(
                range, ResidualInfoAccu(), body, ResidualInfoAccu::join);

        // output accumulated error
        error = error_accu.info;
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::compute_error_projective_space_homogeneous(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    VecR res;
                    const bool projection_valid =
                            linearize_point_projective_space_homogeneous(obs.pos, lm.p_w_homogeneous, cam.space_matrix, cam.intrinsics,
                                                             ignore_validity_check, res);


                    const bool numerically_valid = res.array().isFinite().all();

                    const Scalar res_squared = res.squaredNorm();
                    const auto [weighted_error, weight] =
                    compute_error_weight(options.residual, res_squared);
                    error_accu.add(numerically_valid, projection_valid,
                                   std::sqrt(res_squared), weighted_error);
                }
            }

            return error_accu;
        };

        // go over all host frames
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        ResidualInfoAccu error_accu = tbb::parallel_reduce(
                range, ResidualInfoAccu(), body, ResidualInfoAccu::join);

        // output accumulated error
        error = error_accu.info;
    }


    // to apply Riemannian manifold optimization
    template <typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> BalBundleAdjustmentHelper<Scalar>::kernel_COD(
            const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M) {
        Eigen::CompleteOrthogonalDecomposition<
                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
                cod;
        cod.compute(M);
        unsigned rk = cod.rank();
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> P =
                cod.colsPermutation();
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V =
                cod.matrixZ().transpose();
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Kernel =
                P * V.block(0, rk, V.rows(), V.cols() - rk);
        return Kernel;
    }

// initialization of VarProj: we derive v*(u0) = (G(u)^T G(u))^-1 G(u)^T obs ,
// in line with "Revisiting the Variable Projection Method for Separable Nonlinear Least Squares Problems" (Hong et al., CVPR 2017)
    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_pOSE(Scalar alpha,
            const Vec2& obs, const Mat34& T_c_w) {
        Mat4 T_c_w_mat;

        T_c_w_mat.row(0) = sqrt(1.0 - alpha) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt(1.0 - alpha) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(alpha) * T_c_w.row(0);
        T_c_w_mat.row(3) = sqrt(alpha) * T_c_w.row(1);

        Eigen::Matrix<Scalar,4,4> Gu = T_c_w_mat;
        Vec4 obs_extended;
        obs_extended.setZero();
        obs_extended(2) = sqrt(alpha) * obs(0);
        obs_extended(3) = sqrt(alpha) * obs(1);
        Vec4 v_init_hom = Gu.colPivHouseholderQr().solve(obs_extended);
        const Scalar mx = v_init_hom[0]/ v_init_hom[3];
        const Scalar my = v_init_hom[1]/ v_init_hom[3];
        const Scalar mz = v_init_hom[2]/ v_init_hom[3];

        Vec3 v_init = Vec3(mx, my, mz);
        return v_init;

    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE(Scalar alpha,
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR_pOSE& res,
            MatRP_pOSE* d_res_d_xi, MatRL_pOSE* d_res_d_l) {

        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt(1.0 - alpha) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt(1.0 - alpha) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(alpha) * T_c_w.row(0);
        T_c_w_mat.row(3) = sqrt(alpha) * T_c_w.row(1);

        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        res = p_c_3d;
        res(2) -= sqrt(alpha) * obs(0);
        res(3) -= sqrt(alpha) * obs(1);

        bool projection_valid = true;

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }

        if (d_res_d_xi) {
            // d res / d pose = Jp
            Mat4_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = 1;
            d_p_d_xi(0,8) = -lm_p_w.homogeneous()(0) * obs(0);
            d_p_d_xi(0,9) = -lm_p_w.homogeneous()(1) * obs(0);
            d_p_d_xi(0,10) = -lm_p_w.homogeneous()(2) * obs(0);
            d_p_d_xi(0,11) = - obs(0);
            d_p_d_xi.row(0) *= sqrt(1.0 - alpha);

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);
            d_p_d_xi.row(1) *= sqrt(1.0 - alpha);

            d_p_d_xi(2,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,3) = 1;
            d_p_d_xi.row(2) *= sqrt(alpha);

            d_p_d_xi(3,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(3,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(3,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(3,7) = 1;
            d_p_d_xi.row(3) *= sqrt(alpha);

            *d_res_d_xi = d_p_d_xi;
        }

        if (d_res_d_l) {
            // d res/ d landmark = Jl
            *d_res_d_l = T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space_homogeneous(
            const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w, const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR& res,
            MatRP_projective_space* d_res_d_xi, MatRL_projective_space_homogeneous* d_res_d_l) {

        Mat4 T_c_w_mat;
        T_c_w_mat.setZero();
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        T_c_w_mat.row(2) = T_c_w.row(2);
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;
        Vec4 p_c_3d;

        p_c_3d = T_c_w_mat * lm_p_w;

        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_l) {
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, &d_res_d_p);
        } else {
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, nullptr);
        }


        res -= obs;

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }

        if (d_res_d_xi) {
            // Jp = d res / d pose
            Mat4_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w(0);
            d_p_d_xi(0,1) = lm_p_w(1);
            d_p_d_xi(0,2) = lm_p_w(2);
            d_p_d_xi(0,3) = lm_p_w(3);

            d_p_d_xi(1,4) = lm_p_w(0);
            d_p_d_xi(1,5) = lm_p_w(1);
            d_p_d_xi(1,6) = lm_p_w(2);
            d_p_d_xi(1,7) = lm_p_w(3);

            d_p_d_xi(2,8) = lm_p_w(0);
            d_p_d_xi(2,9) = lm_p_w(1);
            d_p_d_xi(2,10) = lm_p_w(2);
            d_p_d_xi(2,11) = lm_p_w(3);

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }


        if (d_res_d_l) {
            // Jl = d res / d landmark
            *d_res_d_l = d_res_d_p * T_c_w_mat;
        }

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE(Scalar alpha,
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR_pOSE& res, MatRP_pOSE* d_res_d_xi, MatRL_pOSE* d_res_d_l) {

        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt(1.0 - alpha) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt(1.0 - alpha) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(alpha) * T_c_w.row(0);
        T_c_w_mat.row(3) = sqrt(alpha) * T_c_w.row(1);

        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();
        res = p_c_3d;
        bool projection_valid = true;

        res(2) -= sqrt(alpha) * obs(0);
        res(3) -= sqrt(alpha) * obs(1);

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        if (d_res_d_xi) {
            // d res / d pose = Jp
            Mat4_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = 1;
            d_p_d_xi(0,8) = -lm_p_w.homogeneous()(0) * obs(0);
            d_p_d_xi(0,9) = -lm_p_w.homogeneous()(1) * obs(0);
            d_p_d_xi(0,10) = -lm_p_w.homogeneous()(2) * obs(0);
            d_p_d_xi(0,11) = - obs(0);

            d_p_d_xi.row(0) *= sqrt(1.0 - alpha);

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);;
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);

            d_p_d_xi.row(1) *= sqrt(1.0 - alpha);

            d_p_d_xi(2,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,3) = 1;

            d_p_d_xi.row(2) *= sqrt(alpha);

            d_p_d_xi(3,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(3,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(3,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(3,7) = 1;

            d_p_d_xi.row(3) *= sqrt(alpha);

            *d_res_d_xi = d_p_d_xi;

        }

        if (d_res_d_l) {
            // d res/ d landmark = Jl
            *d_res_d_l = T_c_w_mat.template topLeftCorner<4, 3>();
        }
        return projection_valid;
    }

#ifdef ROOTBA_INSTANTIATIONS_FLOAT
template class BalBundleAdjustmentHelper<float>;
#endif

// The helper in double is used by the ceres iteration callback, so always
// compile it; it should not be a big compilation overhead.
//#ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template class BalBundleAdjustmentHelper<double>;
//#endif

}  // namespace rootba_povar
