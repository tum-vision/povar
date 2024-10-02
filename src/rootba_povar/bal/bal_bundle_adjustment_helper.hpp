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
#pragma once

#include "rootba_povar/bal/bal_problem.hpp"
#include "rootba_povar/bal/residual_info.hpp"
#include "rootba_povar/bal/solver_options.hpp"


namespace rootba_povar {

template <typename Scalar>
class BalBundleAdjustmentHelper {
 public:

  static constexpr size_t POSE_SIZE = 6;
  static constexpr size_t INTRINSICS_SIZE = 3;
  static constexpr size_t CAMERA_SIZE = POSE_SIZE + INTRINSICS_SIZE;
  static constexpr size_t LANDMARK_SIZE = 3;
  static constexpr size_t RESIDUAL_SIZE = 2;

  using Vec2 = Mat<Scalar, 2, 1>;
  using Vec3 = Mat<Scalar, 3, 1>;
  using Vec4 = Mat<Scalar, 4, 1>;
  using Vec5 = Mat<Scalar, 5, 1>;
  using VecR = Mat<Scalar, RESIDUAL_SIZE, 1>;
  using VecR_pOSE = Mat<Scalar, RESIDUAL_SIZE+2, 1>;


  using Mat11 = Mat<Scalar, 1, 1>;
  using Mat14 = Mat<Scalar, 1, 4>;
  using Mat18 = Mat<Scalar, 1, 8>;
  using Mat3 = Mat<Scalar, 3, 3>;
  using Mat4 = Mat<Scalar, 4, 4>;
  using Mat54 = Mat<Scalar, 5, 4>;
  using Mat5 = Mat<Scalar, 5, 5>;
  using Mat43 = Mat<Scalar, 4, 3>;
  using Mat24 = Mat<Scalar, 2, 4>;
  using Mat34 = Mat<Scalar, 3, 4>;
  using Mat38 = Mat<Scalar, 3, 8>;
  using Mat48 = Mat<Scalar, 4, 8>;
  using Mat4_11 = Mat<Scalar, 4, 11>;
  using Mat3_12 = Mat<Scalar, 3, 12>;
  using Mat4_12 = Mat<Scalar, 4, 12>;
  using Mat5_12 = Mat<Scalar, 5, 12>;
  using MatRP_affine_space = Mat<Scalar, RESIDUAL_SIZE, 8>;
  using MatRP_projective_space = Mat<Scalar, RESIDUAL_SIZE, 12>;
  using MatRP = Mat<Scalar, RESIDUAL_SIZE, POSE_SIZE>;
  using MatRI = Mat<Scalar, RESIDUAL_SIZE, INTRINSICS_SIZE>;
  using MatRL = Mat<Scalar, RESIDUAL_SIZE, LANDMARK_SIZE>;
  using MatRL_projective_space_homogeneous = Mat<Scalar, RESIDUAL_SIZE, 4>;

  using MatRL_pOSE = Mat<Scalar, 4, 3>;
  using MatRL_pOSE_homogeneous = Mat<Scalar, 4, 4>;
  using MatRP_pOSE = Mat<Scalar, 4, 12>;
  using MatRI_pOSE = Mat<Scalar, 4, INTRINSICS_SIZE>;

  //@Simon: metric upgrade
  using MatRH = Mat<Scalar,9,3>;
  using MatR_alpha = Mat<Scalar,9,9>;
  using MatR_alpha_v2 = Mat<Scalar,9,1>;
  using Mat93 = Mat<Scalar,9,3>;
  //using VecR_metric = Mat<Scalar, 3,3>;
  using VecR_metric = Mat<Scalar, 9,1>;
  using Mat9 = Mat<Scalar,9,9>;
  using Mat9_12 = Mat<Scalar,9,12>;
  using Vec9 = Mat<Scalar,9,1>;

  using SE3 = Sophus::SE3<Scalar>;
  using SO3 = Sophus::SO3<Scalar>;

  // compute error and residual weight (for jacobian) according to robust loss
  static std::tuple<Scalar, Scalar> compute_error_weight(
      const BalResidualOptions& options, Scalar res_squared);


    static Eigen::Matrix<Scalar, 3,1> initialize_varproj_pOSE(Scalar alpha,
            const Vec2& obs, const Mat34& T_c_w);


    static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> kernel_COD(
            const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M);

  // compute the error for all observations
    static void compute_error_pOSE( BalProblem<Scalar>& bal_problem,
                                            const SolverOptions& options, ResidualInfo& error,
                                            bool initialization_varproj = false);

    static void compute_error_projective_space_homogeneous( BalProblem<Scalar>& bal_problem,
                                                const SolverOptions& options, ResidualInfo& error,
                                                bool initialization_varproj = false);


  // linearize one observation
    static bool linearize_point_pOSE(Scalar alpha,
            const Vec2& obs, const Vec3& lm_p_w,
            const Mat34& T_c_w, const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR_pOSE& res,
            MatRP_pOSE* d_res_d_xi = nullptr,
            MatRL_pOSE* d_res_d_l = nullptr);

    static bool linearize_point_projective_space_homogeneous(
            const Vec2& obs, const Vec4& lm_p_w,
            const Mat34& T_c_w, const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR& res,
            MatRP_projective_space* d_res_d_xi = nullptr,
            MatRL_projective_space_homogeneous * d_res_d_l = nullptr);


  static void initialize_varproj_lm_pOSE(Scalar alpha, BalProblem<Scalar>& bal_problem);
  static void setzeros_varproj_lm(BalProblem<Scalar>& bal_problem);


    static bool update_landmark_jacobian_pOSE(Scalar alpha,
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR_pOSE& res, MatRP_pOSE* d_res_d_xi, MatRL_pOSE* d_res_d_l);

};

}  // namespace rootba_povar
