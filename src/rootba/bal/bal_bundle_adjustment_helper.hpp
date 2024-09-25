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

#include "rootba/bal/bal_problem.hpp"
#include "rootba/bal/residual_info.hpp"
#include "rootba/bal/solver_options.hpp"

namespace rootba {

template <typename Scalar>
class BalBundleAdjustmentHelper {
 public:
  using IntrinsicsT = basalt::BalCamera<Scalar>;

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
  using VecR_RpOSE = Mat<Scalar, RESIDUAL_SIZE+1, 1>;
  using VecR_RpOSE_ML = Mat<Scalar, RESIDUAL_SIZE-1, 1>;
  using VecR_expOSE = Mat<Scalar, RESIDUAL_SIZE+1, 1>;
  using VecR_pOSE_rOSE = Mat<Scalar, RESIDUAL_SIZE+3, 1>;
  using VecR_rOSE = Mat<Scalar, RESIDUAL_SIZE+1, 1>;

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
    using MatRL_RpOSE = Mat<Scalar, 3, 3>;
    using MatRL_RpOSE_ML = Mat<Scalar, 1, 3>;
  using MatRL_expOSE = Mat<Scalar, 3, 3>;
  using MatRL_pOSE_homogeneous = Mat<Scalar, 4, 4>;
  using MatRP_pOSE = Mat<Scalar, 4, 12>;
    using MatRP_RpOSE = Mat<Scalar, 3, 8>;
    using MatRP_RpOSE_ML = Mat<Scalar, 1, 8>;
  using MatRP_expOSE = Mat<Scalar, 3, 12>;
  using MatRI_pOSE = Mat<Scalar, 4, INTRINSICS_SIZE>;
    using MatRI_RpOSE = Mat<Scalar, 3, INTRINSICS_SIZE>;
    using MatRI_RpOSE_ML = Mat<Scalar, 1, INTRINSICS_SIZE>;
  using MatRI_expOSE = Mat<Scalar, 3, INTRINSICS_SIZE>;

  using MatRL_pOSE_rOSE = Mat<Scalar, 5, 3>;
  using MatRP_pOSE_rOSE = Mat<Scalar, 5, 12>;
  using MatRI_pOSE_rOSE = Mat<Scalar, 5, INTRINSICS_SIZE>;

  using MatRL_rOSE = Mat<Scalar, 3, 3>;
  using MatRP_rOSE = Mat<Scalar, 3, 12>;
  using MatRI_rOSE = Mat<Scalar, 3, INTRINSICS_SIZE>;

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

    static std::tuple<Scalar, Scalar> compute_error_weight_cauchy(
            const BalResidualOptions& options, Scalar res_squared);

    static std::tuple<Scalar, Scalar> compute_error_weight_huber(
            const BalResidualOptions& options, Scalar res_squared);

  static  Eigen::Matrix<Scalar, 3,1> initialize_varproj(
            const Vec2& obs, const SE3& T_c_w,
            const basalt::BalCamera<Scalar>& intr);
  static Eigen::Matrix<Scalar, 3,1> initialize_varproj_affine_space(
          const Vec2& obs, const Mat34& T_c_w,
          const basalt::BalCamera<Scalar>& intr);

    static Eigen::Matrix<Scalar, 3,1> initialize_varproj_pOSE(int alpha,
            const Vec2& obs, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr);


    static Eigen::Matrix<Scalar, 3,1> initialize_varproj_RpOSE(double alpha,
                                                              const Vec2& obs, const Mat34& T_c_w,
                                                              const basalt::BalCamera<Scalar>& intr);

    static Eigen::Matrix<Scalar, 3,1> initialize_varproj_expOSE(int alpha,
                                                              const Vec2& obs, const Vec3& y_tilde,
                                                              const Vec3& lm_p_w_equilibrium,
                                                              const Mat34& T_c_w, const Mat34& T_c_w_equilibrium,
                                                              const basalt::BalCamera<Scalar>& intr);

    static Eigen::Matrix<Scalar, 3,1> initialize_varproj_expOSE_v2(int alpha,
                                                                const Vec2& obs, Vec3 y_tilde,
                                                                const Vec3& lm_p_w_equilibrium,
                                                                const Mat34& T_c_w, const Mat34& T_c_w_equilibrium,
                                                                const basalt::BalCamera<Scalar>& intr);


    static Eigen::Matrix<Scalar, 3,1> initialize_varproj_pOSE_rOSE(int alpha,
                                                              const Vec2& obs, const Mat34& T_c_w,
                                                              const basalt::BalCamera<Scalar>& intr);

    static Eigen::Matrix<Scalar, 3,1> initialize_varproj_rOSE(int alpha,
                                                              const Vec2& obs, const Mat34& T_c_w,
                                                              const basalt::BalCamera<Scalar>& intr);

    static Eigen::Matrix<Scalar, 4,1> initialize_varproj_pOSE_homogeneous(int alpha,
                                                              const Vec2& obs, const Mat34& T_c_w,
                                                              const basalt::BalCamera<Scalar>& intr);

    static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> kernel_COD(
            const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& M);

  // compute the error for all observations
  static void compute_error(//const BalProblem<Scalar>& bal_problem,
                            const BalProblem<Scalar>& bal_problem,
                            const SolverOptions& options, ResidualInfo& error);

  static void compute_error_affine_space( BalProblem<Scalar>& bal_problem,
                                          const SolverOptions& options, ResidualInfo& error,
                                          bool initialization_varproj = false);

    static void compute_error_pOSE( BalProblem<Scalar>& bal_problem,
                                            const SolverOptions& options, ResidualInfo& error,
                                            bool initialization_varproj = false);

    static void compute_error_RpOSE( BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options, ResidualInfo& error,
                                    bool initialization_varproj = false);

    static void compute_error_RpOSE_refinement( BalProblem<Scalar>& bal_problem,
                                     const SolverOptions& options, double alpha, ResidualInfo& error);
    static void compute_error_RpOSE_ML( BalProblem<Scalar>& bal_problem,
                                                const SolverOptions& options, ResidualInfo& error);

    static void compute_error_metric_upgrade( BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options, ResidualInfo& error);

    static void compute_error_expOSE( BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options, ResidualInfo& error,
                                    bool initialization_varproj = false);

    static void compute_error_pOSE_rOSE( BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options, ResidualInfo& error,
                                    bool initialization_varproj = false);

    static void compute_error_rOSE( BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options, ResidualInfo& error,
                                    bool initialization_varproj = false);

    static void compute_error_pOSE_homogeneous( BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& options, ResidualInfo& error,
                                    bool initialization_varproj = false);

    static void compute_error_projective_space( BalProblem<Scalar>& bal_problem,
                                            const SolverOptions& options, ResidualInfo& error,
                                            bool initialization_varproj = false);

    static void compute_error_projective_space_homogeneous( BalProblem<Scalar>& bal_problem,
                                                const SolverOptions& options, ResidualInfo& error,
                                                bool initialization_varproj = false);

    static void compute_error_projective_space_homogeneous_RpOSE( BalProblem<Scalar>& bal_problem,
                                                            const SolverOptions& options, ResidualInfo& error,
                                                            bool initialization_varproj = false);

    static void compute_error_projective_space_homogeneous_RpOSE_test_rotation( BalProblem<Scalar>& bal_problem,
                                                                  const SolverOptions& options, ResidualInfo& error,
                                                                  bool initialization_varproj = false);

  static void  compute_error_refine(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error);

  // linearize one observation
  static bool linearize_point(const Vec2& obs, const Vec3& lm_p_w,
                              const SE3& cam_T_c_w,
                              const basalt::BalCamera<Scalar>& intr,
                              bool ignore_validity_check, VecR& res,
                              MatRP* d_res_d_xi = nullptr,
                              MatRI* d_res_d_i = nullptr,
                              MatRL* d_res_d_l = nullptr);

  static bool linearize_point_affine_space(
          const Vec2& obs, const Vec3& lm_p_w,
          const Mat34& T_c_w, //@Simon: or create a submatrix when we load the dataset of size Mat24
          const basalt::BalCamera<Scalar>& intr,
          const bool ignore_validity_check,
          VecR& res,
          const bool initialization_varproj = false,
          MatRP_affine_space* d_res_d_xi = nullptr,
          MatRI* d_res_d_i = nullptr,
          MatRL* d_res_d_l = nullptr);

    static bool linearize_point_pOSE(int alpha,
            const Vec2& obs, const Vec3& lm_p_w,
            const Mat34& T_c_w, //@Simon: or create a submatrix when we load the dataset of size Mat24
            const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR_pOSE& res,
            const bool initialization_varproj = false,
            MatRP_pOSE* d_res_d_xi = nullptr,
            MatRI_pOSE* d_res_d_i = nullptr,
            MatRL_pOSE* d_res_d_l = nullptr);

    static bool linearize_point_RpOSE(double alpha,
                                     const Vec2& obs, const Vec3& lm_p_w,
                                     const Mat34& T_c_w, //@Simon: or create a submatrix when we load the dataset of size Mat24
                                     const basalt::BalCamera<Scalar>& intr,
                                     const bool ignore_validity_check,
                                     VecR_RpOSE& res,
                                     const bool initialization_varproj = false,
                                     MatRP_RpOSE* d_res_d_xi = nullptr,
                                     MatRI_RpOSE* d_res_d_i = nullptr,
                                     MatRL_RpOSE* d_res_d_l = nullptr);

    static bool linearize_point_RpOSE_refinement(double alpha,
                                      const Vec2& obs, const Vec2& rpose_eq,
                                      const Vec3& lm_p_w,
                                      const Mat34& T_c_w, //@Simon: or create a submatrix when we load the dataset of size Mat24
                                      const basalt::BalCamera<Scalar>& intr,
                                      const bool ignore_validity_check,
                                      VecR_RpOSE& res,
                                      MatRP_RpOSE* d_res_d_xi = nullptr,
                                      MatRI_RpOSE* d_res_d_i = nullptr,
                                      MatRL_RpOSE* d_res_d_l = nullptr);

    static bool linearize_point_RpOSE_ML(const Vec2& obs, const Vec2& rpose_eq,
                                                 const Vec3& lm_p_w,
                                                 const Mat34& T_c_w, //@Simon: or create a submatrix when we load the dataset of size Mat24
                                                 const basalt::BalCamera<Scalar>& intr,
                                                 const bool ignore_validity_check,
                                                 VecR_RpOSE_ML& res,
                                                 MatRP_RpOSE_ML* d_res_d_xi = nullptr,
                                                 MatRI_RpOSE_ML* d_res_d_i = nullptr,
                                                 MatRL_RpOSE_ML* d_res_d_l = nullptr);


    static bool linearize_point_expOSE(int alpha,
                                     const Vec2& obs, const Vec3& y_tilde,
                                     const Vec3& lm_p_w, const Vec3& lm_p_w_equilibrium,
                                     const Mat34& T_c_w, const Mat34& T_c_w_equilibrium, //@Simon: or create a submatrix when we load the dataset of size Mat24
                                     const basalt::BalCamera<Scalar>& intr,
                                     const bool ignore_validity_check,
                                     VecR_expOSE& res,
                                     const bool initialization_varproj = false,
                                     MatRP_expOSE* d_res_d_xi = nullptr,
                                     MatRI_expOSE* d_res_d_i = nullptr,
                                     MatRL_expOSE* d_res_d_l = nullptr);

    static bool linearize_point_pOSE_rOSE(int alpha,
                                     const Vec2& obs, const Vec3& lm_p_w,
                                     const Mat34& T_c_w, //@Simon: or create a submatrix when we load the dataset of size Mat24
                                     const basalt::BalCamera<Scalar>& intr,
                                     const bool ignore_validity_check,
                                     VecR_pOSE_rOSE& res,
                                     const bool initialization_varproj = false,
                                     MatRP_pOSE_rOSE* d_res_d_xi = nullptr,
                                     MatRI_pOSE_rOSE* d_res_d_i = nullptr,
                                     MatRL_pOSE_rOSE* d_res_d_l = nullptr);

    static bool linearize_point_rOSE(int alpha,
                                     const Vec2& obs, const Vec3& lm_p_w,
                                     const Mat34& T_c_w, //@Simon: or create a submatrix when we load the dataset of size Mat24
                                     const basalt::BalCamera<Scalar>& intr,
                                     const bool ignore_validity_check,
                                     VecR_rOSE& res,
                                     const bool initialization_varproj = false,
                                     MatRP_rOSE* d_res_d_xi = nullptr,
                                     MatRI_rOSE* d_res_d_i = nullptr,
                                     MatRL_rOSE* d_res_d_l = nullptr);

    static bool linearize_point_pOSE_homogeneous(int alpha,
                                     const Vec2& obs, const Vec4& lm_p_w,
                                     const Mat34& T_c_w, //@Simon: or create a submatrix when we load the dataset of size Mat24
                                     const basalt::BalCamera<Scalar>& intr,
                                     const bool ignore_validity_check,
                                     VecR_pOSE& res,
                                     const bool initialization_varproj = false,
                                     MatRP_pOSE* d_res_d_xi = nullptr,
                                     MatRI_pOSE* d_res_d_i = nullptr,
                                     MatRL_pOSE_homogeneous* d_res_d_l = nullptr);

    static bool linearize_metric_upgrade(const Mat3& PH,
                                         const Mat3& PHHP,
                                         const Mat34& space_matrix_intrinsics,
                                         const Scalar& alpha,
                                         const basalt::BalCamera<Scalar>& intr,
                                         const bool ignore_validity_check,
                                         VecR_metric& res,
                                         MatRH* d_res_d_H = nullptr,
                                         MatR_alpha* d_res_d_alpha = nullptr);

    static bool linearize_metric_upgrade_v2(const Mat3& PH,
                                         const Mat3& PHHP,
                                         const Mat34& space_matrix_intrinsics,
                                         const Scalar& alpha,
                                         const basalt::BalCamera<Scalar>& intr,
                                         const bool ignore_validity_check,
                                         VecR_metric& res,
                                         MatRH* d_res_d_H = nullptr,
                                            MatR_alpha* d_res = nullptr,
                                         MatR_alpha_v2* d_res_d_alpha = nullptr);

    static bool linearize_metric_upgrade_v3(const Vec3& c,
                                            const Mat3& PH,
                                            const Mat3& PHHP,
                                            const Mat34& space_matrix_intrinsics,
                                            const Scalar& alpha,
                                            const basalt::BalCamera<Scalar>& intr,
                                            const bool ignore_validity_check,
                                            VecR_metric& res,
                                            MatRH* d_res_d_H = nullptr,
                                            MatR_alpha* d_res = nullptr,
                                            MatR_alpha_v2* d_res_d_alpha = nullptr);

    static bool linearize_metric_upgrade_v3_pollefeys(const Scalar& f,
                                            const Vec3& c,
                                            const Mat3& PH,
                                            const Mat3& PHHP,
                                            const Mat34& space_matrix_intrinsics,
                                            const Scalar& alpha,
                                            const basalt::BalCamera<Scalar>& intr,
                                            const bool ignore_validity_check,
                                            VecR_metric& res,
                                            MatRH* d_res_d_H = nullptr,
                                            MatR_alpha* d_res = nullptr,
                                            MatR_alpha_v2* d_res_d_alpha = nullptr);

    static bool update_jacobian_metric_upgrade(const Mat3& PH,
                                            const Mat3& PHHP,
                                            const Mat34& space_matrix_intrinsics,
                                            const Scalar& alpha,
                                            const basalt::BalCamera<Scalar>& intr,
                                            const bool ignore_validity_check,
                                            VecR_metric& res,
                                            MatRH* d_res_d_H = nullptr,
                                            MatR_alpha* d_res = nullptr,
                                            MatR_alpha_v2* d_res_d_alpha = nullptr);

    static bool update_jacobian_metric_upgrade_v3(const Vec3& c,
                                                const Mat3& PH,
                                               const Mat3& PHHP,
                                               const Mat34& space_matrix_intrinsics,
                                               const Scalar& alpha,
                                               const basalt::BalCamera<Scalar>& intr,
                                               const bool ignore_validity_check,
                                               VecR_metric& res,
                                               MatRH* d_res_d_H = nullptr,
                                               MatR_alpha* d_res = nullptr,
                                               MatR_alpha_v2* d_res_d_alpha = nullptr);


    static bool linearize_point_projective_space(
            const Vec2& obs, const Vec3& lm_p_w,
            const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR& res,
            const bool initialization_varproj = false,
            MatRP_projective_space* d_res_d_xi = nullptr,
            MatRI* d_res_d_i = nullptr,
            MatRL* d_res_d_l = nullptr);

    static bool linearize_point_projective_space_homogeneous(
            const Vec2& obs, const Vec4& lm_p_w,
            const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR& res,
            const bool initialization_varproj = false,
            MatRP_projective_space* d_res_d_xi = nullptr,
            MatRI* d_res_d_i = nullptr,
            MatRL_projective_space_homogeneous * d_res_d_l = nullptr);

    static bool linearize_point_projective_space_homogeneous_RpOSE(
            const Vec2& obs, const Vec4& lm_p_w,
            const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR& res,
            const bool initialization_varproj = false,
            MatRP_projective_space* d_res_d_xi = nullptr,
            MatRI* d_res_d_i = nullptr,
            MatRL_projective_space_homogeneous * d_res_d_l = nullptr);

    static bool linearize_point_projective_space_homogeneous_RpOSE_test_rotation(
            const Vec2& obs, const Vec4& lm_p_w,
            const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr,
            const bool ignore_validity_check,
            VecR& res,
            const bool initialization_varproj = false,
            MatRP_projective_space* d_res_d_xi = nullptr,
            MatRI* d_res_d_i = nullptr,
            MatRL_projective_space_homogeneous * d_res_d_l = nullptr);

    static bool linearize_point_refine(
            const Vec2& obs, const Vec3& lm_p_w, const SE3& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res,
            MatRP* d_res_d_xi = nullptr, MatRI* d_res_d_i = nullptr, MatRL* d_res_d_l = nullptr);

  static void initialize_varproj_lm(BalProblem<Scalar>& bal_problem);
  static void initialize_varproj_lm_affine_space(BalProblem<Scalar>& bal_problem);
  static void initialize_varproj_lm_pOSE(int alpha, BalProblem<Scalar>& bal_problem);
  static void initialize_varproj_lm_RpOSE(double alpha, BalProblem<Scalar>& bal_problem);
  static void initialize_varproj_lm_expOSE(int alpha, BalProblem<Scalar>& bal_problem);
  static void initialize_varproj_lm_pOSE_rOSE(int alpha, BalProblem<Scalar>& bal_problem);
  static void initialize_varproj_lm_rOSE(int alpha, BalProblem<Scalar>& bal_problem);
  static void initialize_varproj_lm_pOSE_homogeneous(int alpha, BalProblem<Scalar>& bal_problem);

  static void setzeros_varproj_lm(BalProblem<Scalar>& bal_problem);
  static void setzeros_varproj_lm_homogeneous(BalProblem<Scalar>& bal_problem);

    static bool update_landmark_jacobian(
            const Vec2& obs, const Vec3& lm_p_w, const SE3& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l);

    static bool update_landmark_jacobian_affine_space(
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP_affine_space* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l);

    static bool update_landmark_jacobian_pOSE(int alpha,
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR_pOSE& res, MatRP_pOSE* d_res_d_xi, MatRI_pOSE* d_res_d_i, MatRL_pOSE* d_res_d_l);

    static bool update_landmark_jacobian_RpOSE(double alpha,
                                              const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
                                              const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                              VecR_RpOSE& res, MatRP_RpOSE* d_res_d_xi, MatRI_RpOSE* d_res_d_i, MatRL_RpOSE* d_res_d_l);

    static bool update_landmark_jacobian_RpOSE_refinement(double alpha,
                                               const Vec2& obs, const Vec2& rpose_eq, const Vec3& lm_p_w, const Mat34& T_c_w,
                                               const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                               VecR_RpOSE& res, MatRP_RpOSE* d_res_d_xi, MatRI_RpOSE* d_res_d_i, MatRL_RpOSE* d_res_d_l);

    static bool update_landmark_jacobian_RpOSE_ML(const Vec2& obs, const Vec2& rpose_eq, const Vec3& lm_p_w, const Mat34& T_c_w,
                                                          const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                          VecR_RpOSE_ML& res, MatRP_RpOSE_ML* d_res_d_xi, MatRI_RpOSE_ML* d_res_d_i, MatRL_RpOSE_ML* d_res_d_l);


    static bool update_landmark_jacobian_expOSE(int alpha,
                                              const Vec2& obs, const Vec3& y_tilde,
                                              const Vec3& lm_p_w,
                                              const Vec3& lm_p_w_equilibrium,
                                              const Mat34& T_c_w, const Mat34& T_c_w_equilibrium,
                                              const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                              VecR_expOSE& res, MatRP_expOSE* d_res_d_xi, MatRI_expOSE* d_res_d_i, MatRL_expOSE* d_res_d_l);

    static bool update_landmark_jacobian_pOSE_rOSE(int alpha,
                                              const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
                                              const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                              VecR_pOSE_rOSE& res, MatRP_pOSE_rOSE* d_res_d_xi, MatRI_pOSE_rOSE* d_res_d_i, MatRL_pOSE_rOSE* d_res_d_l);

    static bool update_landmark_jacobian_rOSE(int alpha,
                                              const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
                                              const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                              VecR_rOSE& res, MatRP_rOSE* d_res_d_xi, MatRI_rOSE* d_res_d_i, MatRL_rOSE* d_res_d_l);

    static bool update_landmark_jacobian_pOSE_homogeneous(int alpha,
                                              const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w,
                                              const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                              VecR_pOSE& res, MatRP_pOSE* d_res_d_xi, MatRI_pOSE* d_res_d_i, MatRL_pOSE_homogeneous* d_res_d_l);

    static bool update_landmark_jacobian_projective_space(
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP_projective_space* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l);

    static bool update_landmark_jacobian_projective_space_homogeneous(
            const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP_projective_space* d_res_d_xi, MatRI* d_res_d_i, MatRL_projective_space_homogeneous * d_res_d_l);
};

}  // namespace rootba
