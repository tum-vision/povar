/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
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

@file
@brief Implementation of unified camera model
*/

#pragma once

#include "rootba_povar/basalt_custom/camera/camera_static_assert.hpp"

#include "rootba_povar/basalt_custom/utils/sophus_utils.hpp"

#include <cmath>

namespace basalt {

using std::sqrt;

/// @brief Camera model used in the paper "Bundle Adjustment in the Large".
///
/// See https://grail.cs.washington.edu/projects/bal/ for details.
/// This model has N=3 parameters \f$ \mathbf{i} = \left[f, k_1, k_2
/// \right]^T \f$.
///
/// Unlike the original formulation we assume that the POSITIVE z-axis
/// points in camera direction and thus don't include the "minus" in the
/// perspective projection. You need to consider this when loading BAL data.
///
/// Specifically, for the camera frame we assume the positive z axis pointing
/// forward in view direction and in the image, y is poiting down, x to the
/// right. In the original BAL formulation, the camera points in negative z
/// axis, y is up in the image. Thus when loading the data, we invert the y and
/// z camera axes (y also in the image) in the perspective projection, we don't
/// have the "minus" like in the original Snavely model.
///
/// A 3D point P in camera coordinates is mapped to pixel coordinates p':
/// p  = [P / P.z]_xy    (perspective division)
/// p' =  f * r(p) * p   (conversion to pixel coordinates)
/// r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
///
/// See \ref project and \ref unproject functions for more details.
template <typename Scalar_ = double>
class BalCamera {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 3;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using Vec5 = Eigen::Matrix<Scalar, 5, 1>;
  using Vec9 = Eigen::Matrix<Scalar, 9, 1>;
  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat2 = Eigen::Matrix<Scalar, 2, 2>;
  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;
  using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
  using MatX = Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics
  BalCamera() { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters [f, k1, k2]
  explicit BalCamera(const VecN& p) { param_ = p; }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  BalCamera<Scalar2> cast() const {
    return BalCamera<Scalar2>(param_.template cast<Scalar2>());
  }

  /// @brief Camera model name
  ///
  /// @return "bal"
  static std::string getName() { return "bal"; }


    /// @brief Project the point and optionally compute Jacobians
    template <class DerivedPoint3D, class DerivedPoint2D,
            class DerivedJ3D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
    inline bool project_projective_refinement_matrix_space_without_distortion(const Eigen::MatrixBase<DerivedPoint3D>& p3d,
                                                           Eigen::MatrixBase<DerivedPoint2D>& proj,
                                                           DerivedJ3D d_proj_d_p3d = nullptr) const {
        checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D,
                DerivedJparam, N>();

        const typename EvalOrReference<DerivedPoint3D>::Type p3d_eval(p3d);

        const Scalar& f = param_[0];
        const Scalar& k1 = param_[1];
        const Scalar& k2 = param_[2];

        const Scalar& x = p3d_eval[0];
        const Scalar& y = p3d_eval[1];
        const Scalar& z = p3d_eval[2];

        const Scalar mx = x / z;
        const Scalar my = y / z;

        const Scalar mx2 = mx * mx;
        const Scalar my2 = my * my;

        const Scalar r2 = mx2 + my2;
        const Scalar r4 = r2 * r2;

        proj = Vec2(mx, my);

        const bool is_valid = std::abs(z) >= Sophus::Constants<Scalar>::epsilonSqrt();

        if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
            BASALT_ASSERT(d_proj_d_p3d);
            d_proj_d_p3d->setZero();

            (*d_proj_d_p3d)(0, 0) = 1 / z;
            (*d_proj_d_p3d)(1, 1) = 1 / z;

            (*d_proj_d_p3d)(1, 0) = (*d_proj_d_p3d)(0, 1) =
                    0;

            (*d_proj_d_p3d)(0, 2) = - x / (z * z);
            (*d_proj_d_p3d)(1, 2) = - y / (z * z);

        } else {
            UNUSED(d_proj_d_p3d);
        }

        return is_valid;
    }


  /// @brief Increment intrinsic parameters by inc and clamp the values to the
  /// valid range
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) { param_ += inc; }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is following: \f$ \left[f, k1, k2 \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<BalCamera> getTestProjections() {
    Eigen::aligned_vector<BalCamera> res;

    VecN vec1;

    vec1 << 399.752, -3.78048e-05, 5.37738e-07;
    res.emplace_back(vec1);

    return res;
  }

  /// @brief Resolutions used for unit-tests
  static Eigen::aligned_vector<Eigen::Vector2i> getTestResolutions() {
    Eigen::aligned_vector<Eigen::Vector2i> res;

    res.emplace_back(640, 480);

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param_;
};

}  // namespace basalt
