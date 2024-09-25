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

#include <map>

#include <basalt/camera/bal_camera.hpp>
#include <glog/logging.h>
#include <unordered_set>


#include "rootba/bal/common_types.hpp"
#include "rootba/util/cast.hpp"

namespace rootba {

struct BalDatasetOptions;
struct DatasetSummary;
struct PipelineTimingSummary;

// BAL problem as loaded from the dataset files: Cameras are represented
// as world-to-cam, and landmarks in world coordinates.
//
// For the camera frame we assume the positive z axis pointing forward in view
// direction and in the image, y is poiting down, x to the right. In the
// original BAL formulation, the camera points in negative z axis, y is up in
// the image. Thus when loading the data, we invert the y and z camera axes (y
// also in the image) in the perspective projection, we don't have the "minus"
// like in the original Snavely model.
template <typename Scalar>
class BalProblem {
 public:
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
  using Vec8 = Eigen::Matrix<Scalar, 8, 1>;
  using Vec9 = Eigen::Matrix<Scalar, 9, 1>;
  using Vec11 = Eigen::Matrix<Scalar, 11, 1>;
  using Vec12 = Eigen::Matrix<Scalar, 12, 1>;
  using Vec14 = Eigen::Matrix<Scalar, 14, 1>;
  using Vec15 = Eigen::Matrix<Scalar, 15, 1>;
  using Mat28 = Eigen::Matrix<Scalar, 2, 8>;
  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat34 = Eigen::Matrix<Scalar, 3, 4>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using SE3 = Sophus::SE3<Scalar>;
  using SO3 = Sophus::SO3<Scalar>;

  using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
  using Mat43 = Eigen::Matrix<Scalar, 4, 3>;
  using Mat3 = Eigen::Matrix<Scalar, 3, 3>;


  static constexpr int CAM_STATE_SIZE = 10;

  using CameraModel = basalt::BalCamera<Scalar>;

  struct Observation {
      Vec3 y_tilde;
    Vec2 pos;
    Vec2 rpose_eq;
  };

  struct H_euclidean {
      Mat43 Ht;
      Mat4 H;
      Vec3 plan_infinity;



      void backup_metric_upgrade() {
          Ht_backup_ = Ht;
          H_backup_ = H;
          plan_infinity_backup_ = plan_infinity;
      }

      void restore_metric_upgrade() {
          Ht = Ht_backup_;
          H = H_backup_;
          plan_infinity = plan_infinity_backup_;
      }

  private:
      Mat43 Ht_backup_;
      Mat4 H_backup_;
      Vec3 plan_infinity_backup_;
  };

  struct Camera {
    SE3 T_c_w;               // world-to-cam pose
    //Mat24 space_matrix;
    CameraModel intrinsics;  // per-camera intrinsics
    std::unordered_set<int> linked_cameras;
    Mat34 space_matrix;

    //@Simon: for RpOSE
    std::unordered_set<int> linked_lm;
    //std::map<LandmarkIdx, Observation> obs;  // list of landmark indices


    Vec3 plan_infinity; //@Simon: = c in pOSE, metric upgrade
    Mat34 space_matrix_intrinsics; //@Simon: quantity P_tilde = K^{-1} P in pOSE paper, metric upgrade
    Mat4 H; //@Simon: H in pOSE, metric upgrade
    Mat43 Ht;
    Mat3 PH;
    Mat3 PHHP;
    Scalar alpha;


    VecX params() const {
      VecX p(CAM_STATE_SIZE);
      p.template head<7>() = T_c_w.params();
      p.template tail<3>() = intrinsics.getParam();
      return p;
    }

    void from_params(const VecX& p) {
      CHECK_EQ(p.size(), CAM_STATE_SIZE);
      T_c_w = Eigen::Map<const SE3>(p.data());
      intrinsics = CameraModel(p.template tail<3>());
    }

    void apply_inc_pose(const Vec6& inc) { inc_pose(inc, T_c_w); }

      void apply_inc_pose_affine_space(const Vec8& inc) { inc_pose_affine_space(inc, space_matrix); }

      void apply_inc_pose_projective_space(const Vec12& inc) { inc_pose_projective_space(inc, space_matrix); }

      void apply_inc_pose_projective_space_intrinsics(const Vec3& inc) { inc_intrinsics(inc, intrinsics); }


      void apply_inc_pose_pOSE(const Vec12& inc) { inc_pose_pOSE(inc, space_matrix); }

      void apply_inc_pose_RpOSE(const Vec8& inc) { inc_pose_RpOSE(inc, space_matrix); }


      void apply_inc_pose_expOSE(const Vec12& inc) { inc_pose_pOSE(inc, space_matrix); }

    inline static void inc_pose(const Vec6& inc, SE3& T_c_w) {
      T_c_w = Sophus::se3_expd(inc) * T_c_w;
    }

      inline static void inc_pose_affine_space(const Vec8& inc, Mat34& T_c_w) {
        Mat24 tmp;
        tmp.row(0) = inc.template head<4>();
        tmp.row(1) = inc.template tail<4>();
        T_c_w.row(0) += tmp.row(0);
        T_c_w.row(1) += tmp.row(1);
          //T_c_w.row(0) = T_c_w.row(0) + inc.template head<4>();
          //T_c_w.row(1) = T_c_w.row(1) + inc.template tail<4>();
      }

      inline static void inc_pose_projective_space(const Vec12& inc, Mat34& T_c_w) {
          Mat34 tmp;
          tmp.setZero();
          tmp.row(0) = inc.template head<4>();
          tmp.row(1) = inc.template segment<4>(4);
          tmp.row(2) = inc.template tail<4>();
          //tmp(2,0) = inc[8] ;
          //tmp(2,1) = inc[9] ;
          //tmp(2,2) = inc[10];
          //tmp(2,3) = inc[11];
          T_c_w += tmp;
          //T_c_w.row(0) = T_c_w.row(0) + inc.template head<4>();
          //T_c_w.row(1) = T_c_w.row(1) + inc.template tail<4>();
      }

      inline static void inc_pose_pOSE(const Vec12& inc, Mat34& T_c_w) {
          Mat34 tmp;
          tmp.setZero();
          tmp.row(0) = inc.template head<4>();
          tmp.row(1) = inc.template segment<4>(4);
          tmp.row(2) = inc.template tail<4>();
          //tmp(2,0) = inc[8] ;
          //tmp(2,1) = inc[9] ;
          //tmp(2,2) = inc[10];
          //tmp(2,3) = inc[11];
          T_c_w += tmp;
          //T_c_w.row(0).normalize();
          //T_c_w.row(1).normalize();
          //T_c_w.row(2).normalize();
          //T_c_w.row(0) = T_c_w.row(0) + inc.template head<4>();
          //T_c_w.row(1) = T_c_w.row(1) + inc.template tail<4>();
      }

      inline static void inc_pose_RpOSE(const Vec8& inc, Mat34& T_c_w) {
          Mat34 tmp;
          tmp.setZero();
          tmp.row(0) = inc.template head<4>();
          tmp.row(1) = inc.template tail<4>();
          T_c_w += tmp;
      }


      void apply_inc_intrinsics(const Vec3& inc) {
      inc_intrinsics(inc, intrinsics);
    }



    inline static void inc_intrinsics(const Vec3& inc, CameraModel& intr) {
      intr += inc;
    }


    void backup() {
      T_c_w_backup_ = T_c_w;
      intrinsics_backup_ = intrinsics;
    }

      void backup_metric_upgrade() {
          alpha_ = alpha;
          PH_ = PH;
          PHHP_ = PHHP;
      }

      void restore_metric_upgrade() {
          alpha = alpha_;
          PH = PH_;
          PHHP = PHHP_;
      }

      void backup_affine_space() {
          space_matrix_backup_ = space_matrix;
          intrinsics_backup_ = intrinsics;
      }

      void backup_projective_space_homogeneous() {
          space_matrix_backup_ = space_matrix;
          intrinsics_backup_ = intrinsics;
      }

      void backup_joint() {
          space_matrix_backup_ = space_matrix;
          intrinsics_backup_ = intrinsics;
      }

      void backup_projective_space_homogeneous_lm_landmark() {
          space_matrix_lm_backup_ = space_matrix;
          intrinsics_lm_backup_ = intrinsics;
      }

      void backup_pOSE() {
          space_matrix_backup_ = space_matrix;
          intrinsics_backup_ = intrinsics;
      }

      void backup_RpOSE() {
          space_matrix_backup_ = space_matrix;
          intrinsics_backup_ = intrinsics;
      }

      void backup_expOSE() {
          space_matrix_backup_ = space_matrix;
          intrinsics_backup_ = intrinsics;
      }

      void backup_pOSE_homogeneous() {
          space_matrix_backup_ = space_matrix;
          intrinsics_backup_ = intrinsics;
      }

    inline Mat34& space_matrix_backup() {return space_matrix_backup_; }
    inline const Mat34& space_matrix_backup() const {return space_matrix_backup_; }

    void restore() {
      T_c_w = T_c_w_backup_;
      intrinsics = intrinsics_backup_;
    }

      void restore_affine_space() {
          space_matrix = space_matrix_backup_;
          intrinsics = intrinsics_backup_;
      }
      void restore_projective_space_homogeneous() {
          space_matrix = space_matrix_backup_;
          intrinsics = intrinsics_backup_;
      }

      void restore_joint() {
          space_matrix = space_matrix_backup_;
          intrinsics = intrinsics_backup_;
      }

      void restore_projective_space_homogeneous_lm_landmark() {
          space_matrix = space_matrix_lm_backup_;
          intrinsics = intrinsics_lm_backup_;
      }

      void restore_pOSE() {
          space_matrix = space_matrix_backup_;
          intrinsics = intrinsics_backup_;
      }

      void restore_RpOSE() {
          space_matrix = space_matrix_backup_;
          intrinsics = intrinsics_backup_;
      }

      void restore_expOSE() {
          space_matrix = space_matrix_backup_;
          intrinsics = intrinsics_backup_;
      }

      void restore_pOSE_homogeneous() {
          space_matrix = space_matrix_backup_;
          intrinsics = intrinsics_backup_;
      }

    template <typename Scalar2>
    typename BalProblem<Scalar2>::Camera cast() const {
      typename BalProblem<Scalar2>::Camera res;
      res.T_c_w = T_c_w.template cast<Scalar2>();
      res.intrinsics = intrinsics.template cast<Scalar2>();

      return res;
    }

   private:
    SE3 T_c_w_backup_;
    Mat34 space_matrix_backup_;
    CameraModel intrinsics_backup_;
    Mat34 space_matrix_lm_backup_;
    CameraModel intrinsics_lm_backup_;
    Scalar alpha_;
    Mat3 PH_;
    Mat3 PHHP_;
  };

  struct Landmark {
    Vec3 p_w;                             // point position in world coordinates
    Vec4 p_w_homogeneous;
    std::map<FrameIdx, Observation> obs;  // list of frame indices
    std::unordered_set<int> linked_cam;


    void backup() { p_w_backup_ = p_w; }

    void restore() { p_w = p_w_backup_; }

      void backup_pOSE() { p_w_backup_ = p_w; }

      void backup_RpOSE() { p_w_backup_ = p_w; }

      void backup_expOSE() { p_w_backup_ = p_w; }

      void backup_pOSE_homogeneous() { p_w_homogeneous_backup_ = p_w_homogeneous; }

      void restore_pOSE() { p_w = p_w_backup_; }
      void restore_RpOSE() { p_w = p_w_backup_; }
      void restore_expOSE() { p_w = p_w_backup_; }
      void restore_pOSE_homogeneous() { p_w_homogeneous = p_w_homogeneous_backup_; }

    void backup_projective_space_homogeneous() { p_w_homogeneous_backup_ = p_w_homogeneous; }

      void backup_joint() { p_w_homogeneous_backup_ = p_w_homogeneous; }

    void restore_projective_space_homogeneous() { p_w_homogeneous = p_w_homogeneous_backup_; }

      void restore_joint() { p_w_homogeneous = p_w_homogeneous_backup_; }

      void backup_projective_space_homogeneous_lm_landmark() { p_w_homogeneous_lm_backup_ = p_w_homogeneous; }

      void restore_projective_space_homogeneous_lm_landmark() { p_w_homogeneous = p_w_homogeneous_lm_backup_; }

    template <typename Scalar2>
    typename BalProblem<Scalar2>::Landmark cast() const {
      typename BalProblem<Scalar2>::Landmark res;
      res.p_w = p_w.template cast<Scalar2>();
      for (const auto& [frame_id, o] : obs) {
        res.obs[frame_id].pos = o.pos.template cast<Scalar2>();
      }

      return res;
    }

   inline Vec3& p_w_backup() {return p_w_backup_;}
   inline const Vec3& p_w_backup() const {return p_w_backup_;}

   private:
    Vec3 p_w_backup_;
    Vec4 p_w_homogeneous_lm_backup_;
    Vec4 p_w_homogeneous_backup_;
  };

  using H = std::vector<H_euclidean>;
  using Cameras = std::vector<Camera>;
  using Landmarks = std::vector<Landmark>;

  BalProblem() = default;
  explicit BalProblem(const std::string& path);

  void load_bal(const std::string& path);
    void load_bal_eccv(const std::string& path);
  void load_bundler(const std::string& path);

  void load_bal_varproj(const std::string& path);

  void load_bal_varproj_space_matrix(const std::string& path);

    void load_bal_varproj_space_matrix_write(const std::string& path);

  void load_bal_varproj_space_matrix_homogeneous(const std::string& path);

  bool load_rootba(const std::string& path);
  bool save_rootba(const std::string& path);

  void normalize(double new_scale);
  void normalize_expose(double new_scale);

  void perturb(double rotation_sigma, double translation_sigma,
               double landmark_sigma, int seed = -1);

  void filter_obs(double threshold);

  void postprocress(const BalDatasetOptions& options,
                    PipelineTimingSummary* timing_summary = nullptr);

  void copy_to_camera_state(VecX& camera_state) const;
  void copy_from_camera_state(const VecX& camera_state);

  void backup();
  void backup_affine_space();
  void backup_pOSE();
  void backup_RpOSE();
  void backup_expOSE();
  void backup_joint();
  void backup_pOSE_homogeneous();
  void backup_projective_space_homogeneous();
  void backup_projective_space_homogeneous_lm_landmark();
  void backup_metric_upgrade();
  void restore_metric_upgrade();
  void restore();
  void restore_affine_space();
  void restore_pOSE();
  void restore_RpOSE();
  void restore_expOSE();
  void restore_joint();
  void restore_pOSE_homogeneous();
  void restore_projective_space_homogeneous();
  void restore_projective_space_homogeneous_lm_landmark();

  inline const H_euclidean& h_euclidean() const { return H_; }
  inline const Cameras& cameras() const { return cameras_; }
  inline const Landmarks& landmarks() const { return landmarks_; }

  inline H_euclidean& h_euclidean() { return H_; }
  inline Cameras& cameras() { return cameras_; }
  inline Landmarks& landmarks() { return landmarks_; }


  inline int num_cameras() const { return signed_cast(cameras_.size()); }
  inline int num_landmarks() const { return signed_cast(landmarks_.size()); }
  int num_observations() const;
  int max_num_observations_per_lm() const;
  double compute_rcs_sparsity() const;

  bool quiet() const { return quiet_; }
  void set_quiet(const bool quiet) { quiet_ = quiet; }

  template <typename Scalar2>
  BalProblem<Scalar2> copy_cast() const {
    BalProblem<Scalar2> res;

    res.cameras_.resize(this->cameras_.size());
    res.landmarks_.resize(this->landmarks_.size());

    for (size_t i = 0; i < this->cameras_.size(); i++) {
      res.cameras_[i] = this->cameras_[i].template cast<Scalar2>();
    }

    for (size_t i = 0; i < this->landmarks_.size(); i++) {
      res.landmarks_[i] = this->landmarks_[i].template cast<Scalar2>();
    }

    res.quiet_ = quiet_;

    return res;
  }

  void summarize_problem(DatasetSummary& summary, bool compute_sparsity) const;

  std::string stats_to_string() const;

 private:
  template <typename T>
  friend class BalProblem;

  H_euclidean H_;
  Cameras cameras_;
  Landmarks landmarks_;

  /// quiet means no INFO level log output
  bool quiet_ = false;
};

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary = nullptr,
    PipelineTimingSummary* timing_summary = nullptr);

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const std::string& path, DatasetSummary* dataset_summary = nullptr,
    PipelineTimingSummary* timing_summary = nullptr);

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem_quiet(const std::string& path);

}  // namespace rootba
