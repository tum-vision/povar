/**
BSD 3-Clause License

This file is part of the rootba_povar project.
https://github.com/NikolausDemmel/rootba_povar

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

#include "rootba_povar/bal/bal_problem.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <utility>
#include <chrono>
#include <thread>


#include <absl/container/flat_hash_set.h>
#include <cereal/archives/binary.hpp>
#include <glog/logging.h>

#include "rootba_povar/bal/bal_dataset_options.hpp"
#include "rootba_povar/bal/bal_pipeline_summary.hpp"
#include "rootba_povar/bal/bal_problem_io.hpp"
#include "rootba_povar/cg/block_sparse_matrix.hpp"
#include "rootba_povar/util/format.hpp"
#include "rootba_povar/util/stl_utils.hpp"
#include "rootba_povar/util/time_utils.hpp"

#include <random>

namespace rootba_povar {

namespace {  // helper

    using namespace std::chrono_literals;

template <typename T>
void fscan_or_throw(FILE* fptr, const char* format, T* value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    throw std::runtime_error("");
  }
}

template <typename T, int N>
void fscan_or_throw(FILE* fptr, Eigen::Matrix<T, N, 1>& values) {
  for (int i = 0; i < values.size(); ++i) {
    fscan_or_throw(fptr, "%lf", values.data() + i);
  }
}

void readcommentline_or_throw(FILE* fptr) {
  char buffer[1000];
  bool comment_ok = false;
  while (fgets(buffer, 1000, fptr) != nullptr) {
    size_t len = strlen(buffer);

    if (len == 0) {
      throw std::runtime_error("empty line; expected comment...");
    }

    // first part of line, check # character
    if (!comment_ok) {
      if (buffer[0] == '#') {
        comment_ok = true;
      } else {
        throw std::runtime_error("non-comment line; expected comment...");
      }
    }

    // check if we reached eol
    if (buffer[len - 1] == '\n') {
      return;
    }
  }

  // fgets failed
  throw std::runtime_error("could not read comment line");
}

template <typename T, int N, class RandomEngine>
Eigen::Matrix<T, N, 1> perturbation(const T sigma, RandomEngine& eng) {
  std::normal_distribution<T> normal;
  Eigen::Matrix<T, N, 1> vec;
  vec.setZero();
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] += normal(eng) * sigma;
  }
  return vec;
}

template <typename T>
T median_destructive(std::vector<T>& data) {
  int n = data.size();
  auto mid_point = data.begin() + n / 2;
  std::nth_element(data.begin(), mid_point, data.end());
  return *mid_point;
}

BalDatasetOptions::DatasetType autodetect_input_type(const std::string& path) {
    return BalDatasetOptions::DatasetType::BAL;
}

class BalProblemSaver : public FileSaver<cereal::BinaryOutputArchive> {
 public:
  using Scalar = double;

  inline BalProblemSaver(std::string path,
                         const BalProblem<Scalar>& bal_problem)
      : FileSaver(BAL_PROBLEM_FILE_INFO, std::move(path)),
        bal_problem_(bal_problem) {}

 protected:
  inline bool save_impl(cereal::BinaryOutputArchive& archive) override {
    archive(bal_problem_);
    return true;
  }

  inline std::string format_summary() const override {
    return bal_problem_.stats_to_string();
  }

 private:
  const BalProblem<Scalar>& bal_problem_;
};

class BalProblemLoader : public FileLoader<cereal::BinaryInputArchive> {
 public:
  using Scalar = double;

  inline BalProblemLoader(std::string path, BalProblem<Scalar>& bal_problem)
      : FileLoader(BAL_PROBLEM_FILE_INFO, std::move(path)),
        bal_problem_(bal_problem) {}

 protected:
  inline bool load_impl() override {
    (*archive_)(bal_problem_);
    return true;
  }

  inline std::string format_summary() const override {
    return bal_problem_.stats_to_string();
  }

 private:
  BalProblem<Scalar>& bal_problem_;
};

}  // namespace

    template <typename Scalar>
    void BalProblem<Scalar>::load_bal_eccv(const std::string& path) {
        FILE* fptr = std::fopen(path.c_str(), "r");
        std::random_device rd; // @Simon: necessary to initialize the VarProj algorithm
        std::mt19937 gen(rd());
        if (fptr == nullptr) {
            LOG(FATAL) << "Could not open '{}'"_format(path);
        };

        try {
            // parse header
            int num_cams;
            int num_lms;
            int num_obs;
            fscan_or_throw(fptr, "%d", &num_cams);
            fscan_or_throw(fptr, "%d", &num_lms);
            fscan_or_throw(fptr, "%d", &num_obs);
            CHECK_GT(num_cams, 0);
            CHECK_GT(num_lms, 0);
            CHECK_GT(num_obs, 0);
            // clear memory and re-allocate
            if (cameras_.capacity() > unsigned_cast(num_cams)) {
                decltype(cameras_)().swap(cameras_);
            }
            if (landmarks_.capacity() > unsigned_cast(num_lms)) {
                decltype(landmarks_)().swap(landmarks_);
            }
            cameras_.resize(num_cams);
            landmarks_.resize(num_lms);
            // parse observations
            for (int i = 0; i < num_obs; ++i) {
                int cam_idx;
                int lm_idx;
                fscan_or_throw(fptr, "%d", &cam_idx);
                fscan_or_throw(fptr, "%d", &lm_idx);
                CHECK_GE(cam_idx, 0);
                CHECK_LT(cam_idx, num_cams);
                CHECK_GE(lm_idx, 0);
                CHECK_LT(lm_idx, num_lms);

                auto [obs, inserted] = landmarks_.at(lm_idx).obs.try_emplace(cam_idx);
                /// create <landmark, <cam>>
                landmarks_.at(lm_idx).linked_cam.insert(cam_idx);


                CHECK(inserted) << "Invalid file '{}'"_format(path);
                Eigen::Matrix<double, 2, 1> posd;
                fscan_or_throw(fptr, posd);
                obs->second.pos = posd.cast<Scalar>();

                // For the camera frame we assume the positive z axis pointing
                // forward in view direction and in the image, y is poiting down, x to the
                // right. In the original BAL formulation, the camera points in negative z
                // axis, y is up in the image. Thus when loading the data, we invert the y
                // and z camera axes (y also in the image) in the perspective projection,
                // we don't have the "minus" like in the original Snavely model.

                // invert y axis
                obs->second.pos.y() = -obs->second.pos.y();
            }

            // parse camera parameters
            for (int i = 0; i < num_cams; ++i) {
                Vec15 params;
                Eigen::Matrix<double, 15, 1> paramsd;
                fscan_or_throw(fptr, paramsd);
                params = paramsd.cast<Scalar>();

                auto& cam = cameras_.at(i);
                cam.space_matrix.row(0) = params.template head<4>();
                cam.space_matrix.row(1) = params.template segment<4>(4);
                cam.space_matrix.row(2) =  params.template segment<4>(8);

                cam.intrinsics = CameraModel(params.template tail<3>());
            }
            // parse landmark parameters
            for (int i = 0; i < num_lms; ++i) {
                Eigen::Matrix<double, 3, 1> p_wd;
                fscan_or_throw(fptr, p_wd);
                Eigen::Matrix<double, 3, 1> p_wd_random;
                std::normal_distribution<double> d(0, 1); // @Simon: generate normal distribution
                for (int m = 0; m < 3; ++m) {
                    p_wd_random[m] = d(gen);
                }
                //landmarks_.at(i).p_w = p_wd.cast<Scalar>();
                landmarks_.at(i).p_w = p_wd_random.cast<Scalar>();
            }
            /// build sparse graph associated to the SC
            for (int i = 0; i < num_lms; ++i)
            {
                Landmark lm = landmarks_.at(i);
                std::unordered_set<int> const & linked_cam_i = lm.linked_cam;
                std::unordered_set<int>::const_iterator it = linked_cam_i.begin();
                for (; it != linked_cam_i.end(); it++)
                {
                    int cam_idx1 = *it; //check
                    std::unordered_set<int>::const_iterator it2 = linked_cam_i.begin();
                    for (; it2 != linked_cam_i.end(); it2++)
                    {
                        int cam_idx2 = *it2; //check
                        if (cam_idx1 != cam_idx2)
                        {
                            cameras_.at(cam_idx1).linked_cameras.insert(cam_idx2);
                        }
                    }
                }

            }
        } catch (const std::exception& e) {
            LOG(FATAL) << "Failed to parse '{}'"_format(path);
        }

        if (!quiet_) {
            LOG(INFO)
                    << "Loaded BAL problem ({} cams, {} lms, {} obs) from '{}'"_format(
                            num_cameras(), num_landmarks(), num_observations(), path);
        }

        // Current implementation uses int to compute state vector indices
        CHECK_LT(num_cameras(), std::numeric_limits<int>::max() / CAM_STATE_SIZE);
        std::fclose(fptr);
    }

    //@Simon: for VarProj by considering the camera matrix space instead of SE(3) for poses
    template <typename Scalar>
    void BalProblem<Scalar>::load_bal_varproj_space_matrix_write(const std::string& path) {
        namespace fs = std::filesystem;
        FILE* fptr = std::fopen(path.c_str(), "r");
        if (!fs::is_directory("data_custom") || !fs::exists("data_custom")) {
            fs::create_directory("data_custom");
        }

        std::size_t found = path.find_last_of("/");

        std::string newFileName = "data_custom/" + path.substr(found+1);

        FILE* test = std::fopen(newFileName.c_str(), "w");
        std::random_device rd; // @Simon: necessary to initialize the VarProj algorithm
        std::mt19937 gen(rd());


        if (fptr == nullptr) {
            LOG(FATAL) << "Could not open '{}'"_format(path);
        };
        try {
            // parse header
            int num_cams;
            int num_lms;
            int num_obs;
            fscan_or_throw(fptr, "%d", &num_cams);
            fscan_or_throw(fptr, "%d", &num_lms);
            fscan_or_throw(fptr, "%d", &num_obs);
            fprintf(test,"%d",num_cams);
            fprintf(test, "%s", " ");
            fprintf(test,"%d",num_lms);
            fprintf(test, "%s", " ");
            fprintf(test,"%d",num_obs);
            CHECK_GT(num_cams, 0);
            CHECK_GT(num_lms, 0);
            CHECK_GT(num_obs, 0);
            // clear memory and re-allocate
            if (cameras_.capacity() > unsigned_cast(num_cams)) {
                decltype(cameras_)().swap(cameras_);
            }
            if (landmarks_.capacity() > unsigned_cast(num_lms)) {
                decltype(landmarks_)().swap(landmarks_);
            }
            cameras_.resize(num_cams);
            landmarks_.resize(num_lms);
            // parse observations
            for (int i = 0; i < num_obs; ++i) {
                int cam_idx;
                int lm_idx;
                fscan_or_throw(fptr, "%d", &cam_idx);
                fprintf(test,"\n%d", cam_idx );
                fprintf(test, "%s", " ");
                fscan_or_throw(fptr, "%d", &lm_idx);
                fprintf(test,"%d", lm_idx );
                fprintf(test, "%s", " ");
                CHECK_GE(cam_idx, 0);
                CHECK_LT(cam_idx, num_cams);
                CHECK_GE(lm_idx, 0);
                CHECK_LT(lm_idx, num_lms);

                auto [obs, inserted] = landmarks_.at(lm_idx).obs.try_emplace(cam_idx);
                /// create <landmark, <cam>>
                landmarks_.at(lm_idx).linked_cam.insert(cam_idx);

                CHECK(inserted) << "Invalid file '{}'"_format(path);
                Eigen::Matrix<double, 2, 1> posd;
                fscan_or_throw(fptr, posd);
                fprintf(test,"%lf", posd[0]);
                fprintf(test, "%s", " ");
                fprintf(test,"%lf", posd[1]);
                obs->second.pos = posd.cast<Scalar>();

                // For the camera frame we assume the positive z axis pointing
                // forward in view direction and in the image, y is poiting down, x to the
                // right. In the original BAL formulation, the camera points in negative z
                // axis, y is up in the image. Thus when loading the data, we invert the y
                // and z camera axes (y also in the image) in the perspective projection,
                // we don't have the "minus" like in the original Snavely model.

                // invert y axis
                obs->second.pos.y() = -obs->second.pos.y();
            }

            // parse camera parameters
            for (int i = 0; i < num_cams; ++i) {

                std::normal_distribution<double> d(0, 1); // @Simon: generate normal distribution

                Vec15 params;
                Eigen::Matrix<double, 9, 1> paramsd;
                fscan_or_throw(fptr, paramsd);

                for (int m = 0; m < 15; ++m) {
                    params[m] = d(gen);
                }
                auto& cam = cameras_.at(i);
                cam.space_matrix.row(0) = params.template head<4>();
                cam.space_matrix.row(1) = params.template segment<4>(4);
                cam.space_matrix(2,0) = 0;
                cam.space_matrix(2,1) = 0;
                cam.space_matrix(2,2) = 0;
                cam.space_matrix(2,3) = 1;
                fprintf(test,"\n%lf", cam.space_matrix(0,0));
                fprintf(test,"\n%lf", cam.space_matrix(0,1));
                fprintf(test,"\n%lf", cam.space_matrix(0,2));
                fprintf(test,"\n%lf", cam.space_matrix(0,3));
                fprintf(test,"\n%lf", cam.space_matrix(1,0));
                fprintf(test,"\n%lf", cam.space_matrix(1,1));
                fprintf(test,"\n%lf", cam.space_matrix(1,2));
                fprintf(test,"\n%lf", cam.space_matrix(1,3));
                fprintf(test,"\n%lf", cam.space_matrix(2,0));
                fprintf(test,"\n%lf", cam.space_matrix(2,1));
                fprintf(test,"\n%lf", cam.space_matrix(2,2));
                fprintf(test,"\n%lf", cam.space_matrix(2,3));

                cam.intrinsics = CameraModel(paramsd.cast<Scalar>().template tail<3>());
                fprintf(test,"\n%lf", paramsd.template tail<3>()[0]);
                fprintf(test,"\n%lf", paramsd.template tail<3>()[1]);
                fprintf(test,"\n%lf", paramsd.template tail<3>()[2]);
            }

            // parse landmark parameters
            for (int i = 0; i < num_lms; ++i) { //@Simon: unnecessary for VarProj. We should create a function to derive v*(u0) to get the initialization (u0,v*(u0))
                Eigen::Matrix<double, 3, 1> p_wd;
                fscan_or_throw(fptr, p_wd);
                landmarks_.at(i).p_w = p_wd.cast<Scalar>();
                fprintf(test,"\n%lf",p_wd[0] );
                fprintf(test,"\n%lf",p_wd[1] );
                fprintf(test,"\n%lf",p_wd[2] );
            }

            /// build sparse graph associated to the SC
            for (int i = 0; i < num_lms; ++i)
            {
                Landmark lm = landmarks_.at(i);
                std::unordered_set<int> const & linked_cam_i = lm.linked_cam;
                std::unordered_set<int>::const_iterator it = linked_cam_i.begin();
                for (; it != linked_cam_i.end(); it++)
                {
                    int cam_idx1 = *it;
                    std::unordered_set<int>::const_iterator it2 = linked_cam_i.begin();
                    for (; it2 != linked_cam_i.end(); it2++)
                    {
                        int cam_idx2 = *it2;
                        if (cam_idx1 != cam_idx2)
                        {
                            cameras_.at(cam_idx1).linked_cameras.insert(cam_idx2);
                        }
                    }
                }

            }
        } catch (const std::exception& e) {
            LOG(FATAL) << "Failed to parse '{}'"_format(path);
        }

        if (!quiet_) {
            LOG(INFO)
                    << "Loaded BAL problem ({} cams, {} lms, {} obs) from '{}'"_format(
                            num_cameras(), num_landmarks(), num_observations(), path);
        }
        // Current implementation uses int to compute state vector indices
        CHECK_LT(num_cameras(), std::numeric_limits<int>::max() / CAM_STATE_SIZE);
        std::fclose(test);
        std::fclose(fptr);
    }


template <typename Scalar>
bool BalProblem<Scalar>::save_rootba(const std::string& path) {
  if constexpr (std::is_same_v<Scalar, double>) {
    return BalProblemSaver(path, *this).save();
  } else {
    auto temp = copy_cast<double>();
    return BalProblemSaver(path, temp).save();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::normalize(const double new_scale) {
  // TODO: try out normalization mentioned in MCBA paper to see if it has
  // additional benefit on numerics (note that we already have jacobian scaling)

  // compute median point coordinates (x,y,z)
  std::vector<Scalar> tmp(num_landmarks());
  Vec3 median;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < num_landmarks(); ++i) {
      tmp[i] = landmarks_[i].p_w(j);
    }
    median(j) = median_destructive(tmp);
  }

  // compute median absolute deviation (l1-norm)
  for (int i = 0; i < num_landmarks(); ++i) {
    tmp[i] = (landmarks_[i].p_w - median).template lpNorm<1>();
  }
  const Scalar median_abs_deviation = median_destructive(tmp);

  // normalize scale to constant
  const Scalar scale = new_scale / median_abs_deviation;

  if (!quiet_) {
    LOG(INFO) << "Normalizing BAL problem (median: " << median.transpose()
              << ", MAD: " << median_abs_deviation << ", scale: " << scale
              << ")";
  }


  for (auto& lm : landmarks_) {
    lm.p_w = scale * (lm.p_w - median);
      }


  // update cameras: center = scale * (center - median)
  for (auto& cam : cameras_) {
    SE3 T_w_c = cam.T_c_w.inverse();
    T_w_c.translation() = scale * (T_w_c.translation() - median);
    cam.T_c_w = T_w_c.inverse();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::filter_obs(const double threshold) {
  CHECK_GE(threshold, 0.0);

  if (threshold > 0) {
    if (!quiet_) {
      LOG(INFO) << "Filtering observations with z < {}"_format(threshold);
    }
  } else {
    return;
  }

  // Remove observations with depth of 3D point in camera frame closer than
  // threshold.
  for (auto& lm : landmarks_) {
    for (auto it = lm.obs.cbegin(); it != lm.obs.cend();) {
      const auto& cam = cameras_.at(it->first);
      Vec3 p3d_cam = cam.T_c_w * lm.p_w;

      if (p3d_cam.z() < threshold) {
        it = lm.obs.erase(it);
      } else {
        ++it;
      }
    }
  }

  Landmarks filtered_landmarks;

  // Filter landmarks with number of observations less than 2
  std::copy_if(landmarks_.begin(), landmarks_.end(),
               std::back_inserter(filtered_landmarks),
               [](const auto& lm) { return lm.obs.size() >= 2; });

  landmarks_ = std::move(filtered_landmarks);
}

template <typename Scalar>
void BalProblem<Scalar>::perturb(double rotation_sigma,
                                 double translation_sigma,
                                 double landmark_sigma, int seed) {
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);
  CHECK_GE(landmark_sigma, 0.0);

  if (rotation_sigma > 0 || translation_sigma > 0 || landmark_sigma > 0) {
    if (!quiet_) {
      LOG(INFO) << "Perturbing state (seed: {}): R: {}, t: {}, p: {}"
                   ""_format(seed, rotation_sigma, translation_sigma,
                             landmark_sigma);
    }
  }

  std::random_device r;
  std::default_random_engine eng =
      seed < 0
          ? std::default_random_engine{std::random_device{}()}
          : std::default_random_engine{
                static_cast<std::default_random_engine::result_type>(seed)};

  if (rotation_sigma > 0 || translation_sigma > 0) {
    for (auto& cam : cameras_) {
      // perturb camera center in world coordinates
      if (translation_sigma > 0) {
        SE3 T_w_c = cam.T_c_w.inverse();
        T_w_c.translation() += perturbation<Scalar, 3>(translation_sigma, eng);
        cam.T_c_w = T_w_c.inverse();
      }
      // local rotation perturbation in camera frame
      if (rotation_sigma > 0) {
        cam.T_c_w.so3() =
            SO3::exp(perturbation<Scalar, 3>(rotation_sigma, eng)) *
            cam.T_c_w.so3();
      }
    }
  }

  // perturb landmarks
  if (landmark_sigma > 0) {
    for (auto& lm : landmarks_) {
      lm.p_w += perturbation<Scalar, 3>(landmark_sigma, eng);
    }
  }
}

template <class Scalar>
void BalProblem<Scalar>::postprocress(const BalDatasetOptions& options,
                                      PipelineTimingSummary* timing_summary) {
  Timer t;

  if (options.save_output) {
    save_rootba(options.output_optimized_path);
  }

  if (timing_summary) {
    timing_summary->postprocess_time = t.elapsed();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::copy_to_camera_state(VecX& camera_state) const {
  CHECK_EQ(camera_state.size(), num_cameras() * CAM_STATE_SIZE);
  for (int i = 0; i < num_cameras(); ++i) {
    auto& cam = cameras_[i];
    camera_state.template segment<CAM_STATE_SIZE>(i * CAM_STATE_SIZE) =
        cam.params();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::copy_from_camera_state(const VecX& camera_state) {
  CHECK_EQ(camera_state.size(), num_cameras() * CAM_STATE_SIZE);
  for (int i = 0; i < num_cameras(); ++i) {
    auto& cam = cameras_[i];
    cam.from_params(
        camera_state.template segment<CAM_STATE_SIZE>(i * CAM_STATE_SIZE));
  }
}

template <typename Scalar>
void BalProblem<Scalar>::backup() {
  for (auto& cam : cameras_) {
    cam.backup();
  }
  for (auto& lm : landmarks_) {
    lm.backup();
  }
}

    template <typename Scalar>
    void BalProblem<Scalar>::backup_joint() {
        for (auto& cam : cameras_) {
            cam.backup_joint();
        }
        for (auto& lm : landmarks_) {
            lm.backup_joint();
        }
    }



    template <typename Scalar>
    void BalProblem<Scalar>::backup_pOSE() {
        for (auto& cam : cameras_) {
            cam.backup_pOSE();
        }
        for (auto& lm : landmarks_) {
            lm.backup_pOSE();
        }
    }


template <typename Scalar>
void BalProblem<Scalar>::restore() {
  for (auto& cam : cameras_) {
    cam.restore();
  }
  for (auto& lm : landmarks_) {
    lm.restore();
  }
}

    template <typename Scalar>
    void BalProblem<Scalar>::restore_joint() {
        for (auto& cam : cameras_) {
            cam.restore_joint();
        }
        for (auto& lm : landmarks_) {
            lm.restore_joint();
        }
    }

    template <typename Scalar>
    void BalProblem<Scalar>::restore_pOSE() {
        for (auto& cam : cameras_) {
            cam.restore_pOSE();
        }
        for (auto& lm : landmarks_) {
            lm.restore_pOSE();
        }
    }

template <typename Scalar>
int BalProblem<Scalar>::num_observations() const {
  int num = 0;
  for (auto& lm : landmarks_) {
    num += lm.obs.size();
  }
  return num;
}

template <typename Scalar>
int BalProblem<Scalar>::max_num_observations_per_lm() const {
  int num = 0;
  for (auto& lm : landmarks_) {
    num = std::max(num, static_cast<int>(lm.obs.size()));
  }
  return num;
}

namespace {  // helper

template <class T>
struct is_map {
  static constexpr bool value = false;
};

template <class Key, class Value>
struct is_map<std::map<Key, Value>> {
  static constexpr bool value = true;
};

// NOLINTNEXTLINE
struct default_initialized_atomic_bool : public std::atomic<bool> {
  default_initialized_atomic_bool() { store(false, std::memory_order_relaxed); }
};

}  // namespace

template <typename Scalar>
double BalProblem<Scalar>::compute_rcs_sparsity() const {
  const int num_cams = num_cameras();
  const int num_rcs_blocks = num_cams * num_cams;

  // Note: absl::flat_hash_set<int> is noticably faster than
  // std::unordered_set<int> and a lot faster than
  // std::unordered_set<pair<size_t, size_t>>. An array of bool is a lot
  // faster still, but might need a lot of memory for problems with many
  // cameras.

#if 0
  // absl::flat_hash_set<int> cam_pairs;
  Eigen::VectorX<bool> mask =
      Eigen::VectorX<bool>::Constant(num_rcs_blocks, false);

  for (const auto& lm : landmarks_) {
    for (const auto& [cam_idx_i, _] : lm.obs) {
      for (const auto& [cam_idx_j, _] : lm.obs) {
        if (cam_idx_j < cam_idx_i) {
          int index = cam_idx_i * num_cams + cam_idx_j;
          // cam_pairs.emplace(index);
          mask(index) = true;
        } else {
          // NOTE: the early abort with 'break' assumes ordered lm.obs
          static_assert(is_map<decltype(lm.obs)>::value);
          break;
        }
      }
    }
  }

  // const int num_non_zero_rcs_blocks = num_cams + 2 * cam_pairs.size();
  const int num_non_zero_rcs_blocks = num_cams + 2 * mask.count();
#else

  std::vector<default_initialized_atomic_bool> mask2(num_rcs_blocks);

  // TODO: verify that we really don't need memory barrier before and after
  // parallel for

  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      const auto& lm = landmarks_[r];
      for (const auto& [cam_idx_i, _] : lm.obs) {
        for (const auto& [cam_idx_j, _] : lm.obs) {
          if (cam_idx_j < cam_idx_i) {
            int index = cam_idx_i * num_cams + cam_idx_j;
            mask2[index].store(true, std::memory_order_relaxed);
          } else {
            // NOTE: the early abort with 'break' assumes ordered lm.obs
            static_assert(is_map<decltype(lm.obs)>::value);
            break;
          }
        }
      }
    }
  };

  tbb::blocked_range<size_t> range(0, landmarks_.size());
  tbb::parallel_for(range, body);

  const int num_non_zero_rcs_blocks =
      num_cams + 2 * std::count(mask2.begin(), mask2.end(), true);
#endif

  return 1. - num_non_zero_rcs_blocks / double(num_rcs_blocks);
}

template <class Scalar>
void BalProblem<Scalar>::summarize_problem(DatasetSummary& summary,
                                           bool compute_sparsity) const {
  summary.type = "bal";
  summary.num_cameras = num_cameras();
  summary.num_landmarks = num_landmarks();
  summary.num_observations = num_observations();

  if (compute_sparsity) {
    // can be a bit expensive for dense problems, so compute only when needed
    Timer timer;
    summary.rcs_sparsity = compute_rcs_sparsity();

    if (!quiet_) {
      // output runtime for this computation, b/c it can be quite large for
      // denser problems (so we notice when we should work in improving runtime)
      LOG(INFO) << "Computed RCS sparsity: {:.2f} ({:.3f}s)"_format(
          summary.rcs_sparsity, timer.elapsed());
    }
  }

  auto stats = [](const ArrXd& data) {
    DatasetSummary::Stats res;
    res.mean = data.mean();
    res.min = data.minCoeff();
    res.max = data.maxCoeff();
    res.stddev = std::sqrt((data - res.mean).square().sum() / data.size());
    return res;
  };

  // per landmark observation stats
  {
    ArrXd per_lm_obs(num_landmarks());
    for (int i = 0; i < num_landmarks(); ++i) {
      per_lm_obs(i) = landmarks_.at(i).obs.size();
    }
    summary.per_lm_obs = stats(per_lm_obs);
    CHECK_NEAR(summary.per_lm_obs.mean,
               double(num_observations()) / num_landmarks(), 1e-9);
  }

  // no per hostframe landmark stats
  summary.per_host_lms = DatasetSummary::Stats();
}

template <typename Scalar>
std::string BalProblem<Scalar>::stats_to_string() const {
  DatasetSummary summary;
  summarize_problem(summary, false);

  return "BAL problem stats: {} cams, {} lms, {} obs, per-lm-obs: "
         "{:.1f}+-{:.1f}/{}/{}"
         ""_format(num_cameras(), num_landmarks(), num_observations(),
                   summary.per_lm_obs.mean, summary.per_lm_obs.stddev,
                   int(summary.per_lm_obs.min), int(summary.per_lm_obs.max));
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary) {
  // random seed
  if (options.random_seed >= 0) {
    std::srand(options.random_seed);
  }

  Timer timer;

  // auto detect input type
  BalDatasetOptions::DatasetType input_type = options.input_type;
  if (BalDatasetOptions::DatasetType::AUTO == input_type) {
    input_type = autodetect_input_type(options.input);
    if (!options.quiet) {
      LOG(INFO) << "Autodetected input dataset type as {}."_format(
          wise_enum::to_string(input_type));
    }
  }

  // load dataset as double
  BalProblem<double> bal_problem;
  bal_problem.set_quiet(options.quiet);
  switch (input_type) {
    case BalDatasetOptions::DatasetType::BAL:
        if (options.create_dataset) {
            //To create random dataset with BAL observations
            bal_problem.load_bal_varproj_space_matrix_write(options.input);
            exit(0);
        }
        else {
            //  To load the created dataset
            bal_problem.load_bal_eccv(options.input);
        }

      break;
    default:
      LOG(FATAL) << "unreachable";
  }

  const double time_load = timer.reset();

  // normalize to fixed scale and center (as double, since there are some
  // overflow issues with float for large problems)
  if (options.normalize) {
      bal_problem.normalize(options.normalization_scale);
  }

  // perturb state if sigmas are positive
  bal_problem.perturb(options.rotation_sigma, options.translation_sigma,
                      options.point_sigma, options.random_seed);

  // Filter observations of points closer than threshold to the camera
  bal_problem.filter_obs(options.init_depth_threshold);

  // convert to Scalar if needed
  BalProblem<Scalar> res;
  if constexpr (std::is_same_v<Scalar, double>) {
    res = std::move(bal_problem);
  } else {
    res = bal_problem.copy_cast<Scalar>();
  }

  const double time_preprocess = timer.reset();

  if (timing_summary) {
    timing_summary->load_time = time_load;
    timing_summary->preprocess_time = time_preprocess;
  }

  if (dataset_summary) {
    dataset_summary->input_path = options.input;
    res.summarize_problem(*dataset_summary, true);
  }

  // print some info
  if (!options.quiet) {
    LOG(INFO) << res.stats_to_string();
  }

  return res;
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary) {
  BalDatasetOptions options;
  options.input = path;
  return load_normalized_bal_problem<Scalar>(options, dataset_summary,
                                             timing_summary);
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem_quiet(const std::string& path) {
  BalDatasetOptions options;
  options.input = path;
  options.quiet = true;
  return load_normalized_bal_problem<Scalar>(options);
}

#ifdef ROOTBA_INSTANTIATIONS_FLOAT
template class BalProblem<float>;

template BalProblem<float> load_normalized_bal_problem<float>(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<float> load_normalized_bal_problem<float>(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<float> load_normalized_bal_problem_quiet<float>(
    const std::string& path);
#endif

// BalProblem in double is used by the ceres solver and GUI, so always
// compile it; it should not be a big compilation overhead.
//#ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template class BalProblem<double>;

template BalProblem<double> load_normalized_bal_problem<double>(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<double> load_normalized_bal_problem<double>(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<double> load_normalized_bal_problem_quiet<double>(
    const std::string& path);
//#endif

}  // namespace rootba_povar
