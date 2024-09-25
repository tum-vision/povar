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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glog/logging.h>
#include <pangolin/display/image_view.h>
#include <pangolin/pangolin.h>

#include "rootba/bal/bal_app_options.hpp"
#include "rootba/bal/bal_problem.hpp"
#include "rootba/cli/bal_cli_utils.hpp"
#include "rootba/pangolin/bal_factor_map_display.hpp"
#include "rootba/pangolin/gui_helpers.hpp"
#include "rootba/sc/fp_tree.hpp"
#include "rootba/sc/landmark_block.hpp"

constexpr int POSE_SIZE = 9;
using Scalar = double;

// Pangolin variables
constexpr int UI_WIDTH = 200;

int main(int argc, char** argv) {
  using namespace rootba;

  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  // parse cli and load config
  BalAppOptions options;
  if (!parse_bal_app_arguments("3D viewer for BAL problem.", argc, argv,
                               options)) {
    return 1;
  }

  // load data
  BalProblem<Scalar> bal_problem =
      load_normalized_bal_problem<Scalar>(options.dataset);
  const size_t num_lms = bal_problem.landmarks().size();

  // group with FPTree
  std::vector<std::unordered_set<size_t>> factors_lms_map;
  std::vector<std::unordered_set<size_t>> factors_poses_map;
  std::unordered_set<size_t> non_factors_lms_map;

  // Contains factor idx where having the most landmarks will be at the front
  std::vector<size_t> factors_sorted_order;
  {
    std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>> landmark_blocks;
    LandmarkBlockSC<Scalar, POSE_SIZE>::Options lsc_options;
    landmark_blocks.resize(num_lms);

    for (size_t i = 0; i < num_lms; ++i) {
      landmark_blocks[i].allocate_landmark(bal_problem.landmarks()[i],
                                           lsc_options);
    }

    std::vector<std::vector<size_t>> lms_obs(num_lms);
    for (size_t i = 0; i < num_lms; ++i) {
      lms_obs[i] = landmark_blocks.at(i).get_pose_idx();
    }

    const size_t num_cams = bal_problem.num_cameras();
    FPTree tree(std::move(lms_obs), num_cams);

    std::vector<std::vector<size_t>> factors_lms;
    std::vector<size_t> factors_pose_lm_idx;
    std::vector<size_t> non_factor_lms;

    tree.get_factors(factors_lms, factors_pose_lm_idx, non_factor_lms);

    factors_lms_map.reserve(factors_lms.size());
    for (const auto& factor_lms : factors_lms) {
      std::unordered_set<size_t> factor_lms_map(factor_lms.begin(),
                                                factor_lms.end());
      factors_lms_map.push_back(std::move(factor_lms_map));
    }

    factors_poses_map.reserve(factors_pose_lm_idx.size());
    for (const size_t lm_idx : factors_pose_lm_idx) {
      const auto& poses = landmark_blocks[lm_idx].get_pose_idx();
      std::unordered_set<size_t> factor_poses_map(poses.begin(), poses.end());
      factors_poses_map.push_back(std::move(factor_poses_map));
    }

    factors_sorted_order.resize(factors_lms.size());
    std::iota(factors_sorted_order.begin(), factors_sorted_order.end(), 0);
    std::sort(factors_sorted_order.begin(), factors_sorted_order.end(),
              [&](const size_t x, const size_t y) {
                return factors_lms_map.at(x).size() >
                       factors_lms_map.at(y).size();
              });

    non_factors_lms_map.reserve(non_factor_lms.size());
    std::copy(non_factor_lms.begin(), non_factor_lms.end(),
              std::inserter(non_factors_lms_map, non_factors_lms_map.end()));
  }

  // setup GUI variables and buttons
  pangolin::Var<int> show_factor("ui.show_factor", 0, 0,
                                 factors_lms_map.size() - 1);

  // Note: values used for visualizations in paper:
  //       point_size = 1, cam_weight = 1, cam_size = 0.5
  pangolin::Var<int> point_size("ui.point_size", 2, 1, 5);
  pangolin::Var<int> cam_weight("ui.cam_weight", 2, 1, 5);
  pangolin::Var<double> cam_size("ui.cam_size", 1.5, 0.5, 5);

  pangolin::CreateWindowAndBind("BAL", 1800, 1000);

  glEnable(GL_DEPTH_TEST);

  pangolin::View& main_display = pangolin::CreateDisplay().SetBounds(
      0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0);

  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                        pangolin::Attach::Pix(UI_WIDTH));

  // 3D visualization (initial camera view optimized to see full map)
  // BAL convention seems to be Y-axis up
  const double w = 640;
  const double h = 480;
  const double f = 400;
  const double initial_zoom = 20;
  pangolin::OpenGlRenderState camera_3d_display(
      pangolin::ProjectionMatrix(w, h, f, f, w / 2, h / 2, 1e-2, 1e5),
      pangolin::ModelViewLookAt(initial_zoom * -10, initial_zoom * 8,
                                initial_zoom * -10, 0, 0, 0, pangolin::AxisY));

  pangolin::View& display_3d =
      pangolin::Display("scene").SetAspect(-w / h).SetHandler(
          new pangolin::Handler3D(camera_3d_display));

  main_display.AddDisplay(display_3d);

  rootba::BalFactorMapDisplay map_display(factors_lms_map, factors_poses_map);

  map_display.update(bal_problem, non_factors_lms_map);

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    display_3d.Activate(camera_3d_display);
    glClearColor(1.0F, 1.0F, 1.0F, 1.0F);
    // Visualize factors with most landmarks first
    map_display.draw(factors_sorted_order.at(show_factor),
                     {point_size, cam_weight, cam_size});
    pangolin::glDrawAxis(Sophus::SE3d().matrix(), 10.0);

    pangolin::FinishFrame();
  }

  return 0;
}
