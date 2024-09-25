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

#include "rootba/pangolin/bal_factor_map_display.hpp"

#include "rootba/pangolin/gui_helpers.hpp"
#include "rootba/util/cast.hpp"

namespace rootba {

namespace {

const u_int8_t G_COLOR_NON_FACTOR[3]{0, 0, 0};    // black
const u_int8_t G_COLOR_CAMERA[3]{250, 0, 0};      // red
const u_int8_t G_COLOR_SELECTED[3]{0, 250, 0};    // green
const u_int8_t G_COLOR_FACTOR[3]{169, 169, 169};  // gray

constexpr double G_CAM_SELECTED_SCALE = 1.5;

}  // namespace

BalFactorMapDisplay::BalFactorMapDisplay(
    const std::vector<std::unordered_set<size_t>>& factors_lms,
    const std::vector<std::unordered_set<size_t>>& factors_poses)
    : factors_lms_(factors_lms), factors_poses_(factors_poses) {}

template <typename Scalar>
void BalFactorMapDisplay::update(
    const BalProblem<Scalar>& bal_problem,
    const std::unordered_set<size_t>& non_factors_lms_map) {
  int initial_size = signed_cast(frames_.size());

  if (initial_size < bal_problem.num_cameras()) {
    // create additional displays
    frames_.reserve(bal_problem.num_cameras());
    for (int i = initial_size; i < bal_problem.num_cameras(); ++i) {
      frames_.push_back(std::make_unique<BalFactorFrameDisplay>(i));
    }
  } else {
    // (possibly) remove displays
    frames_.resize(bal_problem.num_cameras());
  }

  for (int i = 0; i < signed_cast(frames_.size()); ++i) {
    frames_.at(i)->update(bal_problem);
  }

  // update points
  const auto& lmids = bal_problem.landmarks();

  points_.clear();
  points_.reserve(lmids.size());

  factors_points_.clear();
  factors_points_.reserve(lmids.size());

  non_factor_points_.clear();
  non_factor_points_.reserve(lmids.size());

  for (size_t i = 0; i < lmids.size(); ++i) {
    points_.push_back(lmids.at(i).p_w.template cast<float>());

    if (non_factors_lms_map.count(i)) {
      non_factor_points_.push_back(points_.back());
    } else {
      factors_points_.push_back(points_.back());
    }
  }

  factors_buffer_.Clear();
  factors_buffer_.Update(factors_points_);

  non_factor_buffer_.Clear();
  non_factor_buffer_.Update(non_factor_points_);
}

void BalFactorMapDisplay::draw(
    const size_t factor_idx,
    const BalFactorMapDisplay::BalFactorMapDisplayOptions& options) {
  const auto& factor_poses = factors_poses_.at(factor_idx);
  for (int i = 0; i < signed_cast(frames_.size()); ++i) {
    const bool is_selected = (factor_poses.find(i) != factor_poses.end());
    frames_.at(i)->draw(is_selected, options);
  }

  update_buffers(factor_idx);
  draw_pointcloud(options);
}

BalFactorFrameDisplay::BalFactorFrameDisplay(FrameIdx frame_id)
    : frame_id_(frame_id) {}

template <typename Scalar>
void BalFactorFrameDisplay::update(const BalProblem<Scalar>& bal_problem) {
  // update pose
  T_w_c_ = bal_problem.cameras()
               .at(frame_id_)
               .T_c_w.inverse()
               .template cast<double>();
}

void BalFactorFrameDisplay::draw(
    bool selected,
    const BalFactorMapDisplay::BalFactorMapDisplayOptions& options) {
  draw_camera(selected, options);
}

bool BalFactorMapDisplay::update_buffers(const int factor_idx) {
  // early abort if nothing changed
  if (last_factor_idx_ == factor_idx) {
    return false;
  }

  const auto& factor_lms = factors_lms_.at(factor_idx);
  current_factors_points_.clear();
  current_factors_points_.reserve(factor_lms.size());

  for (const size_t lm_idx : factors_lms_.at(factor_idx)) {
    current_factors_points_.push_back(points_.at(lm_idx));
  }

  vertex_buffer_.Clear();
  vertex_buffer_.Update(current_factors_points_);

  last_factor_idx_ = factor_idx;
  return true;
}

void BalFactorFrameDisplay::draw_camera(
    bool selected,
    const BalFactorMapDisplay::BalFactorMapDisplayOptions& options) {
  if (selected) {
    render_camera(T_w_c_.matrix(), options.cam_weight, G_COLOR_SELECTED,
                  options.cam_size * G_CAM_SELECTED_SCALE);
  } else {
    render_camera(T_w_c_.matrix(), options.cam_weight, G_COLOR_CAMERA,
                  options.cam_size);
  }
}

void BalFactorMapDisplay::draw_pointcloud(
    const BalFactorMapDisplay::BalFactorMapDisplayOptions& options) {
  glDisable(GL_LIGHTING);

  int point_size = options.point_size;

  glColor3ubv(G_COLOR_FACTOR);
  glPointSize(point_size);
  factors_buffer_.Bind();
  glVertexPointer(factors_buffer_.count_per_element, factors_buffer_.datatype,
                  0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, factors_buffer_.size());
  glDisableClientState(GL_VERTEX_ARRAY);
  factors_buffer_.Unbind();

  glColor3ubv(G_COLOR_NON_FACTOR);
  glPointSize(point_size);
  non_factor_buffer_.Bind();
  glVertexPointer(non_factor_buffer_.count_per_element,
                  non_factor_buffer_.datatype, 0, nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, non_factor_buffer_.size());
  glDisableClientState(GL_VERTEX_ARRAY);
  non_factor_buffer_.Unbind();

  glColor3ubv(G_COLOR_SELECTED);
  glPointSize(point_size + 5);
  vertex_buffer_.Bind();
  glVertexPointer(vertex_buffer_.count_per_element, vertex_buffer_.datatype, 0,
                  nullptr);
  glEnableClientState(GL_VERTEX_ARRAY);
  glDrawArrays(GL_POINTS, 0, vertex_buffer_.size());
  glDisableClientState(GL_VERTEX_ARRAY);
  vertex_buffer_.Unbind();
}

template void BalFactorMapDisplay::update(
    const BalProblem<double>& bal_problem,
    const std::unordered_set<size_t>& non_factors_lms_map);

template void BalFactorMapDisplay::update(
    const BalProblem<float>& bal_problem,
    const std::unordered_set<size_t>& non_factors_lms_map);

template void BalFactorFrameDisplay::update(
    const BalProblem<double>& bal_problem);

template void BalFactorFrameDisplay::update(
    const BalProblem<float>& bal_problem);

}  // namespace rootba
