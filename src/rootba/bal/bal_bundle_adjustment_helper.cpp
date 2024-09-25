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
#include "rootba/bal/bal_bundle_adjustment_helper.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

#include <cmath>


#include <Eigen/Dense>
#include <Eigen/QR>

namespace rootba {

template <typename Scalar>
std::tuple<Scalar, Scalar>
BalBundleAdjustmentHelper<Scalar>::compute_error_weight(
    const BalResidualOptions& options, Scalar res_squared) {
  // TODO: create small class for computing weights and pre-compute huber
  // threshold squared

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
    std::tuple<Scalar, Scalar>
    BalBundleAdjustmentHelper<Scalar>::compute_error_weight_huber(
            const BalResidualOptions& options, Scalar res_squared) {

        const Scalar thresh = options.huber_parameter;
        const Scalar huber_weight =
                res_squared < thresh * thresh ? 1.0 : thresh / std::sqrt(res_squared);
        const Scalar error =
                0.5 * (2 - huber_weight) * huber_weight * res_squared;
        return {error, huber_weight};
    }

    template <typename Scalar>
    std::tuple<Scalar, Scalar>
    BalBundleAdjustmentHelper<Scalar>::compute_error_weight_cauchy(
            const BalResidualOptions& options, Scalar res_squared) {
            return {log(1.0 + res_squared), 1.0};
    }


///@Simon: TODO: FIX IT WITH Jacobian instead of T_c_w and intrinsics....
template <typename Scalar>
void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm(
        BalProblem<Scalar>& bal_problem) {
    auto body = [&](const tbb::blocked_range<int>& range) {
        for (int r = range.begin(); r != range.end(); ++r) {
            const int lm_id = r;
            const auto& lm = bal_problem.landmarks().at(lm_id);
            //bal_problem.landmarks().at(lm_id).p_w.setZero();
            for (const auto& [frame_id, obs] : lm.obs) {
                const auto& cam = bal_problem.cameras().at(frame_id);
                //v_init = initialize_varproj(obs, cam.T_c_w, cam.intrinsics);
                //bal_problem.landmarks().at(lm_id).p_w += v_init;
                bal_problem.landmarks().at(lm_id).p_w += initialize_varproj(obs.pos, cam.T_c_w, cam.intrinsics);
            }
        }
    };
    tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
    tbb::parallel_for(range, body);


}

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm_affine_space(
            BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //bal_problem.landmarks().at(lm_id).p_w.setZero();
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //v_init = initialize_varproj(obs, cam.T_c_w, cam.intrinsics);
                    //bal_problem.landmarks().at(lm_id).p_w += v_init;
                    bal_problem.landmarks().at(lm_id).p_w += initialize_varproj_affine_space(obs.pos, cam.space_matrix, cam.intrinsics);
                }
            }
        };
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm_pOSE(int alpha,
            BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //bal_problem.landmarks().at(lm_id).p_w.setZero();
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //v_init = initialize_varproj(obs, cam.T_c_w, cam.intrinsics);
                    //bal_problem.landmarks().at(lm_id).p_w += v_init;
                    bal_problem.landmarks().at(lm_id).p_w += initialize_varproj_pOSE(alpha, obs.pos, cam.space_matrix, cam.intrinsics);
                }
            }
        };
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm_RpOSE(double alpha,
                                                                       BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //bal_problem.landmarks().at(lm_id).p_w.setZero();
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //v_init = initialize_varproj(obs, cam.T_c_w, cam.intrinsics);
                    //bal_problem.landmarks().at(lm_id).p_w += v_init;
                    bal_problem.landmarks().at(lm_id).p_w += initialize_varproj_RpOSE(alpha, obs.pos, cam.space_matrix, cam.intrinsics);
                }
            }
        };
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm_expOSE(int alpha,
                                                                       BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //bal_problem.landmarks().at(lm_id).p_w.setZero();
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //v_init = initialize_varproj(obs, cam.T_c_w, cam.intrinsics);
                    //bal_problem.landmarks().at(lm_id).p_w += v_init;
                    bal_problem.landmarks().at(lm_id).p_w += initialize_varproj_expOSE(alpha, obs.pos, obs.y_tilde, lm.p_w, cam.space_matrix, cam.space_matrix, cam.intrinsics);
                    //bal_problem.landmarks().at(lm_id).p_w += initialize_varproj_expOSE_v2(alpha, obs.pos, obs.y_tilde, lm.p_w, cam.space_matrix, cam.space_matrix, cam.intrinsics);
                }
            }
        };
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);


    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm_pOSE_rOSE(int alpha,
                                                                       BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //bal_problem.landmarks().at(lm_id).p_w.setZero();
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //v_init = initialize_varproj(obs, cam.T_c_w, cam.intrinsics);
                    //bal_problem.landmarks().at(lm_id).p_w += v_init;
                    bal_problem.landmarks().at(lm_id).p_w += initialize_varproj_pOSE_rOSE(alpha, obs.pos, cam.space_matrix, cam.intrinsics);
                }
            }
        };
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm_rOSE(int alpha,
                                                                       BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //bal_problem.landmarks().at(lm_id).p_w.setZero();
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //v_init = initialize_varproj(obs, cam.T_c_w, cam.intrinsics);
                    //bal_problem.landmarks().at(lm_id).p_w += v_init;
                    bal_problem.landmarks().at(lm_id).p_w += initialize_varproj_rOSE(alpha, obs.pos, cam.space_matrix, cam.intrinsics);
                }
            }
        };
        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::initialize_varproj_lm_pOSE_homogeneous(int alpha,
                                                                       BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //bal_problem.landmarks().at(lm_id).p_w.setZero();
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //v_init = initialize_varproj(obs, cam.T_c_w, cam.intrinsics);
                    //bal_problem.landmarks().at(lm_id).p_w += v_init;
                    bal_problem.landmarks().at(lm_id).p_w_homogeneous += initialize_varproj_pOSE_homogeneous(alpha, obs.pos, cam.space_matrix, cam.intrinsics);
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
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                bal_problem.landmarks().at(lm_id).p_w.setZero();
            }
        };

        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::setzeros_varproj_lm_homogeneous(
            BalProblem<Scalar>& bal_problem) {
        auto body = [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                bal_problem.landmarks().at(lm_id).p_w_homogeneous.setZero();
            }
        };

        tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
        tbb::parallel_for(range, body);
    }

    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::compute_error(const BalProblem<Scalar>& bal_problem,
                                                          const SolverOptions& options,
                                                        ResidualInfo& error) {
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
                            linearize_point(obs.pos, lm.p_w, cam.T_c_w, cam.intrinsics,
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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_affine_space(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR res;
                    const bool projection_valid =
                            linearize_point_affine_space(obs.pos, lm.p_w, cam.space_matrix, cam.intrinsics,
                                            ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR_pOSE res;
                    const bool projection_valid =
                            linearize_point_pOSE(options.alpha, obs.pos, lm.p_w, cam.space_matrix, cam.intrinsics,
                                                         ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_RpOSE(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR_RpOSE res;
                    const bool projection_valid =
                            linearize_point_RpOSE(options.alpha, obs.pos, lm.p_w, cam.space_matrix, cam.intrinsics,
                                                 ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_RpOSE_refinement(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options, double alpha,
            ResidualInfo& error) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR_RpOSE res;
                    const bool projection_valid =
                            linearize_point_RpOSE_refinement(alpha, obs.pos, obs.rpose_eq, lm.p_w, cam.space_matrix, cam.intrinsics,
                                                  ignore_validity_check, res);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_RpOSE_ML(BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR_RpOSE_ML res;
                    const bool projection_valid =
                            linearize_point_RpOSE_ML(obs.pos, obs.rpose_eq, lm.p_w, cam.space_matrix, cam.intrinsics,
                                                             ignore_validity_check, res);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_metric_upgrade(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error) {
        const bool ignore_validity_check = !options.use_projection_validity_check();
        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const auto& cam = bal_problem.cameras().at(r);
                VecR_metric res;
                const bool projection_valid =
                        linearize_metric_upgrade_v3(bal_problem.h_euclidean().plan_infinity , cam.PH, cam.PHHP, cam.space_matrix_intrinsics,
                                                 cam.alpha, cam.intrinsics, ignore_validity_check, res);//, bal_problem);


                const bool numerically_valid = res.array().isFinite().all();

                const Scalar res_squared = res.squaredNorm();
                const auto [weighted_error, weight] =
                compute_error_weight(options.residual, res_squared);
                error_accu.add(numerically_valid, projection_valid,
                               std::sqrt(res_squared), weighted_error);

                    //Vec3 v_init;

            }

            return error_accu;
        };

        // go over all host frames
        tbb::blocked_range<int> range(0, bal_problem.num_cameras());
        ResidualInfoAccu error_accu = tbb::parallel_reduce(
                range, ResidualInfoAccu(), body, ResidualInfoAccu::join);

        // output accumulated error
        error = error_accu.info;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade(const Mat3& PH, const Mat3& PHHP, const Mat34& space_matrix_intrinsics, const Scalar& alpha, const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                 VecR_metric& res, MatRH* d_res_d_H, MatR_alpha* d_res_d_alpha) {

        if (d_res_d_H) {
            Mat9 K;
            K.setZero();
            K(0,0) = 1;
            K(1,3) = 1;
            K(2,6) = 1;
            K(3,1) = 1;
            K(4,4) = 1;
            K(5,7) = 1;
            K(6,2) = 1;
            K(7,5) = 1;
            K(8,8) = 1;

            Mat9_12 PH_kron_P;
            PH_kron_P.template block<3,4>(0,0) = PH(0,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(0,4) = PH(0,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(0,8) = PH(0,2) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,0) = PH(1,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,4) = PH(1,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,8) = PH(1,2) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,0) = PH(2,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,4) = PH(2,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,8) = PH(2,2) * space_matrix_intrinsics;

            Mat9_12 tmp = alpha * (PH_kron_P + K * PH_kron_P);
            Mat93 dres_dH_tmp;
            dres_dH_tmp.setZero();
            dres_dH_tmp.col(0) = tmp.col(3);
            dres_dH_tmp.col(1) = tmp.col(7);
            dres_dH_tmp.col(2) = tmp.col(11);
            *d_res_d_H = dres_dH_tmp;

        }

        if (d_res_d_alpha) {
            Vec9 phhp_vec;
            phhp_vec.setZero();
            phhp_vec.template segment<3>(0) = PHHP.col(0);
            phhp_vec.template segment<3>(3) = PHHP.col(1);
            phhp_vec.template segment<3>(6) = PHHP.col(2);
            phhp_vec.normalize();
            *d_res_d_alpha = phhp_vec * phhp_vec.transpose();
            //@SImon: Try 1:
            //*d_res_d_alpha = PHHP * PHHP.transpose();

            //*d_res_d_alpha = (PHHP/PHHP.norm()) * (PHHP/PHHP.norm()).transpose(); //@Simon: try this normalization
        }

        bool projection_valid = intr.project_metric(PHHP, alpha, res);

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade_v2(const Mat3& PH, const Mat3& PHHP, const Mat34& space_matrix_intrinsics, const Scalar& alpha, const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                     VecR_metric& res, MatRH* d_res_d_H, MatR_alpha* d_res_d_alpha2, MatR_alpha_v2* d_res_d_alpha) {

        if (d_res_d_H) {
            Mat9 K;
            K.setZero();
            K(0,0) = 1;
            K(1,3) = 1;
            K(2,6) = 1;
            K(3,1) = 1;
            K(4,4) = 1;
            K(5,7) = 1;
            K(6,2) = 1;
            K(7,5) = 1;
            K(8,8) = 1;

            Mat9_12 PH_kron_P;
            PH_kron_P.template block<3,4>(0,0) = PH(0,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(0,4) = PH(0,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(0,8) = PH(0,2) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,0) = PH(1,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,4) = PH(1,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,8) = PH(1,2) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,0) = PH(2,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,4) = PH(2,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,8) = PH(2,2) * space_matrix_intrinsics;

            //@Simon: Je Hyeong
            Mat9_12 tmp = alpha * (PH_kron_P + K * PH_kron_P);

            //@SImno: try:
            //Mat9_12 tmp = alpha * (PH_kron_P + PH_kron_P * K);

            Mat93 dres_dH_tmp;
            dres_dH_tmp.setZero();
            //@Simon: Je Hyeong
            dres_dH_tmp.col(0) = tmp.col(3);
            dres_dH_tmp.col(1) = tmp.col(7);
            dres_dH_tmp.col(2) = tmp.col(11);
            //@SImon: try
            //dres_dH_tmp.col(0) = tmp.col(0);
            //dres_dH_tmp.col(1) = tmp.col(1);
            //dres_dH_tmp.col(2) = tmp.col(2);
            *d_res_d_H = dres_dH_tmp;

        }

        if (d_res_d_alpha) {
            Vec9 phhp_vec;
            phhp_vec.setZero();
            phhp_vec.template segment<3>(0) = PHHP.col(0);
            phhp_vec.template segment<3>(3) = PHHP.col(1);
            phhp_vec.template segment<3>(6) = PHHP.col(2);
            //phhp_vec.normalized();
            *d_res_d_alpha = phhp_vec;// * phhp_vec.transpose();
            //phhp_vec.normalize();
            *d_res_d_alpha2 = phhp_vec * phhp_vec.transpose()/Scalar(phhp_vec.transpose() *phhp_vec);
            //@SImon: Try 1:
            //*d_res_d_alpha = PHHP * PHHP.transpose();

            //*d_res_d_alpha = (PHHP/PHHP.norm()) * (PHHP/PHHP.norm()).transpose(); //@Simon: try this normalization
        }

        bool projection_valid = intr.project_metric(PHHP, alpha, res);

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade_v3(const Vec3& c, const Mat3& PH, const Mat3& PHHP, const Mat34& space_matrix_intrinsics, const Scalar& alpha, const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                        VecR_metric& res, MatRH* d_res_d_H, MatR_alpha* d_res_d_alpha2, MatR_alpha_v2* d_res_d_alpha) {

        if (d_res_d_H) {

            Mat93 dres_dH_tmp;
            dres_dH_tmp.setZero();

            dres_dH_tmp(0,0) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(0,1) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(0,2) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(1,0) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,0) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3)* c(0);
            dres_dH_tmp(1,1) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,1) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(1,2) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,2) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(2,0) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,0) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(2,1) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,1) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(2,2) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,2) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(2);

            dres_dH_tmp(3,0) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,0) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(3,1) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,1) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(3,2) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,2) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(4,0) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(0);
            dres_dH_tmp(4,1) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(4,2) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(5,0) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,0) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(5,1) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,1) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(5,2) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,2) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(2);

            dres_dH_tmp(6,0) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,0) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(6,1) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,1) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(6,2) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,2) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(7,0) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,0) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(0);
            dres_dH_tmp(7,1) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,1) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(7,2) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,2) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(8,0) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(8,1) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(8,2) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(2);

            *d_res_d_H = alpha * dres_dH_tmp;

        }

        if (d_res_d_alpha) {
            Vec9 phhp_vec;
            phhp_vec.setZero();
            phhp_vec.template segment<3>(0) = PHHP.col(0);
            phhp_vec.template segment<3>(3) = PHHP.col(1);
            phhp_vec.template segment<3>(6) = PHHP.col(2);
            //phhp_vec.normalized();
            *d_res_d_alpha = phhp_vec;// * phhp_vec.transpose();
            phhp_vec.normalize();
            *d_res_d_alpha2 = phhp_vec * phhp_vec.transpose();///Scalar(phhp_vec.transpose() *phhp_vec);
            //@SImon: Try 1:
            //*d_res_d_alpha = PHHP * PHHP.transpose();

            //*d_res_d_alpha = (PHHP/PHHP.norm()) * (PHHP/PHHP.norm()).transpose(); //@Simon: try this normalization
        }

        bool projection_valid = intr.project_metric(PHHP, alpha, res);

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_metric_upgrade_v3_pollefeys(const Scalar& f, const Vec3& c, const Mat3& PH, const Mat3& PHHP, const Mat34& space_matrix_intrinsics, const Scalar& alpha, const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                        VecR_metric& res, MatRH* d_res_d_H, MatR_alpha* d_res_d_alpha2, MatR_alpha_v2* d_res_d_alpha) {

        if (d_res_d_H) {
            const Scalar& f2 = f * f;

            Mat93 dres_dH_tmp;
            dres_dH_tmp.setZero();

            dres_dH_tmp(0,0) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(0,0) *= f2;
            dres_dH_tmp(0,1) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(0,1) *= f2;
            dres_dH_tmp(0,2) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(1,0) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,0) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3)* c(0);
            dres_dH_tmp(1,0) *= f2;
            dres_dH_tmp(1,1) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,1) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(1,1) *= f2;
            dres_dH_tmp(1,2) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,2) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(2,0) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,0) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(2,0) *= f2;
            dres_dH_tmp(2,1) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,1) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(2,1) *= f2;
            dres_dH_tmp(2,2) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,2) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(2);

            dres_dH_tmp(3,0) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,0) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(3,0) *= f2;
            dres_dH_tmp(3,1) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,1) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(3,1) *= f2;
            dres_dH_tmp(3,2) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,2) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(4,0) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(0);
            dres_dH_tmp(4,0) *= f2;
            dres_dH_tmp(4,1) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(4,1) *= f2;
            dres_dH_tmp(4,2) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(5,0) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,0) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(5,0) *= f2;
            dres_dH_tmp(5,1) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,1) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(5,1) *= f2;
            dres_dH_tmp(5,2) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,2) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(2);

            dres_dH_tmp(6,0) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,0) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(6,0) *= f2;
            dres_dH_tmp(6,1) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,1) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(6,1) *= f2;
            dres_dH_tmp(6,2) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,2) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(7,0) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,0) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(0);
            dres_dH_tmp(7,0) *= f2;
            dres_dH_tmp(7,1) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,1) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(7,1) *= f2;
            dres_dH_tmp(7,2) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,2) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(8,0) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(8,0) *= f2;
            dres_dH_tmp(8,1) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(8,1) *= f2;
            dres_dH_tmp(8,2) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(2);

            *d_res_d_H = alpha * dres_dH_tmp;

        }

        if (d_res_d_alpha) {
            Vec9 phhp_vec;
            phhp_vec.setZero();
            phhp_vec.template segment<3>(0) = PHHP.col(0);
            phhp_vec.template segment<3>(3) = PHHP.col(1);
            phhp_vec.template segment<3>(6) = PHHP.col(2);
            //phhp_vec.normalized();
            *d_res_d_alpha = phhp_vec;// * phhp_vec.transpose();
            phhp_vec.normalize();
            *d_res_d_alpha2 = phhp_vec * phhp_vec.transpose();///Scalar(phhp_vec.transpose() *phhp_vec);
            //@SImon: Try 1:
            //*d_res_d_alpha = PHHP * PHHP.transpose();

            //*d_res_d_alpha = (PHHP/PHHP.norm()) * (PHHP/PHHP.norm()).transpose(); //@Simon: try this normalization
        }

        bool projection_valid = intr.project_metric(PHHP, alpha, res);

        return projection_valid;
    }


    template <typename Scalar>
    void BalBundleAdjustmentHelper<Scalar>::compute_error_expOSE(
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
                    //Vec3 v_init;
                    VecR_expOSE res;

                    const bool projection_valid =
                            linearize_point_expOSE(options.alpha, obs.pos, obs.y_tilde, lm.p_w, lm.p_w_backup(), cam.space_matrix, cam.space_matrix_backup(),
                                                   cam.intrinsics, ignore_validity_check,
                                                   res, initialization_varproj);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_pOSE_rOSE(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR_pOSE_rOSE res;
                    const bool projection_valid =
                            linearize_point_pOSE_rOSE(options.alpha, obs.pos, lm.p_w, cam.space_matrix, cam.intrinsics,
                                                 ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_rOSE(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR_rOSE res;
                    const bool projection_valid =
                            linearize_point_rOSE(options.alpha, obs.pos, lm.p_w, cam.space_matrix, cam.intrinsics,
                                                 ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_pOSE_homogeneous(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR_pOSE res;
                    const bool projection_valid =
                            linearize_point_pOSE_homogeneous(options.alpha, obs.pos, lm.p_w_homogeneous, cam.space_matrix, cam.intrinsics,
                                                 ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_projective_space(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR res;
                    const bool projection_valid =
                            linearize_point_projective_space(obs.pos, lm.p_w, cam.space_matrix, cam.intrinsics,
                                                         ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR res;
                    const bool projection_valid =
                            linearize_point_projective_space_homogeneous(obs.pos, lm.p_w_homogeneous, cam.space_matrix, cam.intrinsics,
                                                             ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_projective_space_homogeneous_RpOSE(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR res;
                    const bool projection_valid =
                            linearize_point_projective_space_homogeneous_RpOSE(obs.pos, lm.p_w_homogeneous, cam.space_matrix, cam.intrinsics,
                                                                         ignore_validity_check, res, initialization_varproj);//, bal_problem);


                    const bool numerically_valid = res.array().isFinite().all();

                    //options.residual.robust_norm = BalResidualOptions::RobustNorm::CAUCHY;

                    const Scalar res_squared = res.squaredNorm();
                    const auto [weighted_error, weight] =
                    //compute_error_weight_huber(options.residual, res_squared);
                    //compute_error_weight_cauchy(options.residual, res_squared);
                    compute_error_weight(options.residual, res_squared);
                    error_accu.add(numerically_valid, projection_valid,
                                   std::sqrt(res_squared), weighted_error);
                    //error_accu.add(numerically_valid, projection_valid,
                    //               res_squared, weighted_error);
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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_projective_space_homogeneous_RpOSE_test_rotation(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error, bool initialization_varproj) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                //if (initialization_varproj) {
                //    bal_problem.landmarks().at(lm_id).p_w.setZero();
                //}
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR res;
                    const bool projection_valid =
                            linearize_point_projective_space_homogeneous_RpOSE_test_rotation(obs.pos, lm.p_w_homogeneous, cam.space_matrix, cam.intrinsics,
                                                                               ignore_validity_check, res, initialization_varproj);//, bal_problem);


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
    void BalBundleAdjustmentHelper<Scalar>::compute_error_refine(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            ResidualInfo& error) {
        const bool ignore_validity_check = !options.use_projection_validity_check();

        // body for parallel reduce
        auto body = [&](const tbb::blocked_range<int>& range,
                        ResidualInfoAccu error_accu) {
            for (int r = range.begin(); r != range.end(); ++r) {
                const int lm_id = r;
                const auto& lm = bal_problem.landmarks().at(lm_id);
                for (const auto& [frame_id, obs] : lm.obs) {
                    const auto& cam = bal_problem.cameras().at(frame_id);
                    //Vec3 v_init;
                    VecR res;
                    const bool projection_valid =
                            linearize_point_refine(obs.pos, lm.p_w, cam.T_c_w, cam.intrinsics,
                                            ignore_validity_check, res);//, bal_problem);


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

    //@Simon: to apply Riemannian manifold optimization for nonseparable VarProj
    template <typename Scalar> // 'Number' can be 'double' or 'std::complex<double>'
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

//@Simon: initialization VarProj: we derive v*(u0) = (G(u)^T G(u))^-1 G(u)^T obs
    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj(
            const Vec2& obs, const SE3& T_c_w,
            const basalt::BalCamera<Scalar>& intr) {
        Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t] // @Simon: in VarProj, should be G(u)


        const Scalar& f = intr.getParam()[0];
        const Scalar& k1 = intr.getParam()[1];
        const Scalar& k2 = intr.getParam()[2];

        Eigen::Matrix<Scalar,2,4> Gu; // = T_c_w_mat.block<2,4>(0,0);
        Gu.row(0) = T_c_w_mat.row(0);
        Gu.row(1) = T_c_w_mat.row(1);
        //Gu.row(2) = T_c_w_mat.row(2);
        //Gu(2,0) = 0;
        //Gu(2,1) = 0;
        //Gu(2,2) = 0;
        //Gu(2,3) = 1;
        //Vec3 obs_homogeneous = obs.homogeneous();

        //Eigen::Matrix<Scalar,4,2> pseudo_inverse_G = (Gu.transpose() * Gu).inverse() * Gu.transpose();

        Eigen::Matrix<Scalar,4,2> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();

        //Vec4 v_init_hom = pseudo_inverse_G  * obs_homogeneous;
        Vec4 v_init_hom = pseudo_inverse_G  * obs;

        const Scalar mx = v_init_hom[0] / f; //@Simon: to check
        const Scalar my = v_init_hom[1] / f; //@Simon: to check
        const Scalar mz = v_init_hom[2] / f;
//@Simon: is done just after in linearize_point
        //const Scalar mx2 = mx * mx;
        //const Scalar my2 = my * my;

        //const Scalar r2 = mx2 + my2;
        //const Scalar r4 = r2 * r2;

        //const Scalar rp = Scalar(1) + k1 * r2 + k2 * r4;

        //Vec3 v_init = Vec3(f * mx * rp, f * my * rp, mz);
        Vec3 v_init = Vec3(mx, my, mz);

        return v_init;

    }

    //@Simon: initialization VarProj: we derive v*(u0) = (G(u)^T G(u))^-1 G(u)^T obs
    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_affine_space(
            const Vec2& obs, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr) {
        Mat34 T_c_w_mat; // @Simon: should be [R t] // @Simon: in VarProj, should be G(u)
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        T_c_w_mat(2,0) = 0;
        T_c_w_mat(2,1) = 0;
        T_c_w_mat(2,2) = 0;
        T_c_w_mat(2,3) = 1;
        const Scalar& f = intr.getParam()[0];
        const Scalar& k1 = intr.getParam()[1];
        const Scalar& k2 = intr.getParam()[2];

        Eigen::Matrix<Scalar,2,4> Gu; // = T_c_w_mat.block<2,4>(0,0);
        Gu.row(0) = T_c_w_mat.row(0);
        Gu.row(1) = T_c_w_mat.row(1);
        //Gu.row(2) = T_c_w_mat.row(2);
        //Gu(2,0) = 0;
        //Gu(2,1) = 0;
        //Gu(2,2) = 0;
        //Gu(2,3) = 1;
        //Vec3 obs_homogeneous = obs.homogeneous();

        //Eigen::Matrix<Scalar,4,2> pseudo_inverse_G = (Gu.transpose() * Gu).inverse() * Gu.transpose();

        Eigen::Matrix<Scalar,4,2> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();

        //Vec4 v_init_hom = pseudo_inverse_G  * obs_homogeneous;
        Vec4 v_init_hom = pseudo_inverse_G  * obs;

        const Scalar mx = v_init_hom[0]/f;// / f; //@Simon: to check
        const Scalar my = v_init_hom[1]/f;// / f; //@Simon: to check
        const Scalar mz = v_init_hom[2]/f;// / f;
//@Simon: is done just after in linearize_point
        //const Scalar mx2 = mx * mx;
        //const Scalar my2 = my * my;

        //const Scalar r2 = mx2 + my2;
        //const Scalar r4 = r2 * r2;

        //const Scalar rp = Scalar(1) + k1 * r2 + k2 * r4;

        //Vec3 v_init = Vec3(f * mx * rp, f * my * rp, mz);
        Vec3 v_init = Vec3(mx, my, mz);

        return v_init;

    }

    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_pOSE(int alpha,
            const Vec2& obs, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr) {
        Mat4 T_c_w_mat;

        T_c_w_mat.row(0) = sqrt((100 - alpha) / 100.0) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt((100 - alpha) / 100.0) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(alpha/100.0) * T_c_w.row(0);
        T_c_w_mat.row(3) = sqrt(alpha/100.0) * T_c_w.row(1);
        const Scalar& f = intr.getParam()[0];
        const Scalar& k1 = intr.getParam()[1];
        const Scalar& k2 = intr.getParam()[2];

        Eigen::Matrix<Scalar,4,4> Gu = T_c_w_mat;
        //Eigen::Matrix<Scalar,4,3> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();
        //Eigen::Matrix<Scalar,4,4> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();
        //Vec4 v_init_hom = pseudo_inverse_G  * obs_homogeneous;
        Vec4 obs_extended;
        obs_extended.setZero();
        obs_extended(2) = sqrt(alpha/100.0) * obs(0);
        obs_extended(3) = sqrt(alpha/100.0) * obs(1);
        Vec4 v_init_hom = Gu.colPivHouseholderQr().solve(obs_extended);
        //Vec4 v_init_hom = pseudo_inverse_G * obs_extended;
        const Scalar mx = v_init_hom[0]/ v_init_hom[3];
        const Scalar my = v_init_hom[1]/ v_init_hom[3];
        const Scalar mz = v_init_hom[2]/ v_init_hom[3];

        Vec3 v_init = Vec3(mx, my, mz);
        return v_init;

    }

    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_RpOSE(double alpha,
                                                                                         const Vec2& obs, const Mat34& T_c_w,
                                                                                         const basalt::BalCamera<Scalar>& intr) {
        Mat34 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt(1.0-alpha) * ( -obs(1) * T_c_w.row(0) + obs(0) * T_c_w.row(1)) / sqrt(obs(0)*obs(0) * obs(1)*obs(1));
        T_c_w_mat.row(1) = sqrt(alpha) * T_c_w.row(0);
        T_c_w_mat.row(2) = sqrt(alpha) * T_c_w.row(1);

        const Scalar& f = intr.getParam()[0];
        const Scalar& k1 = intr.getParam()[1];
        const Scalar& k2 = intr.getParam()[2];

        Eigen::Matrix<Scalar,3,4> Gu = T_c_w_mat;
         //Eigen::Matrix<Scalar,4,3> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();
        //Eigen::Matrix<Scalar,4,4> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();
        //Vec4 v_init_hom = pseudo_inverse_G  * obs_homogeneous;
        Vec3 obs_extended;
        obs_extended.setZero();
        obs_extended(1) = sqrt(alpha) * obs(0);
        obs_extended(2) = sqrt(alpha) * obs(1);
        Vec4 v_init_hom = Gu.colPivHouseholderQr().solve(obs_extended);
        //Vec4 v_init_hom = pseudo_inverse_G * obs_extended;
        const Scalar mx = v_init_hom[0];/// v_init_hom[3]; //@Simon: double check why v_3 = 0;
        const Scalar my = v_init_hom[1];/// v_init_hom[3];
        const Scalar mz = v_init_hom[2];/// v_init_hom[3];


        Vec3 v_init = Vec3(mx, my, mz);
        return v_init;

    }

    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_expOSE(int alpha,
                                                                                         const Vec2& obs, const Vec3& y_tilde,
                                                                                         const Vec3& lm_p_w_equilibrium,
                                                                                         const Mat34& T_c_w, const Mat34& T_c_w_equilibrium,
                                                                                         const basalt::BalCamera<Scalar>& intr) {
        //Vec3 init_eq = Vec3(obs(0), obs(1), 1);

        //Scalar l_exp_equilibrium = std::exp(Scalar((-obs(0) * T_c_w_equilibrium.row(0) - obs(1) * T_c_w_equilibrium.row(1) - T_c_w_equilibrium.row(2)) * init_eq.homogeneous())/sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));
        Scalar l_exp_equilibrium_real = std::exp(Scalar(-sqrt((obs(0)/100.0*y_tilde(0) + obs(1)/100.0 * y_tilde(1) + y_tilde(2)))));


        Mat34 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt((100 - alpha)/100.0) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt((100 - alpha)/100.0) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(l_exp_equilibrium_real/2.0) * sqrt(alpha/100.0) * (obs(0)/100.0 * T_c_w.row(0) + obs(1)/100.0 * T_c_w.row(1) + T_c_w.row(2)) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
        const Scalar& f = intr.getParam()[0];
        const Scalar& k1 = intr.getParam()[1];
        const Scalar& k2 = intr.getParam()[2];

        Eigen::Matrix<Scalar,3,4> Gu = T_c_w_mat;


        Eigen::Matrix<Scalar,4,3> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();
        //Vec4 v_init_hom = pseudo_inverse_G  * obs_homogeneous;

        Vec3 obs_extended;
        obs_extended.setZero();
        obs_extended(2) -= sqrt(alpha/100.0) * sqrt(l_exp_equilibrium_real/2.0) * (1 + Scalar(sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1))); // / sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));

        //Vec4 v_init_hom = Gu.colPivHouseholderQr().solve(obs_extended);
        Vec4 v_init_hom = pseudo_inverse_G * obs_extended;

        const Scalar mx = v_init_hom[0]/ v_init_hom[3];// / f;
        const Scalar my = v_init_hom[1]/ v_init_hom[3];
        const Scalar mz = v_init_hom[2]/ v_init_hom[3];

        Vec3 v_init = Vec3(mx, my, mz);


        return v_init;

    }

    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_expOSE_v2(int alpha,
                                                                                           const Vec2& obs,
                                                                                           Vec3 y_tilde,
                                                                                           const Vec3& lm_p_w_equilibrium,
                                                                                           const Mat34& T_c_w, const Mat34& T_c_w_equilibrium,
                                                                                           const basalt::BalCamera<Scalar>& intr) {
        //y_tilde = Vec3(obs(0), obs(1), 1);
        Vec3 init_eq = Vec3(obs(0), obs(1), 1);

        //Scalar l_exp_equilibrium = std::exp(Scalar((-obs(0) * T_c_w_equilibrium.row(0) - obs(1) * T_c_w_equilibrium.row(1) - T_c_w_equilibrium.row(2)) * init_eq.homogeneous())/sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));
        Scalar l_exp_equilibrium_real = std::exp(Scalar(-sqrt(obs(0)*obs(0) + obs(1) * obs(1) + 1)));


        Eigen::Matrix<Scalar,3,4> Gu = T_c_w;


        Eigen::Matrix<Scalar,4,3> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();
        //Vec4 v_init_hom = pseudo_inverse_G  * obs_homogeneous;

        //Vec4 v_init_hom = Gu.colPivHouseholderQr().solve(obs_extended);
        Vec4 v_init_hom = pseudo_inverse_G * init_eq;

        const Scalar mx = v_init_hom[0]/ v_init_hom[3];// / f;
        const Scalar my = v_init_hom[1]/ v_init_hom[3];
        const Scalar mz = v_init_hom[2]/ v_init_hom[3];

        Vec3 v_init = Vec3(mx, my, mz);


        return v_init;

    }


    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_pOSE_rOSE(int alpha,
                                                                                         const Vec2& obs, const Mat34& T_c_w,
                                                                                         const basalt::BalCamera<Scalar>& intr) {
        Mat54 T_c_w_mat;

        T_c_w_mat.row(0) = (100 - alpha) / 100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha) / 100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(0);
        T_c_w_mat.row(3) = alpha/100.0 * T_c_w.row(1);
        T_c_w_mat.row(4) = T_c_w.row(2) * 10;
        const Scalar& f = intr.getParam()[0];
        const Scalar& k1 = intr.getParam()[1];
        const Scalar& k2 = intr.getParam()[2];

        Eigen::Matrix<Scalar,5,4> Gu = T_c_w_mat;

        Eigen::Matrix<Scalar,4,5> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();

        Vec5 obs_extended;
        obs_extended.setZero();
        obs_extended(2) = alpha/100.0 * obs(0);
        obs_extended(3) = alpha/100.0 * obs(1);
        obs_extended(4) = 1 * 10;
        Vec4 v_init_hom = pseudo_inverse_G * obs_extended;

        const Scalar mx = v_init_hom[0]/ v_init_hom[3];// / f;
        const Scalar my = v_init_hom[1]/ v_init_hom[3];
        const Scalar mz = v_init_hom[2]/ v_init_hom[3];

        Vec3 v_init = Vec3(mx, my, mz);

        return v_init;

    }

    template <typename Scalar>
    Eigen::Matrix<Scalar,3,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_rOSE(int alpha,
                                                                                         const Vec2& obs, const Mat34& T_c_w,
                                                                                         const basalt::BalCamera<Scalar>& intr) {
        Mat34 T_c_w_mat;
        T_c_w_mat.row(0) = (100 - alpha) / 100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha) / 100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(2);
        const Scalar& f = intr.getParam()[0];
        const Scalar& k1 = intr.getParam()[1];
        const Scalar& k2 = intr.getParam()[2];

        Eigen::Matrix<Scalar,3,4> Gu = T_c_w_mat;

        Eigen::Matrix<Scalar,4,3> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();

        Vec3 obs_extended;
        obs_extended.setZero();
        obs_extended(2) = alpha/100.0;
        //obs_extended(3) = alpha/100.0 * obs(1);
        Vec4 v_init_hom = pseudo_inverse_G * obs_extended;
        const Scalar mx = v_init_hom[0]/ v_init_hom[3];// / f;
        const Scalar my = v_init_hom[1]/ v_init_hom[3];
        const Scalar mz = v_init_hom[2]/ v_init_hom[3];

        Vec3 v_init = Vec3(mx, my, mz);

        return v_init;

    }

    template <typename Scalar>
    Eigen::Matrix<Scalar,4,1> BalBundleAdjustmentHelper<Scalar>::initialize_varproj_pOSE_homogeneous(int alpha,
                                                                                         const Vec2& obs, const Mat34& T_c_w,
                                                                                         const basalt::BalCamera<Scalar>& intr) {
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = (100 - alpha) / 100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha) / 100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(0);
        T_c_w_mat.row(3) = alpha/100.0 * T_c_w.row(1);

        const Scalar& f = intr.getParam()[0];
        const Scalar& k1 = intr.getParam()[1];
        const Scalar& k2 = intr.getParam()[2];

        Eigen::Matrix<Scalar,4,4> Gu = T_c_w_mat;
        Eigen::Matrix<Scalar,4,4> pseudo_inverse_G = Gu.completeOrthogonalDecomposition().pseudoInverse();

        Vec4 obs_extended;
        obs_extended.setZero();
        obs_extended(2) = alpha/100.0 * obs(0);
        obs_extended(3) = alpha/100.0 * obs(1);

        Vec4 v_init_hom = pseudo_inverse_G * obs_extended;

        Vec4 v_init = v_init_hom / f;
        return v_init;

    }


    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point(
            const Vec2& obs, const Vec3& lm_p_w, const SE3& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l) {
        Mat4 T_c_w_mat = T_c_w.matrix();

        Vec4 p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }
        res -= obs;


        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }

        if (d_res_d_xi) {
            Mat<Scalar, 4, POSE_SIZE> d_point_d_xi;
            d_point_d_xi.template topLeftCorner<3, 3>() = Mat3::Identity();
            d_point_d_xi.template topRightCorner<3, 3>() =
                    -SO3::hat(p_c_3d.template head<3>());
            d_point_d_xi.row(3).setZero();
            *d_res_d_xi = d_res_d_p * d_point_d_xi;
        }

        if (d_res_d_l) {
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>();
        }

        return projection_valid;
    }


//@Simon: we consider directly the camera matrix space, and not SE(3). For affine model we need 8 parameters
    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_affine_space(
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, const bool initialization_varproj,
            MatRP_affine_space* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        //@Simon: try to force 3rd line to be [0 0 0 1]
        T_c_w_mat(2,0) = 0;
        T_c_w_mat(2,1) = 0;
        T_c_w_mat(2,2) = 0;
        T_c_w_mat(2,3) = 1;
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;
        //Vec4 p_c_3d;
        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();
        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_affine_matrix_space(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_affine_matrix_space(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }



        res -= obs;

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat48 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = 1;

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;


            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }
//        if (d_res_d_xi) {
//            Mat<Scalar, 4, POSE_SIZE> d_point_d_xi;
//            d_point_d_xi.template topLeftCorner<3, 3>() = Mat3::Identity();
//            d_point_d_xi.template topRightCorner<3, 3>() =
//                    -SO3::hat(p_c_3d.template head<3>());
//            d_point_d_xi.row(2).setZero(); //@Simon : double check. Idea: we mimic an affine camera and then 3rd line is 0 0 0 1
//            d_point_d_xi.row(3).setZero();
//            *d_res_d_xi = d_res_d_p * d_point_d_xi;  // @Simon: double check, but should stay similar for affine BA
//            // @Simon: dres/dxi is perhaps [dF/dtj dF/dphi], id est [Jac_pose_translation Jac_pose_rotation]
//        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE(int alpha,
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR_pOSE& res, const bool initialization_varproj,
            MatRP_pOSE* d_res_d_xi, MatRI_pOSE* d_res_d_i, MatRL_pOSE* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt((100 - alpha)/100.0) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt((100 - alpha)/100.0) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        //T_c_w_mat.row(0) = (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        //T_c_w_mat.row(1) = (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(alpha/100.0) * T_c_w.row(0);
        T_c_w_mat.row(3) = sqrt(alpha/100.0) * T_c_w.row(1);


        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat4 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_pOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_pOSE(p_c_3d, res, nullptr, nullptr);
        }



        res(2) -= sqrt(alpha/100.0) * obs(0);
        res(3) -= sqrt(alpha/100.0) * obs(1);

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
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

            d_p_d_xi.row(0) *= sqrt((100-alpha)/100.0);

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);
            d_p_d_xi.row(1) *= sqrt((100-alpha)/100.0);

            d_p_d_xi(2,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,3) = 1;
            d_p_d_xi.row(2) *= sqrt(alpha/100.0);

            d_p_d_xi(3,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(3,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(3,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(3,7) = 1;
            d_p_d_xi.row(3) *= sqrt(alpha/100.0);

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_RpOSE(double alpha,
                                                                 const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
                                                                 const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                 VecR_RpOSE& res, const bool initialization_varproj,
                                                                 MatRP_RpOSE* d_res_d_xi, MatRI_RpOSE* d_res_d_i, MatRL_RpOSE* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat34 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt(1.0-alpha) * (-obs(1)  * T_c_w.row(0) + obs(0) * T_c_w.row(1)) / sqrt(obs(0) * obs(0) + obs(1)*obs(1));
        T_c_w_mat.row(1) = sqrt(alpha) * T_c_w.row(0);
        T_c_w_mat.row(2) = sqrt(alpha) * T_c_w.row(1);

        Vec3 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat3 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_RpOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_RpOSE(p_c_3d, res, nullptr, nullptr);
        }



        res(1) -= sqrt(alpha) * obs(0);
        res(2) -= sqrt(alpha) * obs(1);

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat38 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = -obs(1) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = -obs(1) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = -obs(1) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = -obs(1);
            d_p_d_xi(0,4) = obs(0) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,5) = obs(0) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,6) = obs(0) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,7) = obs(0);

            d_p_d_xi.row(0) *= sqrt(1.0 - alpha) / sqrt(obs(0)*obs(0) + obs(1)*obs(1));


            d_p_d_xi(1,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,3) = 1;
            d_p_d_xi.row(1) *= sqrt(alpha);

            d_p_d_xi(2,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,7) = 1;
            d_p_d_xi.row(2) *= sqrt(alpha);

            *d_res_d_xi = d_res_d_p * d_p_d_xi;


        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<3, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_RpOSE_refinement(double alpha,
                                                                  const Vec2& obs, const Vec2& rpose_eq, const Vec3& lm_p_w, const Mat34& T_c_w,
                                                                  const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                  VecR_RpOSE& res,
                                                                  MatRP_RpOSE* d_res_d_xi, MatRI_RpOSE* d_res_d_i, MatRL_RpOSE* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat34 T_c_w_mat;
        Scalar norm_eq = sqrt(rpose_eq(0)*rpose_eq(0) + rpose_eq(1)*rpose_eq(1));
        Scalar norm_eq_3 = pow(norm_eq,3);
        Scalar prod_obs_eq = - obs(1) * rpose_eq(0) + obs(0) * rpose_eq(1);
        Vec2 Jac;
        Jac(0) = - obs(1) / norm_eq - prod_obs_eq / norm_eq_3 * rpose_eq(0);
        Jac(1) = obs(0) / norm_eq - prod_obs_eq / norm_eq_3 * rpose_eq(1);
        T_c_w_mat.row(0) = sqrt(1.0-alpha) *(Jac(0) * T_c_w.row(0) + Jac(1) * T_c_w.row(1));
        T_c_w_mat.row(1) = sqrt(alpha) * T_c_w.row(0);
        T_c_w_mat.row(2) = sqrt(alpha) * T_c_w.row(1);
        //T_c_w_mat.row(1) = 0.001 * T_c_w.row(0);
        //T_c_w_mat.row(2) = 0.001 * T_c_w.row(1);

        Vec3 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat3 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_RpOSE_refinement(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_RpOSE_refinement(p_c_3d, res, nullptr, nullptr);
        }


        res(0) -= sqrt(1.0 - alpha) * (Jac(0) * rpose_eq(0) + Jac(1) * rpose_eq(1) - (-obs(1) / norm_eq * rpose_eq(0) + obs(0) / norm_eq * rpose_eq(1)));
        res(1) -= sqrt(alpha) * rpose_eq(0);
        res(2) -= sqrt(alpha) * rpose_eq(1);

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat38 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = Jac(0) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = Jac(0) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = Jac(0) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = Jac(0);
            d_p_d_xi(0,4) = Jac(1) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,5) = Jac(1) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,6) = Jac(1) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,7) = Jac(1);

            d_p_d_xi.row(0) *= sqrt(1.0 - alpha);


            d_p_d_xi(1,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,3) = 1;
            d_p_d_xi.row(1) *= sqrt(alpha);

            d_p_d_xi(2,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,7) = 1;
            d_p_d_xi.row(2) *= sqrt(alpha);

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<3, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_RpOSE_ML(const Vec2& obs, const Vec2& rpose_eq, const Vec3& lm_p_w, const Mat34& T_c_w,
                                                                             const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                             VecR_RpOSE_ML& res,
                                                                             MatRP_RpOSE_ML* d_res_d_xi, MatRI_RpOSE_ML* d_res_d_i, MatRL_RpOSE_ML* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat14 T_c_w_mat;
        Scalar norm_eq = sqrt(rpose_eq(0)*rpose_eq(0) + rpose_eq(1)*rpose_eq(1));
        Scalar norm_eq_3 = pow(norm_eq,3);
        Scalar prod_obs_eq = - obs(1) * rpose_eq(0) + obs(0) * rpose_eq(1);
        Vec2 Jac;
        Jac(0) = - obs(1) / norm_eq - prod_obs_eq / norm_eq_3 * rpose_eq(0);
        Jac(1) = obs(0) / norm_eq - prod_obs_eq / norm_eq_3 * rpose_eq(1);

        T_c_w_mat.row(0) = (Jac(0) * T_c_w.row(0) + Jac(1) * T_c_w.row(1));


        Scalar p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        //Mat13 d_res_d_p;
        bool projection_valid = true;
        //if (d_res_d_xi || d_res_d_i || d_res_d_l) {
        //    projection_valid = intr.project_RpOSE_ML(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        //} else {
        //    projection_valid = intr.project_RpOSE_ML(p_c_3d, res, nullptr, nullptr);
        //}
        //(*d_res_d_i)(0,0) = 0;
        //(*d_res_d_i)(0,1) = 0;
        //(*d_res_d_i)(0,2) = 0;

        res(0) = p_c_3d - (Jac(0) * rpose_eq(0) + Jac(1) * rpose_eq(1) - (-obs(1) / norm_eq * rpose_eq(0) + obs(0) / norm_eq * rpose_eq(1)));


        //if (!ignore_validity_check && !projection_valid) {
        //    return false;
        //}
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat18 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = Jac(0) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = Jac(0) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = Jac(0) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = Jac(0);
            d_p_d_xi(0,4) = Jac(1) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,5) = Jac(1) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,6) = Jac(1) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,7) = Jac(1);



            //*d_res_d_xi = d_res_d_p * d_p_d_xi;
            *d_res_d_xi = d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = T_c_w_mat.template topLeftCorner<1, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }


    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_expOSE(int alpha,
                                                                 const Vec2& obs, const Vec3& y_tilde,
                                                                 const Vec3& lm_p_w, const Vec3& lm_p_w_equilibrium,
                                                                 const Mat34& T_c_w, const Mat34& T_c_w_equilibrium,
                                                                 const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                 VecR_expOSE& res, const bool initialization_varproj,
                                                                 MatRP_expOSE* d_res_d_xi, MatRI_expOSE* d_res_d_i, MatRL_expOSE* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Scalar l_exp_equilibrium;
        if (initialization_varproj) {
            l_exp_equilibrium = std::exp(-Scalar(sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1)));
        }
        else {
            l_exp_equilibrium = std::exp(-Scalar(obs(0)/100.0 * y_tilde(0) + obs(1)/100.0 * y_tilde(1) + y_tilde(2))/sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1));
            //l_exp_equilibrium = std::exp(-Scalar((obs(0) * T_c_w_equilibrium.row(0) + obs(1) * T_c_w_equilibrium.row(1) + T_c_w_equilibrium.row(2)) * lm_p_w_equilibrium.homogeneous())/sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));
        }


        Mat34 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt((100 - alpha)/100.0) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt((100 - alpha)/100.0) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(l_exp_equilibrium / 2.0) * sqrt(alpha/100.0) * (obs(0)/100.0 * T_c_w.row(0) + obs(1)/100.0 * T_c_w.row(1) + T_c_w.row(2)) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
        //T_c_w_mat.row(3) = alpha/100.0 * T_c_w.row(1);



        Vec3 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat3 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_expOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_expOSE(p_c_3d, res, nullptr, nullptr);
        }


        if (initialization_varproj) {
            res(2) -= sqrt(alpha/100.0) * sqrt(l_exp_equilibrium / 2.0) * (1 + sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1));
        }
        else {
            res(2) -= sqrt(alpha/100.0) * sqrt(l_exp_equilibrium / 2.0) * (1 + Scalar((obs(0)/100.0 * y_tilde(0) + obs(1)/100.0 * y_tilde(1) + 1)/ sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1))); // / sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));
        }
        //res(3) -= alpha/100.0 * obs(1);

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat3_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = 1;
            d_p_d_xi(0,8) = -lm_p_w.homogeneous()(0) * obs(0);
            d_p_d_xi(0,9) = -lm_p_w.homogeneous()(1) * obs(0);
            d_p_d_xi(0,10) = -lm_p_w.homogeneous()(2) * obs(0);
            d_p_d_xi(0,11) = - obs(0);

            d_p_d_xi.row(0) *= sqrt((100-alpha)/100.0);

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);
            d_p_d_xi.row(1) *= sqrt((100-alpha)/100.0);

            d_p_d_xi(2,0) = obs(0)/100.0 * lm_p_w.homogeneous()(0) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,1) = obs(0)/100.0 * lm_p_w.homogeneous()(1) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,2) = obs(0)/100.0 * lm_p_w.homogeneous()(2) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,3) = obs(0)/100.0 / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);

            d_p_d_xi(2,4) = obs(1)/100.0 * lm_p_w.homogeneous()(0) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,5) = obs(1)/100.0 * lm_p_w.homogeneous()(1) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,6) = obs(1)/100.0 * lm_p_w.homogeneous()(2) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,7) = obs(1)/100.0 / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);

            d_p_d_xi(2,8) = lm_p_w.homogeneous()(0) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,9) = lm_p_w.homogeneous()(1) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,10) = lm_p_w.homogeneous()(2) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,11) = 1 / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi.row(2) *= sqrt(alpha/100.0) * sqrt(l_exp_equilibrium/2.0);


            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<3, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE_rOSE(int alpha,
                                                                 const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
                                                                 const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                 VecR_pOSE_rOSE& res, const bool initialization_varproj,
                                                                 MatRP_pOSE_rOSE* d_res_d_xi, MatRI_pOSE_rOSE* d_res_d_i, MatRL_pOSE_rOSE* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat54 T_c_w_mat;
        T_c_w_mat.row(0) = (100 - alpha)/100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha)/100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(0);
        T_c_w_mat.row(3) = alpha/100.0 * T_c_w.row(1);
        T_c_w_mat.row(4) = T_c_w.row(2) * 10;


        Vec5 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat5 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_pOSE_rOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_pOSE_rOSE(p_c_3d, res, nullptr, nullptr);
        }



        res(2) -= alpha/100.0 * obs(0);
        res(3) -= alpha/100.0 * obs(1);
        res(4) -= 1.0 * 10;

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat5_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = 1;
            d_p_d_xi(0,8) = -lm_p_w.homogeneous()(0) * obs(0);
            d_p_d_xi(0,9) = -lm_p_w.homogeneous()(1) * obs(0);
            d_p_d_xi(0,10) = -lm_p_w.homogeneous()(2) * obs(0);
            d_p_d_xi(0,11) = - obs(0);

            d_p_d_xi.row(0) *= (100-alpha)/100.0;

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);
            d_p_d_xi.row(1) *= (100-alpha)/100.0;

            d_p_d_xi(2,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,3) = 1;
            d_p_d_xi.row(2) *= alpha/100.0;

            d_p_d_xi(3,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(3,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(3,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(3,7) = 1;
            d_p_d_xi.row(3) *= alpha/100.0;

            d_p_d_xi(4,8) = lm_p_w.homogeneous()(0);
            d_p_d_xi(4,9) = lm_p_w.homogeneous()(1);
            d_p_d_xi(4,10) = lm_p_w.homogeneous()(2);
            d_p_d_xi(4,11) = 1;
            d_p_d_xi.row(4) *= 10;


            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<5, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_rOSE(int alpha,
                                                                 const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
                                                                 const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                 VecR_rOSE& res, const bool initialization_varproj,
                                                                 MatRP_rOSE* d_res_d_xi, MatRI_rOSE* d_res_d_i, MatRL_rOSE* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = (100 - alpha)/100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha)/100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        //T_c_w_mat.row(0) = (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        //T_c_w_mat.row(1) = (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(2);
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;


        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat34 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_rOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_rOSE(p_c_3d, res, nullptr, nullptr);
        }



        res(2) -= alpha/100.0;

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
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

            d_p_d_xi.row(0) *= (100-alpha)/100.0;

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);
            d_p_d_xi.row(1) *= (100-alpha)/100.0;

            d_p_d_xi(2,8) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,9) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,10) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,11) = 1;
            d_p_d_xi.row(2) *= alpha/100.0;


            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_pOSE_homogeneous(int alpha,
                                                                 const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w,
                                                                 const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                 VecR_pOSE& res, const bool initialization_varproj,
                                                                 MatRP_pOSE* d_res_d_xi, MatRI_pOSE* d_res_d_i, MatRL_pOSE_homogeneous* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = (100 - alpha)/100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha)/100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(0);
        T_c_w_mat.row(3) = alpha/100.0 * T_c_w.row(1);


        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w;
        Mat4 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_pOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_pOSE(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }



        res(2) -= alpha/100.0 * obs(0);
        res(3) -= alpha/100.0 * obs(1);


        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat4_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w(0);
            d_p_d_xi(0,1) = lm_p_w(1);
            d_p_d_xi(0,2) = lm_p_w(2);
            d_p_d_xi(0,3) = lm_p_w(3);
            d_p_d_xi(0,8) = -lm_p_w(0) * obs(0);
            d_p_d_xi(0,9) = -lm_p_w(1) * obs(0);
            d_p_d_xi(0,10) = -lm_p_w(2) * obs(0);
            d_p_d_xi(0,11) = - lm_p_w(3) * obs(0);

            d_p_d_xi.row(0) *= (100-alpha)/100.0;

            d_p_d_xi(1,4) = lm_p_w(0);
            d_p_d_xi(1,5) = lm_p_w(1);
            d_p_d_xi(1,6) = lm_p_w(2);
            d_p_d_xi(1,7) = lm_p_w(3);
            d_p_d_xi(1,8) = -lm_p_w(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w(1) * obs(1);
            d_p_d_xi(1,10) = -lm_p_w(2) * obs(1);
            d_p_d_xi(1,11) = -lm_p_w(3) * obs(1);
            d_p_d_xi.row(1) *= (100-alpha)/100.0;

            d_p_d_xi(2,0) = lm_p_w(0);
            d_p_d_xi(2,1) = lm_p_w(1);
            d_p_d_xi(2,2) = lm_p_w(2);
            d_p_d_xi(2,3) = lm_p_w(3);
            d_p_d_xi.row(2) *= alpha/100.0;

            d_p_d_xi(3,4) = lm_p_w(0);
            d_p_d_xi(3,5) = lm_p_w(1);
            d_p_d_xi(3,6) = lm_p_w(2);
            d_p_d_xi(3,7) = lm_p_w(3);
            d_p_d_xi.row(3) *= alpha/100.0;

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat; // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }


    //@Simon: we consider directly the camera matrix space, and not SE(3). For projective model we need 11 (or 12 with t?) parameters
    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space(
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, const bool initialization_varproj,
            MatRP_projective_space* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        T_c_w_mat.row(2) = T_c_w.row(2); //@Simon: add the parameter s_i as in the paper
        //T_c_w_mat(2,3) = 1; //@Simon: not necessarily for homogeneous model
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;
        Vec4 p_c_3d;

        //Vec4 lm_p_w_proj;
        //lm_p_w_proj(0) = lm_p_w(0);
        //lm_p_w_proj(1) = lm_p_w(1);
        //lm_p_w_proj(2) = lm_p_w(2);
        //lm_p_w_proj(3) =
        //Scalar lm_p_w_proj = 1 / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
        //lm_p_w.push_back(lm_p_w_proj);
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous() / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
        //p_c_3d = T_c_w_mat * lm_p_w;
        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, nullptr, nullptr);
        }


        res -= obs;


        Scalar lm_p_w_0 = lm_p_w(0) / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
        Scalar lm_p_w_1 = lm_p_w(1) / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
        Scalar lm_p_w_2 = lm_p_w(2) / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
        Scalar lm_p_w_3 = 1  / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        if (d_res_d_xi) {

//@Simon: TRY2
            Mat4_12 d_p_d_xi;
            d_p_d_xi.setZero();
            //d_p_d_xi.row(0).setZero();
            //d_p_d_xi.row(1).setZero();
            //d_p_d_xi.row(2).setZero();
            //d_p_d_xi.row(3).setZero();
            d_p_d_xi(0,0) = lm_p_w_0;
            d_p_d_xi(0,1) = lm_p_w_1;
            d_p_d_xi(0,2) = lm_p_w_2;
            d_p_d_xi(0,3) = lm_p_w_3;

            d_p_d_xi(1,4) = lm_p_w_0;
            d_p_d_xi(1,5) = lm_p_w_1;
            d_p_d_xi(1,6) = lm_p_w_2;
            d_p_d_xi(1,7) = lm_p_w_3;

            d_p_d_xi(2,8) = lm_p_w_0;
            d_p_d_xi(2,9) = lm_p_w_1;
            d_p_d_xi(2,10) = lm_p_w_2;
            d_p_d_xi(2,11) = lm_p_w_3;

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }


        if (d_res_d_l) {
            Mat43 d_p_d_l;
            d_p_d_l.setZero();

            d_p_d_l(0,0) = T_c_w_mat(0,0) * (lm_p_w_3 - lm_p_w(0) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(0,0) -= T_c_w_mat(0,1) * lm_p_w(1) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,0) -= T_c_w_mat(0,2) * lm_p_w(2) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,0) -= T_c_w_mat(0,3) *  lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(0,1) = T_c_w_mat(0,1) * (lm_p_w_3 - lm_p_w(1) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(0,1) -= T_c_w_mat(0,0) * lm_p_w(0) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,1) -= T_c_w_mat(0,2) * lm_p_w(2) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,1) -= T_c_w_mat(0,3)  * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(0,2) = T_c_w_mat(0,2) * (lm_p_w_3 - lm_p_w(2) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(0,2) -= T_c_w_mat(0,0) * lm_p_w(0) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,2) -= T_c_w_mat(0,1) * lm_p_w(1) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,2) -= T_c_w_mat(0,3) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(1,0) = T_c_w_mat(1,0) * (lm_p_w_3 - lm_p_w(0) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(1,0) -= T_c_w_mat(1,1) * lm_p_w(1) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,0) -= T_c_w_mat(1,2) * lm_p_w(2) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,0) -= T_c_w_mat(1,3) *  lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(1,1) = T_c_w_mat(1,1) * (lm_p_w_3 - lm_p_w(1) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(1,1) -= T_c_w_mat(1,0) * lm_p_w(0) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,1) -= T_c_w_mat(1,2) * lm_p_w(2) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,1) -= T_c_w_mat(1,3)  * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(1,2) = T_c_w_mat(1,2) * (lm_p_w_3 - lm_p_w(2) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(1,2) -= T_c_w_mat(1,0) * lm_p_w(0) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,2) -= T_c_w_mat(1,1) * lm_p_w(1) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,2) -= T_c_w_mat(1,3) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(2,0) = T_c_w_mat(2,0) * (lm_p_w_3 - lm_p_w(0) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(2,0) -= T_c_w_mat(2,1) * lm_p_w(1) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,0) -= T_c_w_mat(2,2) * lm_p_w(2) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,0) -= T_c_w_mat(2,3) *  lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(2,1) = T_c_w_mat(2,1) * (lm_p_w_3 - lm_p_w(1) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(2,1) -= T_c_w_mat(2,0) * lm_p_w(0) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,1) -= T_c_w_mat(2,2) * lm_p_w(2) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,1) -= T_c_w_mat(2,3)  * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(2,2) = T_c_w_mat(2,2) * (lm_p_w_3 - lm_p_w(2) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(2,2) -= T_c_w_mat(2,0) * lm_p_w(0) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,2) -= T_c_w_mat(2,1) * lm_p_w(1) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,2) -= T_c_w_mat(2,3) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            *d_res_d_l = d_res_d_p * d_p_d_l;
            //*d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
            // @Simon: is probably Jac_landmark
        }
        return projection_valid;
    }


    //@Simon: for homogeneous coordinates, during refinement with nonlinear VarPro
    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space_homogeneous(
            const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, const bool initialization_varproj,
            MatRP_projective_space* d_res_d_xi, MatRI* d_res_d_i, MatRL_projective_space_homogeneous* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.setZero();
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        T_c_w_mat.row(2) = T_c_w.row(2); //@Simon: add the parameter s_i as in the paper
        //T_c_w_mat(2,3) = 1; //@Simon: not necessarily for homogeneous model
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;
        Vec4 p_c_3d;

        p_c_3d = T_c_w_mat * lm_p_w;


        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, nullptr, nullptr);
        }


        res -= obs;


        if (!ignore_validity_check && !projection_valid) {
            return false;
        }


        if (d_res_d_xi) {
            //@Simon: Jp = d res / d pose
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
            //@Simon: Jl = d res / d landmark
            //*d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>();
            *d_res_d_l = d_res_d_p * T_c_w_mat;
        }

        //assert(projection_valid);
        return projection_valid;
    }

    //@Simon: for homogeneous coordinates, during refinement with nonlinear VarPro
    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space_homogeneous_RpOSE(
            const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, const bool initialization_varproj,
            MatRP_projective_space* d_res_d_xi, MatRI* d_res_d_i, MatRL_projective_space_homogeneous* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.setZero();
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        T_c_w_mat.row(2) = T_c_w.row(2); //@Simon: add the parameter s_i as in the paper
        //T_c_w_mat(2,3) = 1; //@Simon: not necessarily for homogeneous model
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;
        Vec4 p_c_3d;

        p_c_3d = T_c_w_mat * lm_p_w;


        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_projective_refinement_matrix_space_with_distortion(p_c_3d, obs, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_projective_refinement_matrix_space_with_distortion(p_c_3d, obs, res, nullptr, nullptr);
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, nullptr, nullptr);
        }


        res -= obs;


        if (!ignore_validity_check && !projection_valid) {
            return false;
        }


        if (d_res_d_xi) {
            //@Simon: Jp = d res / d pose
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
            //@Simon: Jl = d res / d landmark
            //*d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>();
            *d_res_d_l = d_res_d_p * T_c_w_mat;
        }

        //assert(projection_valid);
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_projective_space_homogeneous_RpOSE_test_rotation(
            const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, const bool initialization_varproj,
            MatRP_projective_space* d_res_d_xi, MatRI* d_res_d_i, MatRL_projective_space_homogeneous* d_res_d_l) {

        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.setZero();
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        T_c_w_mat.row(2) = T_c_w.row(2); //@Simon: add the parameter s_i as in the paper
        //T_c_w_mat(2,3) = 1; //@Simon: not necessarily for homogeneous model
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;
        Vec4 p_c_3d;

        p_c_3d = T_c_w_mat * lm_p_w;


        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_projective_refinement_matrix_space_with_distortion_test_rotation(p_c_3d, obs, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_projective_refinement_matrix_space_with_distortion_test_rotation(p_c_3d, obs, res, nullptr, nullptr);
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, nullptr, nullptr);
        }

        //std::cout << "in test,    before obs,    res = " << res << "\n";
        res -= obs;

        //std::cout << "in test,   after obs,     res = " << res << "\n";
        //std::cout << "in test,    depth = " << p_c_3d[2] << "\n";

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }


        if (d_res_d_xi) {
            //@Simon: Jp = d res / d pose
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
            //@Simon: Jl = d res / d landmark
            //*d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>();
            *d_res_d_l = d_res_d_p * T_c_w_mat;
        }

        //assert(projection_valid);
        return projection_valid;
    }


    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::linearize_point_refine(
            const Vec2& obs, const Vec3& lm_p_w, const SE3& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res,
            MatRP* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l) {

        Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Vec4 p_c_3d;

        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();
        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }
        res -= obs;

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }

        if (d_res_d_xi) {
            Mat<Scalar, 4, POSE_SIZE> d_point_d_xi;
            d_point_d_xi.template topLeftCorner<3, 3>() = Mat3::Identity();
            d_point_d_xi.template topRightCorner<3, 3>() =
                    -SO3::hat(p_c_3d.template head<3>());
            d_point_d_xi.row(3).setZero();
            *d_res_d_xi = d_res_d_p * d_point_d_xi;  // @Simon: double check, but should stay similar for affine BA
            // @Simon: dres/dxi is perhaps [dF/dtj dF/dphi], id est [Jac_pose_translation Jac_pose_rotation]
        }

        if (d_res_d_l) {
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
            // @Simon: is probably Jac_landmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian(
            const Vec2& obs, const Vec3& lm_p_w, const SE3& T_c_w,
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l) {
        Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]

        //@Simon: try to force 3rd line to be [0 0 0 1]
        T_c_w_mat(2,0) = 0;
        T_c_w_mat(2,1) = 0;
        T_c_w_mat(2,2) = 0;
        T_c_w_mat(2,3) = 1;

        Vec4 p_c_3d = T_c_w_mat * lm_p_w.homogeneous(); // @Simon: P = R X + t

        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) { //@Simon: Check the role of d_res_d_p (unnecessary?)
            //@Simon: d_res_d_p = d_proj_d_p3d
            //@Simon: d_res_d_i = d_proj_d_intrinsics (typically we only consider f for affine camera)
            //@Simon: res = (f * x, f * y)
            projection_valid = intr.project_affine(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_affine(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }
        res -= obs; //@Simon: here we get epsilon = (f*x,f*y) - obs;

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
// @Simon: check if d_res_d_xi is necessary
        if (d_res_d_xi) {
            Mat<Scalar, 4, POSE_SIZE> d_point_d_xi;
            d_point_d_xi.template topLeftCorner<3, 3>() = Mat3::Identity();
            d_point_d_xi.template topRightCorner<3, 3>() =
                    -SO3::hat(p_c_3d.template head<3>());
            d_point_d_xi.row(2).setZero(); //@Simon: not sure: to check. The idea: we mimic an affine camera with 3rd row is 0 0 0 1
            d_point_d_xi.row(3).setZero();
            *d_res_d_xi = d_res_d_p * d_point_d_xi;  // @Simon: double check, but should stay similar for affine BA
            // @Simon: dres/dxi is perhaps [dF/dtj dF/dphi], id est [Jac_pose_translation Jac_pose_rotation]
        }

        if (d_res_d_l) {
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
            // @Simon: is probably Jac_landmark
        }

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_affine_space(
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP_affine_space* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l) {
        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        //@Simon: try to force 3rd line to be [0 0 0 1]
        T_c_w_mat(2,0) = 0;
        T_c_w_mat(2,1) = 0;
        T_c_w_mat(2,2) = 0;
        T_c_w_mat(2,3) = 1;
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;

        //Vec4 p_c_3d = T_c_w_mat * lm_p_w.homogeneous(); // @Simon: P = R X + t
        Vec4 p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) { //@Simon: Check the role of d_res_d_p (unnecessary?)
            //@Simon: d_res_d_p = d_proj_d_p3d
            //@Simon: d_res_d_i = d_proj_d_intrinsics (typically we only consider f for affine camera)
            //@Simon: res = (f * x, f * y)
            projection_valid = intr.project_affine_matrix_space(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_affine_matrix_space(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }
        res -= obs; //@Simon: here we get epsilon = (f*x,f*y) - obs;

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
// @Simon: check if d_res_d_xi is necessary
        //@Simon: d_res_d_xi is of size (2,8)
        if (d_res_d_xi) {
            Mat48 d_tmp;
            d_tmp.setZero();
            d_tmp(0,0) = lm_p_w.homogeneous()(0);
            d_tmp(0,1) = lm_p_w.homogeneous()(1);
            d_tmp(0,2) = lm_p_w.homogeneous()(2);
            d_tmp(0,3) = 1;

            d_tmp(1,4) = lm_p_w.homogeneous()(0);
            d_tmp(1,5) = lm_p_w.homogeneous()(1);
            d_tmp(1,6) = lm_p_w.homogeneous()(2);
            d_tmp(1,7) = 1;
            *d_res_d_xi = d_res_d_p * d_tmp;
//            d_res_d_xi(0,0) = lm_p_w.homogeneous()(0);
//            d_res_d_xi(0,1) = lm_p_w.homogeneous()(1);
//            d_res_d_xi(0,2) = lm_p_w.homogeneous()(2);
//            d_res_d_xi(0,3) = 1;
//            d_res_d_xi(1,4) = lm_p_w.homogeneous()(0);
//            d_res_d_xi(1,5) = lm_p_w.homogeneous()(1);
//            d_res_d_xi(1,6) = lm_p_w.homogeneous()(2);
//            d_res_d_xi(1,7) = 1;

        }

        if (d_res_d_l) {
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
            // @Simon: is probably Jac_landmark
        }

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE(int alpha,
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR_pOSE& res, MatRP_pOSE* d_res_d_xi, MatRI_pOSE* d_res_d_i, MatRL_pOSE* d_res_d_l) {
        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt((100 - alpha)/100.0) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt((100 - alpha)/100.0) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(alpha/100.0) * T_c_w.row(0);
        T_c_w_mat.row(3) = sqrt(alpha/100.0) * T_c_w.row(1);

        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();
        Mat4 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_pOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_pOSE(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }


        res(2) -= sqrt(alpha/100.0) * obs(0);
        res(3) -= sqrt(alpha/100.0) * obs(1);

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
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

            d_p_d_xi.row(0) *= sqrt((100 - alpha)/100.0);

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);;
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);

            d_p_d_xi.row(1) *= sqrt((100 - alpha)/100.0);

            d_p_d_xi(2,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,3) = 1;

            d_p_d_xi.row(2) *= sqrt(alpha/100.0);

            d_p_d_xi(3,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(3,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(3,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(3,7) = 1;

            d_p_d_xi.row(3) *= sqrt(alpha/100.0);

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_RpOSE(double alpha,
                                                                          const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
                                                                          const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                          VecR_RpOSE& res, MatRP_RpOSE* d_res_d_xi, MatRI_RpOSE* d_res_d_i, MatRL_RpOSE* d_res_d_l) {
        Mat34 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt(1.0 - alpha) * (-obs(1)  * T_c_w.row(0) + obs(0) * T_c_w.row(1)) / sqrt(obs(0) * obs(0) + obs(1)*obs(1));
        T_c_w_mat.row(1) = sqrt(alpha) * T_c_w.row(0);
        T_c_w_mat.row(2) = sqrt(alpha) * T_c_w.row(1);

        Vec3 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat3 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_RpOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_RpOSE(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }


        res(1) -= sqrt(alpha) * obs(0);
        res(2) -= sqrt(alpha) * obs(1);

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat38 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = -obs(1) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = -obs(1) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = -obs(1) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = -obs(1);
            d_p_d_xi(0,4) = obs(0) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,5) = obs(0) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,6) = obs(0) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,7) = obs(0);

            d_p_d_xi.row(0) *= sqrt(1.0-alpha) / sqrt(obs(0)*obs(0) + obs(1)*obs(1));


            d_p_d_xi(1,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,3) = 1;
            d_p_d_xi.row(1) *= sqrt(alpha);

            d_p_d_xi(2,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,7) = 1;
            d_p_d_xi.row(2) *= sqrt(alpha);

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<3, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_jacobian_metric_upgrade(const Mat3& PH, const Mat3& PHHP, const Mat34& space_matrix_intrinsics, const Scalar& alpha, const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                           VecR_metric& res, MatRH* d_res_d_H, MatR_alpha* d_res_d_alpha2, MatR_alpha_v2* d_res_d_alpha){
        if (d_res_d_H) {
            Mat9 K;
            K.setZero();
            K(0,0) = 1;
            K(1,3) = 1;
            K(2,6) = 1;
            K(3,1) = 1;
            K(4,4) = 1;
            K(5,7) = 1;
            K(6,2) = 1;
            K(7,5) = 1;
            K(8,8) = 1;

            Mat9_12 PH_kron_P;
            PH_kron_P.template block<3,4>(0,0) = PH(0,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(0,4) = PH(0,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(0,8) = PH(0,2) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,0) = PH(1,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,4) = PH(1,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(3,8) = PH(1,2) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,0) = PH(2,0) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,4) = PH(2,1) * space_matrix_intrinsics;
            PH_kron_P.template block<3,4>(6,8) = PH(2,2) * space_matrix_intrinsics;

            Mat9_12 tmp = alpha * (PH_kron_P + K * PH_kron_P);
            Mat93 dres_dH_tmp;
            dres_dH_tmp.setZero();
            dres_dH_tmp.col(0) = tmp.col(3);
            dres_dH_tmp.col(1) = tmp.col(7);
            dres_dH_tmp.col(2) = tmp.col(11);
            *d_res_d_H = dres_dH_tmp;
        }

        if (d_res_d_alpha) {
            Vec9 phhp_vec;
            phhp_vec.setZero();
            phhp_vec.template segment<3>(0) = PHHP.col(0);
            phhp_vec.template segment<3>(3) = PHHP.col(1);
            phhp_vec.template segment<3>(6) = PHHP.col(2);
            //phhp_vec.normalized();
            *d_res_d_alpha = phhp_vec;// * phhp_vec.transpose();
            phhp_vec.normalize();
            *d_res_d_alpha2 = phhp_vec * phhp_vec.transpose();
            //@SImon: Try 1:
            //*d_res_d_alpha = PHHP * PHHP.transpose();

            //*d_res_d_alpha = (PHHP/PHHP.norm()) * (PHHP/PHHP.norm()).transpose(); //@Simon: try this normalization
        }

        bool projection_valid = intr.project_metric(PHHP, alpha, res);

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_jacobian_metric_upgrade_v3(const Vec3& c, const Mat3& PH, const Mat3& PHHP, const Mat34& space_matrix_intrinsics, const Scalar& alpha, const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                           VecR_metric& res, MatRH* d_res_d_H, MatR_alpha* d_res_d_alpha2, MatR_alpha_v2* d_res_d_alpha){
        if (d_res_d_H) {

            Mat93 dres_dH_tmp;
            dres_dH_tmp.setZero();

            dres_dH_tmp(0,0) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(0,1) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(0,2) = 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(0,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(1,0) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,0) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3)* c(0);
            dres_dH_tmp(1,1) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,1) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(1,2) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,2) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(2,0) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,0) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(2,1) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,1) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(2,2) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,2) + space_matrix_intrinsics(0,3)* space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(0,3) *  space_matrix_intrinsics(2,3) * c(2);

            dres_dH_tmp(3,0) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,0) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(3,1) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,1) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(3,2) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(1,2) + space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(4,0) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(0);
            dres_dH_tmp(4,1) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(4,2) = 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(1,3) * space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(5,0) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,0) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(5,1) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,1) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(5,2) = space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,2) + space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(1,3) *  space_matrix_intrinsics(2,3) * c(2);

            dres_dH_tmp(6,0) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,0) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(0);
            dres_dH_tmp(6,1) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,1) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(1);
            dres_dH_tmp(6,2) = space_matrix_intrinsics(0,3) * space_matrix_intrinsics(2,2) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(0,3) * c(2);

            dres_dH_tmp(7,0) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,0) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(0);
            dres_dH_tmp(7,1) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,1) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(1);
            dres_dH_tmp(7,2) = space_matrix_intrinsics(1,3) * space_matrix_intrinsics(2,2) + space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(1,3) * c(2);

            dres_dH_tmp(8,0) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,0) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(0);
            dres_dH_tmp(8,1) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,1) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(1);
            dres_dH_tmp(8,2) = 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,2) + 2 * space_matrix_intrinsics(2,3) * space_matrix_intrinsics(2,3) * c(2);

            *d_res_d_H = alpha * dres_dH_tmp;

        }


        if (d_res_d_alpha) {
            Vec9 phhp_vec;
            phhp_vec.setZero();
            phhp_vec.template segment<3>(0) = PHHP.col(0);
            phhp_vec.template segment<3>(3) = PHHP.col(1);
            phhp_vec.template segment<3>(6) = PHHP.col(2);
            //phhp_vec.normalized();
            *d_res_d_alpha = phhp_vec;// * phhp_vec.transpose();
            //phhp_vec.normalize();
            *d_res_d_alpha2 = phhp_vec * phhp_vec.transpose()/Scalar(phhp_vec.transpose() * phhp_vec);
            //@SImon: Try 1:
            //*d_res_d_alpha = PHHP * PHHP.transpose();

            //*d_res_d_alpha = (PHHP/PHHP.norm()) * (PHHP/PHHP.norm()).transpose(); //@Simon: try this normalization
        }

        bool projection_valid = intr.project_metric(PHHP, alpha, res);

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_RpOSE_refinement(double alpha,
                                                                           const Vec2& obs, const Vec2& rpose_eq, const Vec3& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
                                                                           const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                           VecR_RpOSE& res, MatRP_RpOSE* d_res_d_xi, MatRI_RpOSE* d_res_d_i, MatRL_RpOSE* d_res_d_l) {
        Mat34 T_c_w_mat;
        Scalar norm_eq = sqrt(rpose_eq(0)*rpose_eq(0) + rpose_eq(1)*rpose_eq(1));
        Scalar norm_eq_3 = pow(norm_eq,3);
        Scalar prod_obs_eq = - obs(1) * rpose_eq(0) + obs(0) * rpose_eq(1);
        Vec2 Jac;
        Jac(0) = - obs(1) / norm_eq - prod_obs_eq / norm_eq_3 * rpose_eq(0);
        Jac(1) = obs(0) / norm_eq - prod_obs_eq / norm_eq_3 * rpose_eq(1);
        T_c_w_mat.row(0) = sqrt(1.0-alpha) *(Jac(0) * T_c_w.row(0) + Jac(1) * T_c_w.row(1));
        T_c_w_mat.row(1) = sqrt(alpha) * T_c_w.row(0);
        T_c_w_mat.row(2) = sqrt(alpha) * T_c_w.row(1);


        Vec3 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

        Mat3 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_RpOSE_refinement(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_RpOSE_refinement(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }

        res(0) -= sqrt(1.0-alpha) * (Jac(0) * rpose_eq(0) + Jac(1) * rpose_eq(1) - (-obs(1) / norm_eq * rpose_eq(0) + obs(0) / norm_eq * rpose_eq(1)));
        res(1) -= sqrt(alpha) * rpose_eq(0);
        res(2) -= sqrt(alpha) * rpose_eq(1);

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat38 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = Jac(0) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = Jac(0) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = Jac(0) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = Jac(0);
            d_p_d_xi(0,4) = Jac(1) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,5) = Jac(1) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,6) = Jac(1) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,7) = Jac(1);

            d_p_d_xi.row(0) *= sqrt(1.0-alpha);


            d_p_d_xi(1,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,3) = 1;
            d_p_d_xi.row(1) *= sqrt(alpha);

            d_p_d_xi(2,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,7) = 1;
            d_p_d_xi.row(2) *= sqrt(alpha);

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<3, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_RpOSE_ML(const Vec2& obs, const Vec2& rpose_eq, const Vec3& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
                                                                                      const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                                      VecR_RpOSE_ML& res, MatRP_RpOSE_ML* d_res_d_xi, MatRI_RpOSE_ML* d_res_d_i, MatRL_RpOSE_ML* d_res_d_l) {
        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat14 T_c_w_mat;
        Scalar norm_eq = sqrt(rpose_eq(0)*rpose_eq(0) + rpose_eq(1)*rpose_eq(1));
        Scalar norm_eq_3 = pow(norm_eq,3);
        Scalar prod_obs_eq = - obs(1) * rpose_eq(0) + obs(0) * rpose_eq(1);
        Vec2 Jac;
        Jac(0) = - obs(1) / norm_eq - prod_obs_eq / norm_eq_3 * rpose_eq(0);
        Jac(1) = obs(0) / norm_eq - prod_obs_eq / norm_eq_3 * rpose_eq(1);

        T_c_w_mat.row(0) = (Jac(0) * T_c_w.row(0) + Jac(1) * T_c_w.row(1));


        Scalar p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();


        res(0) = p_c_3d - (Jac(0) * rpose_eq(0) + Jac(1) * rpose_eq(1) - (-obs(1) / norm_eq * rpose_eq(0) + obs(0) / norm_eq * rpose_eq(1)));


        // valid &= res.array().isFinite().all();
        bool projection_valid;
        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat18 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = Jac(0) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = Jac(0) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = Jac(0) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = Jac(0);
            d_p_d_xi(0,4) = Jac(1) * lm_p_w.homogeneous()(0);
            d_p_d_xi(0,5) = Jac(1) * lm_p_w.homogeneous()(1);
            d_p_d_xi(0,6) = Jac(1) * lm_p_w.homogeneous()(2);
            d_p_d_xi(0,7) = Jac(1);



            //*d_res_d_xi = d_res_d_p * d_p_d_xi;
            *d_res_d_xi = d_p_d_xi;


        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = T_c_w_mat.template topLeftCorner<1, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }



    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_expOSE(int alpha,
                                                                          const Vec2& obs, const Vec3& y_tilde,
                                                                          const Vec3& lm_p_w, const Vec3& lm_p_w_equilibrium,
                                                                          const Mat34& T_c_w, const Mat34& T_c_w_equilibrium,
                                                                          const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                          VecR_expOSE& res, MatRP_expOSE* d_res_d_xi, MatRI_expOSE* d_res_d_i, MatRL_expOSE* d_res_d_l) {
        //Scalar l_exp_equilibrium = exp(-Scalar((obs(0) * T_c_w_equilibrium.row(0) + obs(1) * T_c_w_equilibrium.row(1) + T_c_w_equilibrium.row(2)) * lm_p_w_equilibrium.homogeneous())/sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));
        //Scalar l_exp_equilibrium = exp(-Scalar((obs(0) * T_c_w_equilibrium.row(0) + obs(1) * T_c_w_equilibrium.row(1) + T_c_w_equilibrium.row(2)) * lm_p_w_equilibrium.homogeneous())/sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));
        Scalar l_exp_equilibrium = exp(-Scalar(obs(0)/100.0 * y_tilde(0) + obs(1)/100.0 * y_tilde(1) + y_tilde(2))/sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1));

        Mat34 T_c_w_mat;
        T_c_w_mat.row(0) = sqrt((100 - alpha)/100.0) * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = sqrt((100 - alpha)/100.0) * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = sqrt(l_exp_equilibrium/2.0) * sqrt(alpha/100.0) * (obs(0)/100.0 * T_c_w.row(0) + obs(1)/100.0 * T_c_w.row(1) + T_c_w.row(2)) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);

        Vec3 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();
        Mat3 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_expOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_expOSE(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }


        //res(2) -= sqrt(alpha/100.0) * sqrt(l_exp_equilibrium/2.0) * (1 + Scalar(1/ sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1))* obs(0) * y_tilde(0) + Scalar(1/ sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1)) * obs(1) * y_tilde(1) + Scalar(1/ sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1))* y_tilde(2)); // / sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));
        res(2) -= sqrt(alpha/100.0) * sqrt(l_exp_equilibrium/2.0) * (1 + Scalar(obs(0)/100.0 * y_tilde(0)+ obs(1)/100.0 * y_tilde(1) + y_tilde(2)) /sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1)); // / sqrt(obs(0) * obs(0) + obs(1) * obs(1) + 1));



        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat3_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = 1;
            d_p_d_xi(0,8) = -lm_p_w.homogeneous()(0) * obs(0);
            d_p_d_xi(0,9) = -lm_p_w.homogeneous()(1) * obs(0);
            d_p_d_xi(0,10) = -lm_p_w.homogeneous()(2) * obs(0);
            d_p_d_xi(0,11) = - obs(0);
            d_p_d_xi.row(0) *= sqrt((100-alpha)/100.0);

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);
            d_p_d_xi.row(1) *= sqrt((100-alpha)/100.0);

            d_p_d_xi(2,0) = obs(0)/100.0 * lm_p_w.homogeneous()(0) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,1) = obs(0)/100.0 * lm_p_w.homogeneous()(1) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,2) = obs(0)/100.0 * lm_p_w.homogeneous()(2) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,3) = obs(0)/100.0 / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);

            d_p_d_xi(2,4) = obs(1)/100.0 * lm_p_w.homogeneous()(0) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,5) = obs(1)/100.0 * lm_p_w.homogeneous()(1) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,6) = obs(1)/100.0 * lm_p_w.homogeneous()(2) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,7) = obs(1)/100.0 / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);

            d_p_d_xi(2,8) = lm_p_w.homogeneous()(0) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,9) = lm_p_w.homogeneous()(1) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,10) = lm_p_w.homogeneous()(2) / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);
            d_p_d_xi(2,11) = 1 / sqrt(obs(0)/100.0 * obs(0)/100.0 + obs(1)/100.0 * obs(1)/100.0 + 1);

            d_p_d_xi.row(2) *= sqrt(alpha/100.0) * sqrt(l_exp_equilibrium/2.0);

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }
        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<3, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }


    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE_rOSE(int alpha,
                                                                          const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
                                                                          const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                          VecR_pOSE_rOSE& res, MatRP_pOSE_rOSE* d_res_d_xi, MatRI_pOSE_rOSE* d_res_d_i, MatRL_pOSE_rOSE* d_res_d_l) {
        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat54 T_c_w_mat;
        T_c_w_mat.row(0) = (100 - alpha)/100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha)/100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(0);
        T_c_w_mat.row(3) = alpha/100.0 * T_c_w.row(1);
        T_c_w_mat.row(4) = T_c_w.row(2) * 10;

        Vec5 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();
        Mat5 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_pOSE_rOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
        } else {
            projection_valid = intr.project_pOSE_rOSE(p_c_3d, res, nullptr, nullptr);
        }


        res(2) -= alpha/100.0 * obs(0);
        res(3) -= alpha/100.0 * obs(1);
        res(4) -= 1.0 * 10;

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }

        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat5_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(0,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(0,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(0,3) = 1;
            d_p_d_xi(0,8) = -lm_p_w.homogeneous()(0) * obs(0);
            d_p_d_xi(0,9) = -lm_p_w.homogeneous()(1) * obs(0);
            d_p_d_xi(0,10) = -lm_p_w.homogeneous()(2) * obs(0);
            d_p_d_xi(0,11) = - obs(0);

            d_p_d_xi.row(0) *= (100 - alpha)/100.0;

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);;
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);

            d_p_d_xi.row(1) *= (100 - alpha)/100.0;

            d_p_d_xi(2,0) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,1) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,2) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,3) = 1;

            d_p_d_xi.row(2) *= alpha/100.0;

            d_p_d_xi(3,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(3,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(3,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(3,7) = 1;

            d_p_d_xi.row(3) *= alpha/100.0;

            d_p_d_xi(4,8) = lm_p_w.homogeneous()(0);
            d_p_d_xi(4,9) = lm_p_w.homogeneous()(1);
            d_p_d_xi(4,10) = lm_p_w.homogeneous()(2);
            d_p_d_xi(4,11) = 1;

            d_p_d_xi.row(4) *= 10;

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<5, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }


    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_rOSE(int alpha,
                                                                          const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
                                                                          const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                          VecR_rOSE& res, MatRP_rOSE* d_res_d_xi, MatRI_rOSE* d_res_d_i, MatRL_rOSE* d_res_d_l) {
        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = (100 - alpha)/100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha)/100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(2);
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;
        //T_c_w_mat.row(3) = alpha/100.0 * T_c_w.row(1);

        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w.homogeneous();
        Mat34 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_rOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_rOSE(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }


        res(2) -= alpha/100.0;
        //res(3) -= alpha/100.0 * obs(1);

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
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

            d_p_d_xi.row(0) *= (100 - alpha)/100.0;

            d_p_d_xi(1,4) = lm_p_w.homogeneous()(0);
            d_p_d_xi(1,5) = lm_p_w.homogeneous()(1);
            d_p_d_xi(1,6) = lm_p_w.homogeneous()(2);
            d_p_d_xi(1,7) = 1;
            d_p_d_xi(1,8) = -lm_p_w.homogeneous()(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w.homogeneous()(1) * obs(1);;
            d_p_d_xi(1,10) = -lm_p_w.homogeneous()(2) * obs(1);
            d_p_d_xi(1,11) = -obs(1);

            d_p_d_xi.row(1) *= (100 - alpha)/100.0;

            d_p_d_xi(2,8) = lm_p_w.homogeneous()(0);
            d_p_d_xi(2,9) = lm_p_w.homogeneous()(1);
            d_p_d_xi(2,10) = lm_p_w.homogeneous()(2);
            d_p_d_xi(2,11) = 1;

            d_p_d_xi.row(2) *= alpha/100.0;


            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_pOSE_homogeneous(int alpha,
                                                                          const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
                                                                          const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
                                                                          VecR_pOSE& res, MatRP_pOSE* d_res_d_xi, MatRI_pOSE* d_res_d_i, MatRL_pOSE_homogeneous* d_res_d_l) {
        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = (100 - alpha)/100.0 * (T_c_w.row(0) - T_c_w.row(2) * obs(0));
        T_c_w_mat.row(1) = (100 - alpha)/100.0 * (T_c_w.row(1) - T_c_w.row(2) * obs(1));
        T_c_w_mat.row(2) = alpha/100.0 * T_c_w.row(0);
        T_c_w_mat.row(3) = alpha/100.0 * T_c_w.row(1);

        Vec4 p_c_3d;
        p_c_3d = T_c_w_mat * lm_p_w;
        Mat4 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) {
            projection_valid = intr.project_pOSE(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: originally, it is intr.proj
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i); //@Simon: here res is changed... ISSUE
            //projection_valid = intr.project(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_pOSE(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_affine_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project(p_c_3d, res, nullptr, nullptr);
        }


        res(2) -= obs(0);
        res(3) -= obs(1);

        // valid &= res.array().isFinite().all();

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
        //@Simon: d_res_d_xi is of size (2,8)
        //@Simon: d_res_d_xi is d res / d camera_parameters
        //@Simon: change into pointers as in the initial function
        if (d_res_d_xi) { //@Simon TODO: FIX BY WRITING d_res_d_xi = d_res_d_p * d_p_d_xi
            //@Simon: d res / d pose = Jp
            Mat4_12 d_p_d_xi;
            d_p_d_xi.setZero();
            d_p_d_xi(0,0) = lm_p_w(0);
            d_p_d_xi(0,1) = lm_p_w(1);
            d_p_d_xi(0,2) = lm_p_w(2);
            d_p_d_xi(0,3) = lm_p_w(3);
            d_p_d_xi(0,8) = -lm_p_w(0) * obs(0);
            d_p_d_xi(0,9) = -lm_p_w(1) * obs(0);
            d_p_d_xi(0,10) = -lm_p_w(2) * obs(0);
            d_p_d_xi(0,11) = - lm_p_w(3)*obs(0);

            d_p_d_xi.row(0) *= (100 - alpha)/100.0;

            d_p_d_xi(1,4) = lm_p_w(0);
            d_p_d_xi(1,5) = lm_p_w(1);
            d_p_d_xi(1,6) = lm_p_w(2);
            d_p_d_xi(1,7) = lm_p_w(3);
            d_p_d_xi(1,8) = -lm_p_w(0) * obs(1);
            d_p_d_xi(1,9) = -lm_p_w(1) * obs(1);;
            d_p_d_xi(1,10) = -lm_p_w(2) * obs(1);
            d_p_d_xi(1,11) = -lm_p_w(3)*obs(1);

            d_p_d_xi.row(1) *= (100 - alpha)/100.0;

            d_p_d_xi(2,0) = lm_p_w(0);
            d_p_d_xi(2,1) = lm_p_w(1);
            d_p_d_xi(2,2) = lm_p_w(2);
            d_p_d_xi(2,3) = lm_p_w(3);

            d_p_d_xi.row(2) *= alpha/100.0;

            d_p_d_xi(3,4) = lm_p_w(0);
            d_p_d_xi(3,5) = lm_p_w(1);
            d_p_d_xi(3,6) = lm_p_w(2);
            d_p_d_xi(3,7) = lm_p_w(3);

            d_p_d_xi.row(3) *= alpha/100.0;

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            //@Simon: d res/ d landmark = Jl
            *d_res_d_l = d_res_d_p * T_c_w_mat; // = dc'/dc dc/dx R = dF / dlandmark
        }
        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_projective_space(
            const Vec2& obs, const Vec3& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP_projective_space* d_res_d_xi, MatRI* d_res_d_i, MatRL* d_res_d_l) {
        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        T_c_w_mat.row(2) = T_c_w.row(2);
        //T_c_w_mat(2,3) = 1;
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;

        Vec4 p_c_3d = T_c_w_mat * lm_p_w.homogeneous() / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) { //@Simon: Check the role of d_res_d_p (unnecessary?)
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i);
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, nullptr, nullptr);
        }
        res -= obs;

        // valid &= res.array().isFinite().all();

        Scalar lm_p_w_0 = lm_p_w(0) / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
        Scalar lm_p_w_1 = lm_p_w(1) / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
        Scalar lm_p_w_2 = lm_p_w(2) / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
        Scalar lm_p_w_3 = 1  / sqrt(pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

        if (!ignore_validity_check && !projection_valid) {
            return false;
        }
// @Simon: check if d_res_d_xi is necessary
        //@Simon: d_res_d_xi is of size (2,8)
        if (d_res_d_xi) {

//@Simon: TRY2
            Mat4_12 d_p_d_xi;
            d_p_d_xi.setZero();
            //d_p_d_xi.row(0).setZero();
            //d_p_d_xi.row(1).setZero();
            //d_p_d_xi.row(2).setZero();
            //d_p_d_xi.row(3).setZero();
            d_p_d_xi(0,0) = lm_p_w_0;
            d_p_d_xi(0,1) = lm_p_w_1;
            d_p_d_xi(0,2) = lm_p_w_2;
            d_p_d_xi(0,3) = lm_p_w_3;

            d_p_d_xi(1,4) = lm_p_w_0;
            d_p_d_xi(1,5) = lm_p_w_1;
            d_p_d_xi(1,6) = lm_p_w_2;
            d_p_d_xi(1,7) = lm_p_w_3;

            d_p_d_xi(2,8) = lm_p_w_0;
            d_p_d_xi(2,9) = lm_p_w_1;
            d_p_d_xi(2,10) = lm_p_w_2;
            d_p_d_xi(2,11) = lm_p_w_3;

            *d_res_d_xi = d_res_d_p * d_p_d_xi;

        }

        if (d_res_d_l) {
            Mat43 d_p_d_l;
            d_p_d_l.setZero();

            d_p_d_l(0,0) = T_c_w_mat(0,0) * (lm_p_w_3 - lm_p_w(0) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(0,0) -= T_c_w_mat(0,1) * lm_p_w(1) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,0) -= T_c_w_mat(0,2) * lm_p_w(2) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,0) -= T_c_w_mat(0,3) *  lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(0,1) = T_c_w_mat(0,1) * (lm_p_w_3 - lm_p_w(1) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(0,1) -= T_c_w_mat(0,0) * lm_p_w(0) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,1) -= T_c_w_mat(0,2) * lm_p_w(2) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,1) -= T_c_w_mat(0,3)  * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(0,2) = T_c_w_mat(0,2) * (lm_p_w_3 - lm_p_w(2) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(0,2) -= T_c_w_mat(0,0) * lm_p_w(0) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,2) -= T_c_w_mat(0,1) * lm_p_w(1) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(0,2) -= T_c_w_mat(0,3) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(1,0) = T_c_w_mat(1,0) * (lm_p_w_3 - lm_p_w(0) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(1,0) -= T_c_w_mat(1,1) * lm_p_w(1) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,0) -= T_c_w_mat(1,2) * lm_p_w(2) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,0) -= T_c_w_mat(1,3) *  lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(1,1) = T_c_w_mat(1,1) * (lm_p_w_3 - lm_p_w(1) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(1,1) -= T_c_w_mat(1,0) * lm_p_w(0) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,1) -= T_c_w_mat(1,2) * lm_p_w(2) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,1) -= T_c_w_mat(1,3)  * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(1,2) = T_c_w_mat(1,2) * (lm_p_w_3 - lm_p_w(2) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(1,2) -= T_c_w_mat(1,0) * lm_p_w(0) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,2) -= T_c_w_mat(1,1) * lm_p_w(1) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(1,2) -= T_c_w_mat(1,3) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(2,0) = T_c_w_mat(2,0) * (lm_p_w_3 - lm_p_w(0) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(2,0) -= T_c_w_mat(2,1) * lm_p_w(1) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,0) -= T_c_w_mat(2,2) * lm_p_w(2) * lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,0) -= T_c_w_mat(2,3) *  lm_p_w_0 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(2,1) = T_c_w_mat(2,1) * (lm_p_w_3 - lm_p_w(1) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(2,1) -= T_c_w_mat(2,0) * lm_p_w(0) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,1) -= T_c_w_mat(2,2) * lm_p_w(2) * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,1) -= T_c_w_mat(2,3)  * lm_p_w_1 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            d_p_d_l(2,2) = T_c_w_mat(2,2) * (lm_p_w_3 - lm_p_w(2) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1)) ;
            d_p_d_l(2,2) -= T_c_w_mat(2,0) * lm_p_w(0) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,2) -= T_c_w_mat(2,1) * lm_p_w(1) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);
            d_p_d_l(2,2) -= T_c_w_mat(2,3) * lm_p_w_2 / (pow(lm_p_w(0),2) + pow(lm_p_w(1),2) + pow(lm_p_w(2),2) + 1);

            *d_res_d_l = d_res_d_p * d_p_d_l;

            //*d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>(); // = dc'/dc dc/dx R = dF / dlandmark
            // @Simon: is probably Jac_landmark
        }

        return projection_valid;
    }

    template <typename Scalar>
    bool BalBundleAdjustmentHelper<Scalar>::update_landmark_jacobian_projective_space_homogeneous(
            const Vec2& obs, const Vec4& lm_p_w, const Mat34& T_c_w, //@Simon: or use a submatrix of size Mat24
            const basalt::BalCamera<Scalar>& intr, const bool ignore_validity_check,
            VecR& res, MatRP_projective_space* d_res_d_xi, MatRI* d_res_d_i, MatRL_projective_space_homogeneous * d_res_d_l) {
        //Mat4 T_c_w_mat = T_c_w.matrix(); // @Simon: should be [R t]
        Mat4 T_c_w_mat;
        T_c_w_mat.row(0) = T_c_w.row(0);
        T_c_w_mat.row(1) = T_c_w.row(1);
        T_c_w_mat.row(2) = T_c_w.row(2);
        //T_c_w_mat(2,3) = 1;
        T_c_w_mat(3,0) = 0;
        T_c_w_mat(3,1) = 0;
        T_c_w_mat(3,2) = 0;
        T_c_w_mat(3,3) = 1;

        Vec4 p_c_3d = T_c_w_mat * lm_p_w;

        Mat24 d_res_d_p;
        bool projection_valid;
        if (d_res_d_xi || d_res_d_i || d_res_d_l) { //@Simon: Check the role of d_res_d_p (unnecessary?)
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, &d_res_d_p, d_res_d_i);
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, &d_res_d_p, d_res_d_i);
        } else {
            projection_valid = intr.project_projective_refinement_matrix_space_without_distortion(p_c_3d, res, nullptr, nullptr);
            //projection_valid = intr.project_projective_refinement_matrix_space(p_c_3d, res, nullptr, nullptr);
        }
        res -= obs;


        if (!ignore_validity_check && !projection_valid) {
            return false;
        }

        if (d_res_d_xi) {

            Mat4_12 d_p_d_xi;
            d_p_d_xi.setZero();
            //d_p_d_xi.row(0).setZero();
            //d_p_d_xi.row(1).setZero();
            //d_p_d_xi.row(2).setZero();
            //d_p_d_xi.row(3).setZero();
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
            *d_res_d_l = d_res_d_p * T_c_w_mat;
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

}  // namespace rootba
