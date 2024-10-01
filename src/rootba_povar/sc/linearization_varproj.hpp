//
// Created by Simon on 03/07/2023.
//

#pragma once

#include <mutex>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "rootba_povar/bal/bal_problem.hpp"
#include "rootba_povar/sc/landmark_block.hpp"
#include "rootba_povar/util/assert.hpp"
#include "rootba_povar/util/cast.hpp"
#include "rootba_povar/util/format.hpp"

namespace rootba_povar {

    template<typename Scalar_, int POSE_SIZE_>
    class LinearizationVarProj {
    public:
        using Scalar = Scalar_;
        static constexpr int POSE_SIZE = POSE_SIZE_;
        static constexpr int POSE_SIZE_hom = POSE_SIZE_ + 4;

        using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
        using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        using Mat36 = Eigen::Matrix<Scalar, 3, 6>;


        using Options = typename LandmarkBlockSC<Scalar, POSE_SIZE>::Options;

        LinearizationVarProj(BalProblem <Scalar> &bal_problem, const Options &options,
                             const bool enable_reduction_alg = true)
                : options_(options),
                  bal_problem_(bal_problem),
                  num_cameras_(bal_problem_.cameras().size()),
                  pose_mutex_(num_cameras_) {
            const size_t num_landmarks = bal_problem_.landmarks().size();
            landmark_blocks_.resize(num_landmarks);

            auto body = [&](const tbb::blocked_range<size_t> &range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].allocate_landmark(bal_problem_.landmarks()[r],
                                                          options_);
                }
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            tbb::parallel_for(range, body);

        };

        bool linearize_problem_pOSE(Scalar alpha) {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t> &range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_pOSE(bal_problem_.cameras(), alpha);
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_projective_space_homogeneous() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t> &range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_projective_space_homogeneous(bal_problem_.cameras());
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_nullspace() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t> &range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_nullspace(bal_problem_.cameras());
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };


        bool linearize_problem_refine() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t> &range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_refine(bal_problem_.cameras());
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        size_t num_cols_reduced() const { return num_cameras_ * POSE_SIZE; }


        Scalar back_substitute_joint(const VecX &pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));

            auto body = [&](const tbb::blocked_range<size_t> &range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].back_substitute_joint(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }


        Scalar back_substitute_poBA(const VecX &pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * POSE_SIZE));

            auto body = [&](const tbb::blocked_range<size_t> &range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].back_substitute_poBA(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        VecX get_Jp_diag2_pOSE() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>> &
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t> &range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto &lb = landmark_blocks[r];
                        lb.add_Jp_diag2_pOSE(res);
                    }
                }

                Reductor(Reductor &a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor &b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>> &landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ *POSE_SIZE, landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }


        VecX get_Jp_diag2_projective_space() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>> &
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t> &range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto &lb = landmark_blocks[r];
                        lb.add_Jp_diag2_projective_space(res);
                    }
                }

                Reductor(Reductor &a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor &b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>> &landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ *POSE_SIZE_hom, landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }


        void scale_Jl_cols_pOSE() {
            auto body = [&](const tbb::blocked_range<size_t> &range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_pOSE();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jl_cols_homogeneous() {
            auto body = [&](const tbb::blocked_range<size_t> &range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_homogeneous();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }


        void scale_Jp_cols_joint(const VecX &jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t> &range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_joint(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_pOSE(const VecX &jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t> &range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_pOSE(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void set_pose_damping(const Scalar lambda) {
            ROOTBA_ASSERT(lambda >= 0);

            pose_damping_diagonal_ = lambda;
        }


        inline bool has_pose_damping() const { return pose_damping_diagonal_ > 0; }

        void set_landmark_damping(const Scalar lambda) {
            ROOTBA_ASSERT(lambda >= 0);

            auto body = [&](const tbb::blocked_range<size_t> &range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].set_landmark_damping(lambda);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void set_landmark_damping_joint(const Scalar lambda) {
            ROOTBA_ASSERT(lambda >= 0);

            auto body = [&](const tbb::blocked_range<size_t> &range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].set_landmark_damping_joint(lambda);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }


        void print_block(const std::string &filename, size_t block_idx) {
            landmark_blocks_[block_idx].printStorage(filename);
        }
    protected:
        Options options_;
        BalProblem<Scalar>& bal_problem_;
        size_t num_cameras_;

        mutable std::vector<std::mutex> H_pp_mutex_;
        mutable std::vector<std::mutex> pose_mutex_;

        std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>> landmark_blocks_;
        std::vector<LandmarkBlockSC<Scalar, POSE_SIZE_hom>> landmark_blocks_hom_;
        //std::vector<LandmarkBlockSC<Scalar, POSE_SIZE_AFFINE>> landmark_blocks_affine_;
        //std::vector<LandmarkBlockSC<Scalar, POSE_SIZE_AFFINE>> landmark_blocks_affine_space_;

        Scalar pose_damping_diagonal_ = 0;


    };  // namespace rootba_povar
}