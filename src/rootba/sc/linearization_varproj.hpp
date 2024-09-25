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

#include "rootba/bal/bal_problem.hpp"
#include "rootba/qr/linearization_utils.hpp"
#include "rootba/sc/landmark_block.hpp"
#include "rootba/util/assert.hpp"
#include "rootba/util/cast.hpp"
#include "rootba/util/format.hpp"

namespace rootba {

    template <typename Scalar_, int POSE_SIZE_>
    class LinearizationVarProj {
    public:
        using Scalar = Scalar_;
        //static constexpr int POSE_SIZE = 11;
        static constexpr int POSE_SIZE = POSE_SIZE_;
        //static constexpr int POSE_SIZE = POSE_SIZE_;
        //static constexpr int POSE_SIZE_AFFINE = POSE_SIZE_;
        static constexpr int POSE_SIZE_AFFINE = POSE_SIZE_ - 4;
        //static constexpr int POSE_SIZE_AFFINE = 11;

        static constexpr int POSE_SIZE_hom = POSE_SIZE_ + 4;

        using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
        using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        using Mat36 = Eigen::Matrix<Scalar, 3, 6>;

        //using Options = typename LandmarkBlockSC<Scalar, POSE_SIZE>::Options;

        using Options = typename LandmarkBlockSC<Scalar, POSE_SIZE>::Options;

        LinearizationVarProj(BalProblem<Scalar>& bal_problem, const Options& options,
                        const bool enable_reduction_alg = true)
                : options_(options),
                  bal_problem_(bal_problem),
                  num_cameras_(bal_problem_.cameras().size()),
                  pose_mutex_(num_cameras_) {
            const size_t num_landmarks = bal_problem_.landmarks().size();
            landmark_blocks_.resize(num_landmarks);
            //landmarks_blocks_affine_.resize(num_landmarks);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].allocate_landmark(bal_problem_.landmarks()[r],
                                                          options_);
                }
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            tbb::parallel_for(range, body);

            // H_pp_mutex_ won't be allocated when this class is inherited,
            // only used in get_Hb
            if (enable_reduction_alg && options_.reduction_alg == 1) {
                // TODO@demmeln: why not just `H_pp_mutex_.resize()`?
                // TODO@tin: Type must be moveable for resizing/
                // https://stackoverflow.com/questions/16465633/how-can-i-use-something-like-stdvectorstdmutex
                std::vector<std::mutex> tmp(num_cameras_ * num_cameras_);
                H_pp_mutex_.swap(tmp);  // avoid move and copy
            }
        };

        // return value `false` indicates numerical failure --> linearization at this
        // state is unusable. Numeric check is only performed for residuals that were
        // considered to be used (valid), which depends on use_valid_projections_only
        // setting.
        bool linearize_problem() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark(bal_problem_.cameras());
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_affine_space() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_affine_space(bal_problem_.cameras());
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_pOSE(int alpha) {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
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

        bool linearize_problem_RpOSE(double alpha) {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_RpOSE(bal_problem_.cameras(), alpha);
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_RpOSE_refinement(double alpha) {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_RpOSE_refinement(bal_problem_.cameras(), alpha);
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_RpOSE_ML() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_RpOSE_ML(bal_problem_.cameras());
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_expOSE(int alpha, bool init) {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_expOSE(bal_problem_.cameras(), alpha, init);
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_pOSE_rOSE(int alpha) {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_pOSE_rOSE(bal_problem_.cameras(), alpha);
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_rOSE(int alpha) {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_rOSE(bal_problem_.cameras(), alpha);
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };


        bool linearize_problem_pOSE_homogeneous(int alpha) {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_pOSE_homogeneous(bal_problem_.cameras(), alpha);
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };


        bool linearize_problem_projective_space() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_projective_space(bal_problem_.cameras());
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

            auto body = [&](const tbb::blocked_range<size_t>& range,
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

        bool linearize_problem_projective_space_homogeneous_RpOSE() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_projective_space_homogeneous_RpOSE(bal_problem_.cameras());
                    numerically_valid &= !landmark_blocks_[r].is_numerical_failure();
                }
                return numerically_valid;
            };

            tbb::blocked_range<size_t> range(0, num_landmarks);
            const bool numerically_valid =
                    tbb::parallel_reduce(range, true, body, std::logical_and<>());

            return numerically_valid;
        };

        bool linearize_problem_projective_space_homogeneous_storage() {
            ROOTBA_ASSERT(bal_problem_.landmarks().size() == landmark_blocks_.size());

            size_t num_landmarks = landmark_blocks_.size();

            auto body = [&](const tbb::blocked_range<size_t>& range,
                            bool numerically_valid) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].linearize_landmark_projective_space_homogeneous_storage(bal_problem_.cameras());
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

            auto body = [&](const tbb::blocked_range<size_t>& range,
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

            auto body = [&](const tbb::blocked_range<size_t>& range,
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

        Scalar back_substitute(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 9));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].back_substitute(pose_inc, l_diff);
                }
                return l_diff;
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar back_substitute_joint(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
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

        Scalar back_substitute_joint_RpOSE(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 13));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].back_substitute_joint_RpOSE(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            return l_diff;
        }

        Scalar back_substitute_poBA(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 15));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
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

        Scalar landmark_closed_form(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * POSE_SIZE));

            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            //std::cout << "IN LINEARIZATION VARPROJ l123 BEFORE REDUCE lm_ptr_[0]->p_w = " << bal_problem_.landmarks().at(0).p_w.norm() << "\n";
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            //std::cout << "IN LINEARIZATION VARPROJ l127 AFTER REDUCE lm_ptr_[0]->p_w = " << bal_problem_.landmarks().at(0).p_w.norm() << "\n";
            return l_diff;
        }

        Scalar landmark_closed_form_affine_space(const VecX& pose_inc) {
            ROOTBA_ASSERT(pose_inc.size() == signed_cast(num_cameras_ * 11));
            std::cout << "IN CLOSED_FORM_AFFINE_SPACE l204 \n";
            auto body = [&](const tbb::blocked_range<size_t>& range, Scalar l_diff) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].landmark_closed_form_affine_space(pose_inc, l_diff, bal_problem_.cameras());
                }
                return l_diff;
            };
            //std::cout << "IN LINEARIZATION VARPROJ l123 BEFORE REDUCE lm_ptr_[0]->p_w = " << bal_problem_.landmarks().at(0).p_w.norm() << "\n";
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            Scalar l_diff =
                    tbb::parallel_reduce(range, Scalar(0), body, std::plus<Scalar>());
            //std::cout << "IN LINEARIZATION VARPROJ l127 AFTER REDUCE lm_ptr_[0]->p_w = " << bal_problem_.landmarks().at(0).p_w.norm() << "\n";
            return l_diff;
        }

        VecX get_Jp_diag2() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * 9, landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }

        VecX get_Jp_diag2_affine_space() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_affine_space(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE_AFFINE , landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }

        VecX get_Jp_diag2_pOSE() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_pOSE(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }

        VecX get_Jp_diag2_RpOSE() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_RpOSE(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }

        VecX get_Jp_diag2_RpOSE_ML() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_RpOSE_ML(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }

        VecX get_Jp_diag2_expOSE() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_expOSE(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }

        VecX get_Jp_diag2_pOSE_rOSE() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_pOSE_rOSE(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }

        VecX get_Jp_diag2_rOSE() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_rOSE(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }


        VecX get_Jp_diag2_pOSE_homogeneous() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_pOSE_homogeneous(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE , landmark_blocks_);

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
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_projective_space(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                //const std::vector<LandmarkBlockSC<Scalar, 15>>& landmark_blocks;
                const std::vector<LandmarkBlockSC<Scalar, 11>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE_hom , landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }

        VecX get_Jp_diag2_projective_space_RpOSE() const {
            struct Reductor {
                Reductor(size_t num_rows,
                         const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE_hom>>&
                         landmark_blocks)
                        : num_rows(num_rows), landmark_blocks(landmark_blocks) {
                    res.setZero(num_rows);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Jp_diag2_projective_space_RpOSE(res);
                    }
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : num_rows(a.num_rows), landmark_blocks(a.landmark_blocks) {
                    res.setZero(num_rows);
                };

                inline void join(const Reductor& b) { res += b.res; }

                size_t num_rows;
                //const std::vector<LandmarkBlockSC<Scalar, 15>>& landmark_blocks;
                const std::vector<LandmarkBlockSC<Scalar, 15>>& landmark_blocks;
                VecX res;
            };

            Reductor r(num_cameras_ * POSE_SIZE_hom , landmark_blocks_hom_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_hom_.size());
            tbb::parallel_reduce(range, r);

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJpTJp_blockdiag

            // Note: ignore damping here

            return r.res;
        }


        void scale_Jl_cols() {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jl_cols_pOSE() {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_pOSE();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jl_cols_RpOSE() {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_RpOSE();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jl_cols_RpOSE_ML() {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_RpOSE_ML();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jl_cols_expOSE() {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_expOSE();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jl_cols_pOSE_homogeneous() {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_pOSE_homogeneous();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }


        void scale_Jl_cols_homogeneous() {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_homogeneous();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jl_cols_homogeneous_RpOSE() {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jl_cols_homogeneous_RpOSE();
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_joint(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_joint(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_affine(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_affine(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_pOSE(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_pOSE(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_RpOSE(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_RpOSE(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_RpOSE_ML(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_RpOSE_ML(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_expOSE(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_expOSE(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_pOSE_rOSE(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_pOSE_rOSE(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_rOSE(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_rOSE(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void scale_Jp_cols_pOSE_homogeneous(const VecX& jacobian_scaling) {
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].scale_Jp_cols_pOSE_homogeneous(jacobian_scaling);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }
        /// compute scale and apply scaling in one go, e.g. for unit tests.
        void compute_Jp_scale_and_scale_Jp_cols() {
            const VecX Jp_scale2 = get_Jp_diag2();
            const VecX Jp_scaling =
                    compute_jacobi_scaling(Jp_scale2, options_.jacobi_scaling_eps);
            scale_Jp_cols(Jp_scaling);
        }

        void set_pose_damping(const Scalar lambda) {
            ROOTBA_ASSERT(lambda >= 0);

            pose_damping_diagonal_ = lambda;
        }


        inline bool has_pose_damping() const { return pose_damping_diagonal_ > 0; }

        void set_landmark_damping(const Scalar lambda) {
            ROOTBA_ASSERT(lambda >= 0);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].set_landmark_damping(lambda);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void set_landmark_damping_joint(const Scalar lambda) {
            ROOTBA_ASSERT(lambda >= 0);

            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    landmark_blocks_[r].set_landmark_damping_joint(lambda);
                }
            };

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);
        }

        void get_Hb(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
            if (options_.reduction_alg == 0) {
                std::cout << "get_hb_r starts \n";
                get_hb_r(H_pp, b_p);
            } else if (options_.reduction_alg == 1) {
                std::cout << "get_hb_f starts \n";
                get_hb_f(H_pp, b_p);
            } else {
                LOG(FATAL) << "options_.reduction_alg " << options_.reduction_alg
                           << " is not supported.";
            }
            H_pp.recompute_keys();
        }

        void get_Hb_refine(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
            if (options_.reduction_alg == 0) {
                std::cout << "get_hb_r starts \n";
                get_hb_r_refine(H_pp, b_p);
            } else if (options_.reduction_alg == 1) {
                std::cout << "get_hb_f starts \n";
                get_hb_f_refine(H_pp, b_p);
            } else {
                LOG(FATAL) << "options_.reduction_alg " << options_.reduction_alg
                           << " is not supported.";
            }
            H_pp.recompute_keys();
        }

        BlockDiagonalAccumulator<Scalar> get_jacobi() const {
            struct Reductor {
                Reductor(const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& lm_blocks)
                        : lm_blocks(lm_blocks) {}

                Reductor(Reductor& a, tbb::split /*unused*/) : lm_blocks(a.lm_blocks) {}

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = lm_blocks[r];
                        lb.add_jp_t_jp_blockdiag(accum);
                    }
                }

                inline void join(Reductor& b) { accum.join(b.accum); }

                BlockDiagonalAccumulator<Scalar> accum;
                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& lm_blocks;
            };

            Reductor r(landmark_blocks_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            if (has_pose_damping()) {
                r.accum.add_diag(
                        num_cameras_, POSE_SIZE,
                        VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_));
            }

            // TODO: double check including vs not including pose damping here in usage
            // and make it clear in API; see also getJp_diag2

            return r.accum;
        }

        void print_block(const std::string& filename, size_t block_idx) {
            landmark_blocks_[block_idx].printStorage(filename);
        }

    protected:
        void get_hb_r(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
            struct Reductor {
                Reductor(const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                landmark_blocks,
                         size_t num_cameras)
                        : landmark_blocks(landmark_blocks),
                          num_cameras(num_cameras),
                          H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
                    b_p.setZero(num_cameras * POSE_SIZE);
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : landmark_blocks(a.landmark_blocks),
                          num_cameras(a.num_cameras),
                          H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
                    b_p.setZero(a.num_cameras * POSE_SIZE);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Hb_varproj(H_pp, b_p);
                    }
                }

                inline void join(Reductor& other) {
                    H_pp.join(other.H_pp);
                    b_p += other.b_p;
                }

                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                size_t num_cameras;

                BlockSparseMatrix<Scalar> H_pp;
                VecX b_p;
            };

            Reductor r(landmark_blocks_, num_cameras_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            if (has_pose_damping()) {
                r.H_pp.add_diag(
                        num_cameras_, POSE_SIZE,
                        VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_));
            }

            H_pp = std::move(r.H_pp);
            b_p = std::move(r.b_p);
        }

        void get_hb_r_refine(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
            struct Reductor {
                Reductor(const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>&
                landmark_blocks,
                         size_t num_cameras)
                        : landmark_blocks(landmark_blocks),
                          num_cameras(num_cameras),
                          H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
                    b_p.setZero(num_cameras * POSE_SIZE);
                }

                Reductor(Reductor& a, tbb::split /*unused*/)
                        : landmark_blocks(a.landmark_blocks),
                          num_cameras(a.num_cameras),
                          H_pp(num_cameras * POSE_SIZE, num_cameras * POSE_SIZE) {
                    b_p.setZero(a.num_cameras * POSE_SIZE);
                }

                void operator()(const tbb::blocked_range<size_t>& range) {
                    for (size_t r = range.begin(); r != range.end(); ++r) {
                        const auto& lb = landmark_blocks[r];
                        lb.add_Hb(H_pp, b_p);
                    }
                }

                inline void join(Reductor& other) {
                    H_pp.join(other.H_pp);
                    b_p += other.b_p;
                }

                const std::vector<LandmarkBlockSC<Scalar, POSE_SIZE>>& landmark_blocks;
                size_t num_cameras;

                BlockSparseMatrix<Scalar> H_pp;
                VecX b_p;
            };

            Reductor r(landmark_blocks_, num_cameras_);

            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_reduce(range, r);

            if (has_pose_damping()) {
                r.H_pp.add_diag(
                        num_cameras_, POSE_SIZE_AFFINE,
                        VecX::Constant(num_cameras_ * POSE_SIZE_AFFINE, pose_damping_diagonal_));
            }

            H_pp = std::move(r.H_pp);
            b_p = std::move(r.b_p);
        }


        void get_hb_f(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
            ROOTBA_ASSERT(H_pp_mutex_.size() == num_cameras_ * num_cameras_);

            // Fill H_pp and b_p
            b_p.setZero(H_pp.rows);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_[r];
                    lb.add_Hb_varproj(H_pp, b_p, H_pp_mutex_, pose_mutex_, num_cameras_);
                }
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            if (has_pose_damping()) {
                H_pp.add_diag(
                        num_cameras_, POSE_SIZE,
                        VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_));
            }
        }

        void get_hb_f_refine(BlockSparseMatrix<Scalar>& H_pp, VecX& b_p) const {
            ROOTBA_ASSERT(H_pp_mutex_.size() == num_cameras_ * num_cameras_);

            // Fill H_pp and b_p
            b_p.setZero(H_pp.rows);
            auto body = [&](const tbb::blocked_range<size_t>& range) {
                for (size_t r = range.begin(); r != range.end(); ++r) {
                    const auto& lb = landmark_blocks_[r];
                    lb.add_Hb(H_pp, b_p, H_pp_mutex_, pose_mutex_, num_cameras_);
                }
            };
            tbb::blocked_range<size_t> range(0, landmark_blocks_.size());
            tbb::parallel_for(range, body);

            if (has_pose_damping()) {
                H_pp.add_diag(
                        num_cameras_, POSE_SIZE,
                        VecX::Constant(num_cameras_ * POSE_SIZE, pose_damping_diagonal_));
            }
        }

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
    };

}  // namespace rootba
