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

#include "rootba/solver/bal_bundle_adjustment.hpp"

#include <set>

#include <magic_enum/magic_enum.hpp>

#include "rootba/solver/linearizor.hpp"
#include "rootba/solver/solver_summary.hpp"
#include "rootba/util/format.hpp"
#include "rootba/util/own_or_reference.hpp"
#include "rootba/util/system_utils.hpp"
#include "rootba/util/tbb_utils.hpp"
#include "rootba/util/time_utils.hpp"

#include <random>
#include <cmath>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace rootba {

namespace {

void finish_iteration(SolverSummary& summary, IterationSummary& it_summary) {
  // step_solver_time: trying to count like ceres, all linear solver steps;
  // note that in "staged" execution mode 'scale_landmark_jacobian_time' and
  // 'perform_qr_time_in_seconds' will be 0 and thus not be reflected in the
  // step_solver_time (otoh we also cannot count the whole stage1_time, since
  // that would include jacobian_evaluation_time).
  it_summary.step_solver_time_in_seconds =
      it_summary.scale_landmark_jacobian_time_in_seconds +
      it_summary.perform_qr_time_in_seconds +
      it_summary.stage2_time_in_seconds +
      it_summary.solve_reduced_system_time_in_seconds +
      it_summary.back_substitution_time_in_seconds;

  // cost change
  if (it_summary.iteration > 0) {
    it_summary.cost_change =
        it_summary.cost.compared_to(summary.iterations.back().cost);
  }

  // memory
  {
    MemoryInfo mi;
    if (get_memory_info(mi)) {
      it_summary.resident_memory = mi.resident_memory;
      it_summary.resident_memory_peak = mi.resident_memory_peak;
    }
  }

  // push iteration
  summary.iterations.push_back(it_summary);

  // flush output
  std::cout.flush();
}

    void finish_iteration_inner(SolverSummary& summary, IterationSummary& it_summary) {
        // step_solver_time: trying to count like ceres, all linear solver steps;
        // note that in "staged" execution mode 'scale_landmark_jacobian_time' and
        // 'perform_qr_time_in_seconds' will be 0 and thus not be reflected in the
        // step_solver_time (otoh we also cannot count the whole stage1_time, since
        // that would include jacobian_evaluation_time).
        it_summary.step_solver_time_in_seconds =
                it_summary.scale_landmark_jacobian_time_in_seconds +
                it_summary.perform_qr_time_in_seconds +
                it_summary.stage2_time_in_seconds +
                it_summary.solve_reduced_system_time_in_seconds +
                it_summary.back_substitution_time_in_seconds;

        // cost change
        if (it_summary.iteration > 0) {
            it_summary.cost_change_inner =
                    it_summary.cost.compared_to(summary.iterations.back().cost_inner);
        }

        // memory
        {
            MemoryInfo mi;
            if (get_memory_info(mi)) {
                it_summary.resident_memory = mi.resident_memory;
                it_summary.resident_memory_peak = mi.resident_memory_peak;
            }
        }

        // push iteration
        summary.iterations.push_back(it_summary);

        // flush output
        std::cout.flush();
    }



void finish_solve(SolverSummary& summary, const SolverOptions& options) {
  switch (options.solver_type) {
    case SolverOptions::SolverType::SQUARE_ROOT:
      summary.solver_type = "bal_qr";
      break;
    case SolverOptions::SolverType::IMPLICIT_SQUARE_ROOT:
      summary.solver_type = "bal_implicit_qr";
      break;
    case SolverOptions::SolverType::EXPLICIT_POWER_SCHUR_COMPLEMENT:
      summary.solver_type = "bal_explicit_power_cg";
      break;
    case SolverOptions::SolverType::IMPLICIT_SQUARE_ROOT2:
      summary.solver_type = "bal_implicit_qr2";
      break;
    case SolverOptions::SolverType::FACTOR_SQUARE_ROOT:
      summary.solver_type = "bal_factor_qr";
      break;
    case SolverOptions::SolverType::SCHUR_COMPLEMENT:
      summary.solver_type = "bal_sc";
      break;
    case SolverOptions::SolverType::IMPLICIT_SCHUR_COMPLEMENT:
      summary.solver_type = "bal_implicit_sc";
      break;
    case SolverOptions::SolverType::FACTOR_SCHUR_COMPLEMENT:
      summary.solver_type = "bal_factor_sc";
      break;
    case SolverOptions::SolverType::POWER_SCHUR_COMPLEMENT:
      summary.solver_type = "bal_power_sc";
      break;
    case SolverOptions::SolverType::HYBRID_SCHUR_COMPLEMENT:
      summary.solver_type = "bal_hybrid_sc";
      break;
    case SolverOptions::SolverType::IMPLICIT_POWER_SCHUR_COMPLEMENT:
      summary.solver_type = "bal_implicit_power_sc";
      break;
    case SolverOptions::SolverType::POWER_VARPROJ:
      summary.solver_type = "power_variable_projection";
      break;
    case SolverOptions::SolverType::VARPROJ:
      summary.solver_type = "variable_projection";
      break;
    default:
      LOG(FATAL) << "unreachable";
  }

  summary.initial_cost = summary.iterations.front().cost;

  // final_cost: find last successful iteration
  for (auto it = summary.iterations.rbegin(); it != summary.iterations.rend();
       ++it) {
    if (it->step_is_successful) {
      summary.final_cost = it->cost;
      break;
    }
  }

  // TODO: this check depends on when error we use for LM... so maybe remove it
  // CHECK_LE(summary.final_cost.valid.error, summary.initial_cost.valid.error);

  summary.num_successful_steps =
      -1;  // don't count iteration 0, for which step_is_successful == true
  summary.num_unsuccessful_steps = 0;
  for (const auto& it : summary.iterations) {
    if (it.step_is_successful) {
      ++summary.num_successful_steps;
    } else {
      ++summary.num_unsuccessful_steps;
    }
  }

  summary.logging_time_in_seconds = 0;  // currently this is not computed

  summary.linear_solver_time_in_seconds = 0;
  summary.residual_evaluation_time_in_seconds = 0;
  summary.jacobian_evaluation_time_in_seconds = 0;
  for (const auto& it : summary.iterations) {
    summary.linear_solver_time_in_seconds += it.step_solver_time_in_seconds;
    summary.residual_evaluation_time_in_seconds +=
        it.residual_evaluation_time_in_seconds;
    summary.jacobian_evaluation_time_in_seconds +=
        it.jacobian_evaluation_time_in_seconds;
  }

  // memory & threads
  {
    MemoryInfo mi;
    if (get_memory_info(mi)) {
      summary.resident_memory_peak = mi.resident_memory_peak;
    }
  }

  // Effective available hardware threads (respecting process limits).
  summary.num_threads_available = tbb_task_arena_max_concurrency();
}

    void finish_solve_inner(SolverSummary& summary, const SolverOptions& options) {
        switch (options.solver_type) {
            case SolverOptions::SolverType::SQUARE_ROOT:
                summary.solver_type = "bal_qr";
                break;
            case SolverOptions::SolverType::IMPLICIT_SQUARE_ROOT:
                summary.solver_type = "bal_implicit_qr";
                break;
            case SolverOptions::SolverType::EXPLICIT_POWER_SCHUR_COMPLEMENT:
                summary.solver_type = "bal_explicit_power_cg";
                break;
            case SolverOptions::SolverType::IMPLICIT_SQUARE_ROOT2:
                summary.solver_type = "bal_implicit_qr2";
                break;
            case SolverOptions::SolverType::FACTOR_SQUARE_ROOT:
                summary.solver_type = "bal_factor_qr";
                break;
            case SolverOptions::SolverType::SCHUR_COMPLEMENT:
                summary.solver_type = "bal_sc";
                break;
            case SolverOptions::SolverType::IMPLICIT_SCHUR_COMPLEMENT:
                summary.solver_type = "bal_implicit_sc";
                break;
            case SolverOptions::SolverType::FACTOR_SCHUR_COMPLEMENT:
                summary.solver_type = "bal_factor_sc";
                break;
            case SolverOptions::SolverType::POWER_SCHUR_COMPLEMENT:
                summary.solver_type = "bal_power_sc";
                break;
            case SolverOptions::SolverType::HYBRID_SCHUR_COMPLEMENT:
                summary.solver_type = "bal_hybrid_sc";
                break;
            case SolverOptions::SolverType::IMPLICIT_POWER_SCHUR_COMPLEMENT:
                summary.solver_type = "bal_implicit_power_sc";
                break;
            case SolverOptions::SolverType::POWER_VARPROJ:
                summary.solver_type = "power_variable_projection";
                break;
            case SolverOptions::SolverType::VARPROJ:
                summary.solver_type = "variable_projection";
                break;
            default:
                LOG(FATAL) << "unreachable";
        }

        summary.initial_cost_inner = summary.iterations.front().cost_inner;

        // final_cost: find last successful iteration
        for (auto it = summary.iterations.rbegin(); it != summary.iterations.rend();
             ++it) {
            if (it->step_is_successful) {
                summary.final_cost_inner = it->cost_inner;
                break;
            }
        }

        // TODO: this check depends on when error we use for LM... so maybe remove it
        // CHECK_LE(summary.final_cost.valid.error, summary.initial_cost.valid.error);

        summary.num_successful_steps =
                -1;  // don't count iteration 0, for which step_is_successful == true
        summary.num_unsuccessful_steps = 0;
        for (const auto& it : summary.iterations) {
            if (it.step_is_successful) {
                ++summary.num_successful_steps;
            } else {
                ++summary.num_unsuccessful_steps;
            }
        }

        summary.logging_time_in_seconds = 0;  // currently this is not computed

        summary.linear_solver_time_in_seconds = 0;
        summary.residual_evaluation_time_in_seconds = 0;
        summary.jacobian_evaluation_time_in_seconds = 0;
        for (const auto& it : summary.iterations) {
            summary.linear_solver_time_in_seconds += it.step_solver_time_in_seconds;
            summary.residual_evaluation_time_in_seconds +=
                    it.residual_evaluation_time_in_seconds;
            summary.jacobian_evaluation_time_in_seconds +=
                    it.jacobian_evaluation_time_in_seconds;
        }

        // memory & threads
        {
            MemoryInfo mi;
            if (get_memory_info(mi)) {
                summary.resident_memory_peak = mi.resident_memory_peak;
            }
        }

        // Effective available hardware threads (respecting process limits).
        summary.num_threads_available = tbb_task_arena_max_concurrency();
    }

// compute actual cost difference for deciding if LM step was successful; this
// will be compared to model cost change
double compute_cost_decrease(
    const ResidualInfo& ri_before, const ResidualInfo& ri_after,
    const SolverOptions::OptimizedCost& optimized_cost) {
  switch (optimized_cost) {
    case SolverOptions::OptimizedCost::ERROR:
      return ri_before.all.error - ri_after.all.error;
    case SolverOptions::OptimizedCost::ERROR_VALID:
      return ri_before.valid.error - ri_after.valid.error;
    case SolverOptions::OptimizedCost::ERROR_VALID_AVG:
      return ri_before.valid.error_avg() - ri_after.valid.error_avg();
    default:
      LOG(FATAL) << "unreachable";
  }
}

// check termination based on change in cost value
bool function_tolerance_reached(const IterationSummary& it,
                                const SolverOptions& options,
                                std::string& message) {
  double cost;
  double change;
  switch (options.optimized_cost) {
    case SolverOptions::OptimizedCost::ERROR:
      cost = it.cost.all.error;
      change = std::abs(it.cost_change.all.error);
      break;
    case SolverOptions::OptimizedCost::ERROR_VALID:
    case SolverOptions::OptimizedCost::ERROR_VALID_AVG:
      cost = it.cost.valid.error;
      change = std::abs(it.cost_change.valid.error);
      break;
    default:
      LOG(FATAL) << "unreachable";
  }
  //if (change <= options.function_tolerance * cost) {
  if (change <= 1e-14 * cost) {
    message =
        "Function tolerance reached. |cost_change|/cost: {} <= {}"
        ""_format(change / cost, options.function_tolerance);
    return true;
  } else {
    return false;
  }
}

// check termination based on change in cost value
    bool function_tolerance_reached_inner(const IterationSummary& it,
                                    const SolverOptions& options,
                                    std::string& message) {
        double cost;
        double change;
        switch (options.optimized_cost) {
            case SolverOptions::OptimizedCost::ERROR:
                cost = it.cost_inner.all.error;
                change = std::abs(it.cost_change_inner.all.error);
                break;
            case SolverOptions::OptimizedCost::ERROR_VALID:
            case SolverOptions::OptimizedCost::ERROR_VALID_AVG:
                cost = it.cost_inner.valid.error;
                change = std::abs(it.cost_change_inner.valid.error);
                break;
            default:
                LOG(FATAL) << "unreachable";
        }

        if (change <= options.function_tolerance * cost) {
            message =
                    "Function tolerance reached. |cost_change|/cost: {} <= {}"
                    ""_format(change / cost, options.function_tolerance);
            return true;
        } else {
            return false;
        }
    }

// format info about new error, depending on 'optimized_cost' config
std::string format_new_error_info(
    const ResidualInfo& ri,
    const SolverOptions::OptimizedCost& optimized_cost) {
  switch (optimized_cost) {
    case SolverOptions::OptimizedCost::ERROR:
      return "error: {:.4e} (mean res: {:.2f}, num valid: {})"
             ""_format(ri.all.error, ri.all.residual_mean(), ri.valid.num_obs);
    case SolverOptions::OptimizedCost::ERROR_VALID:
      return "error valid: {:.4e} (mean res: {:.2f}, num: {})"
             ""_format(ri.valid.error, ri.valid.residual_mean(),
                       ri.valid.num_obs);
    case SolverOptions::OptimizedCost::ERROR_VALID_AVG:
      return "error valid avg: {:.4e} (mean res: {:.2f}, num: {})"
             ""_format(ri.valid.error_avg(), ri.valid.residual_mean(),
                       ri.valid.num_obs);
    default:
      LOG(FATAL) << "unreachable";
  }
}

void check_options(const SolverOptions& options) {
  CHECK(options.min_trust_region_radius <= options.initial_trust_region_radius)
      << "Invalid configuration";
  CHECK(options.initial_trust_region_radius <= options.max_trust_region_radius)
      << "Invalid configuration";

  if (options.preconditioner_type !=
          SolverOptions::PreconditionerType::JACOBI &&
      options.preconditioner_type !=
          SolverOptions::PreconditionerType::SCHUR_JACOBI) {
    LOG(FATAL) << "predonditioner {} not implemented"_format(
        wise_enum::to_string(options.preconditioner_type));
  }

  if (options.residual.robust_norm != BalResidualOptions::RobustNorm::NONE &&
      options.residual.robust_norm != BalResidualOptions::RobustNorm::HUBER) {
    LOG(FATAL) << "robust norm {} not implemented"_format(
        wise_enum::to_string(options.residual.robust_norm));
  }

  CHECK_GE(options.jacobi_scaling_epsilon, 0);
}

    template <typename Scalar>
    void refine(BalProblem<Scalar>& bal_problem,
                          const SolverOptions& solver_options,
                          SolverSummary& summary) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        //std::cout << "IN REFINEMENT , bal_problem.landmarks().at(0).p_w.norm() = " << bal_problem.landmarks().at(0).p_w.norm() << "\n";
        //std::cout << "IN REFINEMENT , bal_problem_.cameras().at(0).T_c_w.matrix().norm() = " << bal_problem.cameras().at(0).T_c_w.matrix().norm() << "\n";
        //std::cout << "IN REFINEMENT , bal_problem_.cameras().at(0).intrinsics.norm() = " << bal_problem.cameras().at(0).intrinsics.getParam().norm() << "\n";

        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
            //@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))
            //linearizor->initialize_varproj_lm(initialization_varproj);
            linearizor->compute_error_refine(ri);
            //initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }
            linearizor->linearize_refine();

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc = linearizor->solve_refine(lambda, relative_error_change);

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }

                bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error

                Scalar l_diff = linearizor->apply(std::move(inc));

                //Scalar l_diff = linearizor->closed_form(std::move(inc));

                ResidualInfo ri2;
                linearizor->compute_error_refine(ri2);
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    bal_problem.restore(); /// here we restore if it has been rejected
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }

        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void optimize_homogeneous_joint(BalProblem<Scalar>& bal_problem,
                          const SolverOptions& solver_options,
                          SolverSummary& summary, Timer<std::chrono::high_resolution_clock> timer_total) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;

        //Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        //summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        //std::unique_ptr<Linearizor<Scalar>> linearizor =
        //        Linearizor<Scalar>::create(bal_problem, solver_options, &summary);
        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create_pcg(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
            linearizor->compute_error_projective_space_homogeneous(ri, false);
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }

            linearizor->linearize_projective_space_homogeneous();
            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc = linearizor->solve_joint(lambda, relative_error_change);
                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }

                bal_problem.backup_joint();

                Scalar l_diff = linearizor->apply_joint(std::move(inc));

                ResidualInfo ri2;
                linearizor->compute_error_projective_space_homogeneous(ri2, false);
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    it_summary.step_is_valid = l_diff > 0;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            step_quality > solver_options.min_relative_decrease;
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    bal_problem.restore_joint();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }

        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void optimize_homogeneous_joint_RpOSE(BalProblem<Scalar>& bal_problem,
                                    const SolverOptions& solver_options,
                                    SolverSummary& summary, Timer<std::chrono::high_resolution_clock> timer_total) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;

        //Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        //const int max_lm_iter = solver_options.max_num_iterations;
        const int max_lm_iter = 50;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        //summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        //std::unique_ptr<Linearizor<Scalar>> linearizor =
        //        Linearizor<Scalar>::create(bal_problem, solver_options, &summary);
        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create_pcg(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
            linearizor->compute_error_projective_space_homogeneous_RpOSE(ri, false);
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }

            linearizor->linearize_projective_space_homogeneous_RpOSE();
            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc = linearizor->solve_joint_RpOSE(lambda, relative_error_change);
                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }

                bal_problem.backup_joint();

                Scalar l_diff = linearizor->apply_joint_RpOSE(std::move(inc));

                for (int i = 0; i < bal_problem.cameras().size(); i++) {
                    bal_problem.cameras().at(i).space_matrix.normalize();
                }
                for (int i =0; i< bal_problem.landmarks().size(); i++) {
                    bal_problem.landmarks().at(i).p_w_homogeneous = bal_problem.landmarks().at(i).p_w_homogeneous/bal_problem.landmarks().at(i).p_w_homogeneous[3];
                }

                ResidualInfo ri2;
                linearizor->compute_error_projective_space_homogeneous_RpOSE(ri2, false);
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    it_summary.step_is_valid = l_diff > 0;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            step_quality > solver_options.min_relative_decrease;
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    //lambda *= Scalar(1.0/3);
                    lambda = std::max(min_lambda, lambda);



                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    bal_problem.restore_joint();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }

        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }

template <typename Scalar>
void optimize_lm_ours(BalProblem<Scalar>& bal_problem,
                      const SolverOptions& solver_options,
                      SolverSummary& summary) {
    ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
    TbbConcurrencyObserver concurrency_observer;

    Timer timer_total;

    // preprocessor time includes allocating Linearizor (allocating landmark
    // blocks, factor grouping, etc); everything until the minimizer loop starts.
    Timer timer_preprocessor;

    using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
    const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
    const Scalar vee_factor(solver_options.vee_factor);
    const Scalar initial_vee(solver_options.initial_vee);

    //const int max_lm_iter = solver_options.max_num_iterations;
    const int max_lm_iter = solver_options.max_num_iterations_final;

    Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
    Scalar lambda_vee(initial_vee);

    // check options are valid and abort otherwise
    check_options(solver_options);

    summary = SolverSummary();
    summary.num_linear_solves = 0;
    summary.num_residual_evaluations = 0;
    summary.num_jacobian_evaluations = 0;

    std::unique_ptr<Linearizor<Scalar>> linearizor =
            Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

    // minmizer time starts after preprocessing
    summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
    Timer timer_minimizer;

    bool terminated = false;
    Scalar relative_error_change = 1;

    for (int it = 0; it <= max_lm_iter && !terminated;) {
        IterationSummary it_summary;
        it_summary.iteration = it;
        linearizor->start_iteration(&it_summary);

        Timer timer_iteration;

        // TODO: avoid recomputation of error if we call linearizor->linearize()
        // anyway (only for iteration 0 we can call compute_error instead)...
        ResidualInfo ri;
        linearizor->compute_error(ri);

        std::cout << "Iteration {}, {}\n"_format(
                it, error_summary_oneline(
                        ri, solver_options.use_projection_validity_check()));

        CHECK(ri.is_numerically_valid)
                        << "did not expect numerical failure during linearization";

        // iteration 0 is just error evaluation and logging
        if (it == 0) {
            linearizor->finish_iteration();
            it_summary.cost = ri;
            it_summary.trust_region_radius = 1 / lambda;
            it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
            it_summary.cumulative_time_in_seconds = timer_total.elapsed();
            it_summary.step_is_successful = true;
            it_summary.step_is_valid = true;
            finish_iteration(summary, it_summary);
            ++it;
            continue;
        }

        linearizor->linearize();

        std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                it_summary.stage1_time_in_seconds);

        // Don't limit inner lm iterations (to be in line with ceres)
        constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

        for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
            if (j > 0) {
                std::cout << "Iteration {}, backtracking\n"_format(it);

                it_summary = IterationSummary();
                it_summary.iteration = it;
                linearizor->start_iteration(&it_summary);

                timer_iteration.reset();
            }

            // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
            // IterationSummary with cost and cost_change. Consider the best way to
            // have a general way for linearizor to access this info instead of
            // passing `relative_error_change` to `solve(...)` which is very specific
            // to the current Hybrid solver.

            // dampen and solve linear system
            VecX inc = linearizor->solve(lambda, relative_error_change);
            std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                    it_summary.stage2_time_in_seconds);

            std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                         "{:.3f}s\n"_format(
                    it_summary.linear_solver_message,
                    it_summary.solve_reduced_system_time_in_seconds,
                    it_summary.solve_reduced_system_time_in_seconds /
                    it_summary.linear_solver_iterations);

            // TODO: cleanly abort linear solver on numerical issue

            // TODO: add to log: gradient norm, step norm

            if (!inc.array().isFinite().all()) {
                it_summary.step_is_valid = false;
                it_summary.step_is_successful = false;

                const double iteration_time = timer_iteration.elapsed();
                const double cumulative_time = timer_total.elapsed();

                std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                std::cout
                        << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                           "{:.3f}s, total_time: {:.3f}s\n"
                           ""_format(
                                reason,
                                "Numeric issues when computing increment (contains NaNs)",
                                lambda, it_summary.linear_solver_iterations, iteration_time,
                                cumulative_time);

                lambda = lambda_vee * lambda;
                lambda_vee *= vee_factor;

                linearizor->finish_iteration();
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = iteration_time;
                it_summary.cumulative_time_in_seconds = cumulative_time;
                it_summary.step_is_successful = false;
                finish_iteration(summary, it_summary);

                it++;

                if (lambda > max_lambda) {
                    terminated = true;
                    summary.termination_type = NO_CONVERGENCE;
                    summary.message =
                            "Solver did not converge and reached maximum "
                            "damping lambda of {}"_format(max_lambda);
                }

                continue;
            }

            bal_problem.backup();

            Scalar l_diff = linearizor->apply(std::move(inc));

            ResidualInfo ri2;
            linearizor->compute_error(ri2);
            it_summary.cost = ri2;
            relative_error_change =
                    std::abs(ri.all.error - ri2.all.error) / ri.all.error;

            if (!ri2.is_numerically_valid) {
                it_summary.step_is_valid = false;
                it_summary.step_is_successful = false;
                std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                        error_summary_oneline(
                                ri2, solver_options.use_projection_validity_check()));
            } else {
                // compute "ri - ri2", depending on 'optimized_cost' config
                Scalar f_diff =
                        compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                // model cost by the number of observations. The model assumes the
                // number of valid residuals remains unchanged, so we simply divide the
                // difference by the number of observations before the update.
                if (solver_options.optimized_cost ==
                    SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                    l_diff /= ri.valid.num_obs;
                }

                Scalar step_quality = f_diff / l_diff;

                std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                             "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                             ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                       ri2.valid.error);

                it_summary.relative_decrease = step_quality;

                it_summary.step_is_valid = l_diff > 0;
                it_summary.step_is_successful =
                        it_summary.step_is_valid &&
                        step_quality > solver_options.min_relative_decrease;
            }

            if (it_summary.step_is_successful) {
                ROOTBA_ASSERT(it_summary.step_is_valid);

                const double iteration_time = timer_iteration.elapsed();
                const double cumulative_time = timer_total.elapsed();

                std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                             "{:.3f}s, total_time: {:.3f}s\n"
                             ""_format(format_new_error_info(
                                               ri2, solver_options.optimized_cost),
                                       lambda, it_summary.linear_solver_iterations,
                                       iteration_time, cumulative_time);

                lambda *= Scalar(std::max(
                        1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                lambda = std::max(min_lambda, lambda);

                lambda_vee = initial_vee;

                linearizor->finish_iteration();
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = iteration_time;
                it_summary.cumulative_time_in_seconds = cumulative_time;
                finish_iteration(summary, it_summary);

                it++;

                // check function tolerance
                if (function_tolerance_reached(summary.iterations.back(),
                                               solver_options, summary.message)) {
                    terminated = true;
                    summary.termination_type = CONVERGENCE;
                }

                // stop inner lm loop
                break;
            } else {
                const double iteration_time = timer_iteration.elapsed();
                const double cumulative_time = timer_total.elapsed();

                std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                             "{:.3f}s, total_time: {:.3f}s\n"
                             ""_format(reason,
                                       format_new_error_info(
                                               ri2, solver_options.optimized_cost),
                                       lambda, it_summary.linear_solver_iterations,
                                       iteration_time, cumulative_time);

                lambda = lambda_vee * lambda;
                lambda_vee *= vee_factor;

                linearizor->finish_iteration();
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = iteration_time;
                it_summary.cumulative_time_in_seconds = cumulative_time;
                it_summary.step_is_successful = false;
                finish_iteration(summary, it_summary);

                bal_problem.restore();
                it++;

                if (lambda > max_lambda) {
                    terminated = true;
                    summary.termination_type = NO_CONVERGENCE;
                    summary.message =
                            "Solver did not converge and reached maximum "
                            "damping lambda of {}"_format(max_lambda);
                }
            }
        }
    }

    if (!terminated) {
        summary.termination_type = NO_CONVERGENCE;
        summary.message =
                "Solver did not converge after maximum number of "
                "{} iterations"_format(max_lm_iter);
    }

    summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
    summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
    summary.total_time_in_seconds = timer_total.elapsed();

    summary.num_threads_given = solver_options.num_threads;
    summary.num_threads_used = concurrency_observer.get_peak_concurrency();

    finish_solve(summary, solver_options);

    std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
            summary.final_cost, solver_options.use_projection_validity_check()));
    std::cout << "{}: {}\n"_format(
            magic_enum::enum_name(summary.termination_type), summary.message);
    std::cout.flush();
}

    template <typename Scalar>
    void optimize_lm_ours_affine_space(BalProblem<Scalar>& bal_problem,
                          const SolverOptions& solver_options,
                          SolverSummary& summary, Scalar lambda_old_) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))
            linearizor->initialize_varproj_lm_affine_space(initialization_varproj);
            linearizor->compute_error_affine_space(ri, initialization_varproj);

//@Simon: debug
            //linearizor->compute_error_refine(ri);
//
            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            linearizor->linearize_affine_space(); //@Simon: idea: use camera matrix space instead of SE(3)
//@Simon: debug
            //linearizor->linearize_refine();

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system

                VecX inc = linearizor->solve_affine_space(lambda, relative_error_change);
//@Simon: debug
                //VecX inc = linearizor->solve_refine(lambda, relative_error_change);

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_affine_space();
                //bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error
//@Simon: debug
                //Scalar l_diff = linearizor->apply(std::move(inc));
//
                Scalar l_diff = linearizor->closed_form_affine_space(std::move(inc));

                ResidualInfo ri2;
                linearizor->compute_error_affine_space(ri2, false);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_affine_space();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void optimize_lm_ours_pOSE(BalProblem<Scalar>& bal_problem,
                                       const SolverOptions& solver_options,
                                       SolverSummary& summary,
                               SolverSummary& summary_tmp,
                               Scalar lambda_old_, Timer<std::chrono::high_resolution_clock> timer_total) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        //Timer timer_total;
        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);
        summary = SolverSummary();
        summary_tmp = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;
        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);
        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))
            if (initialization_varproj) {
                linearizor->initialize_varproj_lm_pOSE(solver_options.alpha, initialization_varproj);
            }
            linearizor->compute_error_pOSE(ri, initialization_varproj);

            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                finish_iteration(summary_tmp, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            linearizor->linearize_pOSE(solver_options.alpha); //@Simon: idea: use camera matrix space instead of SE(3)
//@Simon: debug

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc;
                //if (solver_options.direct == true) {
                    inc = linearizor->solve_direct_pOSE(lambda, relative_error_change);
                //}
                //else {
                    //inc = linearizor->solve_pOSE(lambda, relative_error_change);
                    //inc = linearizor->solve_pOSE_poBA(lambda, relative_error_change);
                //}
                //
                //VecX inc = linearizor->solve_pOSE(lambda, relative_error_change);


                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);
                    //finish_iteration(summary_tmp, it_summary); //

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        //summary_tmp.termination_type = NO_CONVERGENCE;
                        //summary_tmp.message =
                        //        "Solver did not converge and reached maximum "
                        //        "damping lambda of {}"_format(max_lambda);
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_pOSE();
                //bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error
//@Simon: debug
                //Scalar l_diff = linearizor->apply(std::move(inc));
//
                //std::cout << "in pose,     before closed_form_pose \n";
                Scalar l_diff = linearizor->closed_form_pOSE(solver_options.alpha, std::move(inc));
                //Scalar l_diff = linearizor->closed_form_pOSE_poBA(solver_options.alpha, std::move(inc));
                //std::cout << "in pose,     after closed_form_pose \n";
                ResidualInfo ri2;
                linearizor->compute_error_pOSE(ri2, false);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    //finish_iteration(summary_tmp, it_summary);
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        //summary_tmp.termination_type = CONVERGENCE;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    //finish_iteration(summary_tmp, it_summary);
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_pOSE();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        //summary_tmp.termination_type = NO_CONVERGENCE;
                        //summary_tmp.message =
                        //        "Solver did not converge and reached maximum "
                        //        "damping lambda of {}"_format(max_lambda);

                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            //summary_tmp.termination_type = NO_CONVERGENCE;
            //summary_tmp.message =
            //        "Solver did not converge after maximum number of "
            //        "{} iterations"_format(max_lm_iter);
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);

        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        //summary_tmp.total_time_in_seconds = timer_total.elapsed();
        summary.postprocessor_time_in_seconds = 0;
        //summary_tmp.num_threads_given = solver_options.num_threads;
        //summary_tmp.num_threads_used = concurrency_observer.get_peak_concurrency();
        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);
        //finish_solve(summary_tmp, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary_tmp.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void optimize_lm_ours_RpOSE(BalProblem<Scalar>& bal_problem,
                               const SolverOptions& solver_options,
                               SolverSummary& summary,
                               SolverSummary& summary_tmp,
                               Scalar lambda_old_, Timer<std::chrono::high_resolution_clock> timer_total) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        //Timer timer_total;
        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);
        summary = SolverSummary();
        summary_tmp = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;
        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);
        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))
            if (initialization_varproj) {
                linearizor->initialize_varproj_lm_RpOSE(solver_options.alpha, initialization_varproj);
            }
            linearizor->compute_error_RpOSE(ri, initialization_varproj);
            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                finish_iteration(summary_tmp, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            linearizor->linearize_RpOSE(solver_options.alpha); //@Simon: idea: use camera matrix space instead of SE(3)
//@Simon: debug
            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc;
                //if (solver_options.direct == true) {
                //inc = linearizor->solve_direct_RpOSE(lambda, relative_error_change);
                //}
                //else {
                inc = linearizor->solve_RpOSE(lambda, relative_error_change);
                //inc = linearizor->solve_pOSE_poBA(lambda, relative_error_change);
                //}
                //
                //VecX inc = linearizor->solve_pOSE(lambda, relative_error_change);


                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);
                    //finish_iteration(summary_tmp, it_summary); //

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        //summary_tmp.termination_type = NO_CONVERGENCE;
                        //summary_tmp.message =
                        //        "Solver did not converge and reached maximum "
                        //        "damping lambda of {}"_format(max_lambda);
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_RpOSE();
                //bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error
//@Simon: debug
                //Scalar l_diff = linearizor->apply(std::move(inc));
//
                Scalar l_diff = linearizor->closed_form_RpOSE(solver_options.alpha, std::move(inc));
                //Scalar l_diff = linearizor->closed_form_pOSE_poBA(solver_options.alpha, std::move(inc));
                ResidualInfo ri2;
                linearizor->compute_error_RpOSE(ri2, false);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    //finish_iteration(summary_tmp, it_summary);
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        //summary_tmp.termination_type = CONVERGENCE;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    //finish_iteration(summary_tmp, it_summary);
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_RpOSE();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        //summary_tmp.termination_type = NO_CONVERGENCE;
                        //summary_tmp.message =
                        //        "Solver did not converge and reached maximum "
                        //        "damping lambda of {}"_format(max_lambda);

                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            //summary_tmp.termination_type = NO_CONVERGENCE;
            //summary_tmp.message =
            //        "Solver did not converge after maximum number of "
            //        "{} iterations"_format(max_lm_iter);
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);

        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        //summary_tmp.total_time_in_seconds = timer_total.elapsed();
        summary.postprocessor_time_in_seconds = 0;
        //summary_tmp.num_threads_given = solver_options.num_threads;
        //summary_tmp.num_threads_used = concurrency_observer.get_peak_concurrency();
        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);
        //finish_solve(summary_tmp, solver_options);

        //@Simon: prepare equilibrium for refinement -> try directly in the refinement function
        //linearizor->rpose_new_equilibrium();

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary_tmp.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void optimize_lm_ours_RpOSE_refinement(BalProblem<Scalar>& bal_problem,
                                const SolverOptions& solver_options,
                                SolverSummary& summary,
                                SolverSummary& summary_tmp,
                                Scalar lambda_old_, Timer<std::chrono::high_resolution_clock> timer_total, double alpha_divided) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        //Timer timer_total;
        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);
        summary = SolverSummary();
        summary_tmp = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;
        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);
        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;
        //@Simon: use a smaller alpha for each refinement
        //solver_options.alpha /= 10.0;


        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;

            //@Simon: prepare equilibrium for refinement
            if (initialization_varproj) {
                linearizor->rpose_new_equilibrium();
                initialization_varproj = false;
                std::cout << "NEW EQUILIBRIUM \n";
            }

            linearizor->compute_error_RpOSE_refinement(ri, solver_options.alpha / alpha_divided);
            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                finish_iteration(summary_tmp, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            linearizor->linearize_RpOSE_refinement(solver_options.alpha/ alpha_divided); //@Simon: idea: use camera matrix space instead of SE(3)
//@Simon: debug
            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc;
                //if (solver_options.direct == true) {
                //inc = linearizor->solve_direct_RpOSE(lambda, relative_error_change);
                //}
                //else {
                inc = linearizor->solve_RpOSE(lambda, relative_error_change);
                std::cout << "inc.norm() = " << inc.norm() << "\n";
                //inc = linearizor->solve_pOSE_poBA(lambda, relative_error_change);
                //}
                //
                //VecX inc = linearizor->solve_pOSE(lambda, relative_error_change);


                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);
                    //finish_iteration(summary_tmp, it_summary); //

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        //summary_tmp.termination_type = NO_CONVERGENCE;
                        //summary_tmp.message =
                        //        "Solver did not converge and reached maximum "
                        //        "damping lambda of {}"_format(max_lambda);
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_RpOSE();
                //bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error
//@Simon: debug
                //Scalar l_diff = linearizor->apply(std::move(inc));
                std::cout << "solver_options.alpha/ divided = " << solver_options.alpha/ alpha_divided << "\n";
                Scalar l_diff = linearizor->closed_form_RpOSE_refinement(solver_options.alpha/ alpha_divided, std::move(inc));
                //Scalar l_diff = linearizor->closed_form_pOSE_poBA(solver_options.alpha, std::move(inc));
                ResidualInfo ri2;
                linearizor->compute_error_RpOSE_refinement(ri2, solver_options.alpha/ alpha_divided);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    //finish_iteration(summary_tmp, it_summary);
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        //summary_tmp.termination_type = CONVERGENCE;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    //finish_iteration(summary_tmp, it_summary);
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_RpOSE();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        //summary_tmp.termination_type = NO_CONVERGENCE;
                        //summary_tmp.message =
                        //        "Solver did not converge and reached maximum "
                        //        "damping lambda of {}"_format(max_lambda);

                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            //summary_tmp.termination_type = NO_CONVERGENCE;
            //summary_tmp.message =
            //        "Solver did not converge after maximum number of "
            //        "{} iterations"_format(max_lm_iter);
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);

        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        //summary_tmp.total_time_in_seconds = timer_total.elapsed();
        summary.postprocessor_time_in_seconds = 0;
        //summary_tmp.num_threads_given = solver_options.num_threads;
        //summary_tmp.num_threads_used = concurrency_observer.get_peak_concurrency();
        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);
        //finish_solve(summary_tmp, solver_options);

        //@Simon: prepare equilibrium for refinement
        //linearizor->rpose_new_equilibrium();

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary_tmp.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void optimize_lm_ours_RpOSE_ML(BalProblem<Scalar>& bal_problem,
                                           const SolverOptions& solver_options,
                                           SolverSummary& summary,
                                           SolverSummary& summary_tmp,
                                           Scalar lambda_old_, Timer<std::chrono::high_resolution_clock> timer_total) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        //Timer timer_total;
        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);
        summary = SolverSummary();
        summary_tmp = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;
        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);
        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;
        //@Simon: use a smaller alpha for each refinement
        //solver_options.alpha /= 10.0;


        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;

            //@Simon: prepare equilibrium for refinement
            if (initialization_varproj) {
                linearizor->rpose_new_equilibrium();
                initialization_varproj = false;
                std::cout << "NEW EQUILIBRIUM \n";
            }

            linearizor->compute_error_RpOSE_ML(ri);
            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                finish_iteration(summary_tmp, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            linearizor->linearize_RpOSE_ML(); //@Simon: idea: use camera matrix space instead of SE(3)
//@Simon: debug
            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc;
                //if (solver_options.direct == true) {
                //inc = linearizor->solve_direct_RpOSE(lambda, relative_error_change);
                //}
                //else {
                inc = linearizor->solve_RpOSE_ML(lambda, relative_error_change);
                std::cout << "inc.norm() = " << inc.norm() << "\n";
                //inc = linearizor->solve_pOSE_poBA(lambda, relative_error_change);
                //}
                //
                //VecX inc = linearizor->solve_pOSE(lambda, relative_error_change);


                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);
                    //finish_iteration(summary_tmp, it_summary); //

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        //summary_tmp.termination_type = NO_CONVERGENCE;
                        //summary_tmp.message =
                        //        "Solver did not converge and reached maximum "
                        //        "damping lambda of {}"_format(max_lambda);
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_RpOSE();
                //bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error
//@Simon: debug
                //Scalar l_diff = linearizor->apply(std::move(inc));

                Scalar l_diff = linearizor->closed_form_RpOSE_ML(std::move(inc));
                //Scalar l_diff = linearizor->closed_form_pOSE_poBA(solver_options.alpha, std::move(inc));
                ResidualInfo ri2;
                linearizor->compute_error_RpOSE_ML(ri2);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    //finish_iteration(summary_tmp, it_summary);
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        //summary_tmp.termination_type = CONVERGENCE;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    //finish_iteration(summary_tmp, it_summary);
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_RpOSE();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        //summary_tmp.termination_type = NO_CONVERGENCE;
                        //summary_tmp.message =
                        //        "Solver did not converge and reached maximum "
                        //        "damping lambda of {}"_format(max_lambda);

                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            //summary_tmp.termination_type = NO_CONVERGENCE;
            //summary_tmp.message =
            //        "Solver did not converge after maximum number of "
            //        "{} iterations"_format(max_lm_iter);
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);

        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        //summary_tmp.total_time_in_seconds = timer_total.elapsed();
        summary.postprocessor_time_in_seconds = 0;
        //summary_tmp.num_threads_given = solver_options.num_threads;
        //summary_tmp.num_threads_used = concurrency_observer.get_peak_concurrency();
        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);
        //finish_solve(summary_tmp, solver_options);

        //@Simon: prepare equilibrium for refinement
        //linearizor->rpose_new_equilibrium();

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary_tmp.termination_type), summary.message);
        std::cout.flush();
    }





    template <typename Scalar>
    void optimize_lm_ours_expOSE(BalProblem<Scalar>& bal_problem,
                               const SolverOptions& solver_options,
                               SolverSummary& summary,
                                 SolverSummary& summary_tmp,
                                 Scalar lambda_old_, Timer<std::chrono::high_resolution_clock> timer_total) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        //Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary_tmp = SolverSummary();

        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))
            if (initialization_varproj) {
                linearizor->initialize_y_tilde_expose();
                linearizor->initialize_varproj_lm_expOSE(solver_options.alpha, initialization_varproj);
                bal_problem.backup_expOSE();

            }
            linearizor->compute_error_expOSE(ri, initialization_varproj);

            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                finish_iteration(summary_tmp, it_summary);
                ++it;
                continue;
            }

            linearizor->linearize_expOSE(solver_options.alpha, initialization_varproj); //@Simon: idea: use camera matrix space instead of SE(3)
            initialization_varproj = false;

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc;
                if (solver_options.direct == true) {
                    inc = linearizor->solve_direct_expOSE(lambda, relative_error_change);
                }
                else {
                    inc = linearizor->solve_expOSE(lambda, relative_error_change);
                }
                //VecX

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary_tmp, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary_tmp.termination_type = NO_CONVERGENCE;
                        summary_tmp.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_expOSE();

                Scalar l_diff = linearizor->closed_form_expOSE(solver_options.alpha, std::move(inc));
                ResidualInfo ri2;
                linearizor->compute_error_expOSE(ri2, false);

                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    //linearizor->update_y_tilde_expose();

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary_tmp, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary_tmp.iterations.back(),
                                                   solver_options, summary_tmp.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary_tmp, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_expOSE();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary_tmp.termination_type = NO_CONVERGENCE;
                        summary_tmp.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary_tmp.termination_type = NO_CONVERGENCE;
            summary_tmp.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        summary_tmp.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary_tmp.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary_tmp.total_time_in_seconds = timer_total.elapsed();
        summary_tmp.num_threads_given = solver_options.num_threads;
        summary_tmp.num_threads_used = concurrency_observer.get_peak_concurrency();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);
        finish_solve(summary_tmp, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }



    template <typename Scalar>
    void optimize_lm_ours_pOSE_rOSE(BalProblem<Scalar>& bal_problem,
                               const SolverOptions& solver_options,
                               SolverSummary& summary, Scalar lambda_old_) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))

            linearizor->initialize_varproj_lm_pOSE_rOSE(solver_options.alpha, initialization_varproj);
            linearizor->compute_error_pOSE_rOSE(ri, initialization_varproj);
//@Simon: debug
            //linearizor->compute_error_refine(ri);
//
            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            linearizor->linearize_pOSE_rOSE(solver_options.alpha); //@Simon: idea: use camera matrix space instead of SE(3)
//@Simon: debug

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc = linearizor->solve_pOSE_rOSE(lambda, relative_error_change);
//@Simon: debug
                //VecX inc = linearizor->solve_refine(lambda, relative_error_change);

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_pOSE();
                //bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error
//@Simon: debug
                //Scalar l_diff = linearizor->apply(std::move(inc));
//
                Scalar l_diff = linearizor->closed_form_pOSE_rOSE(solver_options.alpha, std::move(inc));
                ResidualInfo ri2;
                linearizor->compute_error_pOSE_rOSE(ri2, false);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_pOSE();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void optimize_lm_ours_rOSE(BalProblem<Scalar>& bal_problem,
                               const SolverOptions& solver_options,
                               SolverSummary& summary, Scalar lambda_old_) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))
            linearizor->initialize_varproj_lm_rOSE(solver_options.alpha, initialization_varproj);
            linearizor->compute_error_rOSE(ri, initialization_varproj);
//@Simon: debug
            //linearizor->compute_error_refine(ri);
//
            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            linearizor->linearize_rOSE(solver_options.alpha); //@Simon: idea: use camera matrix space instead of SE(3)
//@Simon: debug
            std::cout << "in pose,   after linearize_rose \n";

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                //std::cout << "in pose,   before solve_pose \n";
                VecX inc = linearizor->solve_rOSE(lambda, relative_error_change);
                //std::cout << "in pose,   after solve_pose \n";
//@Simon: debug
                //VecX inc = linearizor->solve_refine(lambda, relative_error_change);
                std::cout << "IN rOSE FORMULATION     inc = " << inc.norm() << "\n";

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_pOSE();

                Scalar l_diff = linearizor->closed_form_rOSE(solver_options.alpha, std::move(inc));
                //std::cout << "in pose,     after closed_form_pose \n";
                ResidualInfo ri2;
                linearizor->compute_error_rOSE(ri2, false);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_pOSE();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }

    //@Simon: as we consider homogeneous model, try to use Riemannian manifold optimization for poses.
    // Note that we still fix the 4th point coordinates to 1.
    template <typename Scalar>
    void optimize_lm_ours_pOSE_riemannian_manifold(BalProblem<Scalar>& bal_problem,
                               const SolverOptions& solver_options,
                               SolverSummary& summary, Scalar lambda_old_) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))
            linearizor->initialize_varproj_lm_pOSE(solver_options.alpha, initialization_varproj);
            linearizor->compute_error_pOSE(ri, initialization_varproj);
//@Simon: debug
            //linearizor->compute_error_refine(ri);
//
            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            linearizor->linearize_pOSE(solver_options.alpha); //@Simon: idea: use camera matrix space instead of SE(3)
//@Simon: debug
            std::cout << "in pose,   after linearize_pose \n";

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                //std::cout << "in pose,   before solve_pose \n";
                VecX inc = linearizor->solve_pOSE_riemannian_manifold(lambda, relative_error_change);
                //std::cout << "in pose,   after solve_pose \n";
//@Simon: debug
                //VecX inc = linearizor->solve_refine(lambda, relative_error_change);
                std::cout << "IN pOSE FORMULATION     inc = " << inc.norm() << "\n";

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_pOSE();
                //bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error
//@Simon: debug
                //Scalar l_diff = linearizor->apply(std::move(inc));
//
                //std::cout << "in pose,     before closed_form_pose \n";
                Scalar l_diff = linearizor->closed_form_pOSE_riemannian_manifold(solver_options.alpha, std::move(inc));
                //std::cout << "in pose,     after closed_form_pose \n";
                ResidualInfo ri2;
                linearizor->compute_error_pOSE(ri2, false);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_pOSE();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }


    template <typename Scalar>
    void optimize_lm_ours_pOSE_homogeneous(BalProblem<Scalar>& bal_problem,
                               const SolverOptions& solver_options,
                               SolverSummary& summary, Scalar lambda_old_) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: here derive the initial v*(u0) to get the initialization point (u0, v*(u0))
            std::cout << "in optimize,   l1771 ok \n";
            linearizor->initialize_varproj_lm_pOSE_homogeneous(solver_options.alpha, initialization_varproj);
            std::cout << "in optimize,   l1773 ok \n";
            linearizor->compute_error_pOSE_homogeneous(ri, initialization_varproj);
            std::cout << "in optimize,   l1775 ok \n";
//@Simon: debug
            //linearizor->compute_error_refine(ri);
//
            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }
            //linearizor->linearize();
            std::cout << "in optimize,   l1801 ok \n";
            linearizor->linearize_pOSE_homogeneous(solver_options.alpha); //@Simon: idea: use camera matrix space instead of SE(3)
            std::cout << "in optimize,   l1803 ok \n";
            //@Simon: debug
            //linearizor->linearize_refine();

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system
                VecX inc = linearizor->solve_pOSE_homogeneous(lambda, relative_error_change);
                std::cout << "in optimize,   l1832 ok \n";
//@Simon: debug
                //VecX inc = linearizor->solve_refine(lambda, relative_error_change);
                std::cout << "IN pOSE FORMULATION     inc = " << inc.norm() << "\n";

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_pOSE_homogeneous();

                Scalar l_diff = linearizor->closed_form_pOSE_homogeneous(solver_options.alpha, std::move(inc));
                ResidualInfo ri2;
                linearizor->compute_error_pOSE_homogeneous(ri2, false);
//@Simon: debug
                //linearizor->compute_error_refine(ri2);
//
                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    //std::cout << "l540    lambda = " << lambda << "\n";
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_pOSE_homogeneous();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        lambda_old_ = lambda;

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void create_homogeneous_landmark(BalProblem<Scalar>& bal_problem) {
        for (int i = 0; i < bal_problem.num_landmarks(); ++i) {

            bal_problem.landmarks().at(i).p_w_homogeneous.setZero();

//@Simon: without normalization:
            //bal_problem.landmarks().at(i).p_w_homogeneous = bal_problem.landmarks().at(i).p_w.homogeneous();

//@Simon: with normalization:
            bal_problem.landmarks().at(i).p_w_homogeneous = bal_problem.landmarks().at(i).p_w.homogeneous();
            //bal_problem.landmarks().at(i).p_w_homogeneous.normalize();
        }

        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            bal_problem.cameras().at(i).space_matrix.normalize();
        }
        //for (int i =0; i< bal_problem.landmarks().size(); i++) {
        //    bal_problem.landmarks().at(i).p_w_homogeneous.normalize();
        //}

}



    template <typename Scalar>
    void optimize_lm_landmark(std::unique_ptr<Linearizor<Scalar>>& linearizor, BalProblem<Scalar>& bal_problem, SolverSummary& summary,
                              const SolverOptions& solver_options) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;

        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        //const int max_lm_iter = solver_options.max_num_iterations;

        const int max_lm_iter = solver_options.max_num_iterations_inner;

        //Scalar lambda(lambda_old_);
        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);


        // minmizer time starts after preprocessing
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;

        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: in the refinement we use the result of linear VarPro (1st step). Try without this first step with random initialization
            linearizor->compute_error_projective_space_homogeneous_lm_landmark(ri, initialization_varproj);
            initialization_varproj = false;
            std::cout << "Inner iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                //@Simon: create a specific cost for inner iterations
                it_summary.cost_inner = ri;
                //it_summary.cost= ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration_inner(summary, it_summary);
                ++it;
                continue;
            }

            linearizor->linearize_projective_space_homogeneous_lm_landmark();

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Inner iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.



                // dampen and solve linear system
                bal_problem.backup_projective_space_homogeneous_lm_landmark();
                linearizor->non_linear_varpro_landmark(lambda);

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm


                ResidualInfo ri2;
                linearizor->compute_error_projective_space_homogeneous_lm_landmark(ri2, false);

                it_summary.cost_inner = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[inner EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    //if (solver_options.optimized_cost ==
                    //    SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                    //    l_diff /= ri.valid.num_obs;
                    //}

                    //Scalar step_quality = f_diff / l_diff;

                    //std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                    //             "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                    //             ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                    //                       ri2.valid.error);


                    std::cout << "\t[inner EVAL] f_diff {:.4e} "
                                 "ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, ri.valid.error,
                                           ri2.valid.error);

                    //it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[inner Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);
                    lambda *= Scalar(1.0 / 3);
                    //@Simon: TODO: define relative_decrease (and l_diff)
                    //lambda *= Scalar(std::max(
                    //        1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration_inner(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached_inner(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration_inner(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    //bal_problem.restore_projective_space();
                    bal_problem.restore_projective_space_homogeneous_lm_landmark();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        //std::cout << "IN INITIALIZATION, bal_problem.landmarks().at(0).p_w.norm() = " << bal_problem.landmarks().at(0).p_w.norm() << "\n";
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        finish_solve_inner(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost_inner, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }

    template <typename Scalar>
    void initialize_non_linear_varproj(std::unique_ptr<Linearizor<Scalar>>& linearizor,
                                       BalProblem<Scalar>& bal_problem,
                                       SolverSummary& summary,
                                       const SolverOptions& solver_options) {
        //std::cout << "in initialization,    start \n";
        linearizor->linearize_projective_space_homogeneous_lm_landmark();
        //std::cout << "in initialization,    before optimize_lm_landmark \n";
        optimize_lm_landmark(linearizor, bal_problem, summary, solver_options);
        //std::cout << "in initialization,    after optimize_lm_landmark \n";
        //linearizor->closed_form_projective_space_homogeneous_nonlinear_initialization_riemannian_manifold();
        //std::cout << "in initialization,    end \n";


    }


    template <typename Scalar>
    void optimize_lm_ours_projective_refinement(BalProblem<Scalar>& bal_problem,
                                       const SolverOptions& solver_options,
                                       SolverSummary& summary, Scalar lambda_old_) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;

        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        const int max_lm_iter = solver_options.max_num_iterations;

        //Scalar lambda(lambda_old_);
        Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_varproj = true;
        //std::cout << "before initiliaze nonlinear varproj \n";
        initialize_non_linear_varproj(linearizor, bal_problem, summary, solver_options);

        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: in the refinement we use the result of linear VarPro (1st step). Try without this first step with random initialization
            //@Simon: first step of Nonlinear VarProj algorithm: LM on landmark
            //std::cout << "before initiliaze nonlinear varproj \n";
            //initialize_non_linear_varproj(linearizor, bal_problem, summary, solver_options);
            //@Simon: second step: perform one additional GN iteration over v from (u,v*_0)
            //linearizor->closed_form_projective_space_homogeneous_nonlinear_initialization();
            //@Simon: now, we have v*(u) and we can consider
            linearizor->compute_error_projective_space_homogeneous(ri, initialization_varproj);

            initialization_varproj = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                std::cout << "in refine,    it == 0 \n";
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }
            linearizor->linearize_projective_space_homogeneous();

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                std::cout << "in refine,    in the for loop l1973 \n";
                if (j > 0) {
                    std::cout << "in refine,    j > 0 \n";
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.



                // dampen and solve linear system
                VecX inc = linearizor->solve_projective_space_homogeneous_riemannian_manifold(lambda, relative_error_change);

                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_projective_space_homogeneous();


                //@Simon: with Riemannian manifold optimization:
                linearizor->apply_pose_update_riemannian_manifold(std::move(inc));
                //@Simon: without Riemannian manifold optimization:
                //linearizor->apply_pose_update(std::move(inc));
                optimize_lm_landmark(linearizor, bal_problem, summary, solver_options);
                Scalar l_diff = linearizor->closed_form_projective_space_homogeneous(std::move(inc));


                ResidualInfo ri2;
                linearizor->compute_error_projective_space_homogeneous(ri2, false);

                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    Scalar step_quality = f_diff / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda *= Scalar(std::max(
                            1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    //std::cout << "";
                    bal_problem.restore_projective_space_homogeneous();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }


    template <typename Scalar>
    void create_metric_intrinsics(BalProblem<Scalar>& bal_problem) {

        using Vec3 = Mat<Scalar, 3, 1>;
        using Mat4 = Mat<Scalar, 4, 4>;
        using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            Vec3 K = Vec3(1.0 / bal_problem.cameras().at(i).intrinsics.getParam()[0], 1.0 / bal_problem.cameras().at(i).intrinsics.getParam()[0],1.0);
            bal_problem.cameras().at(i).space_matrix_intrinsics = K.asDiagonal() * bal_problem.cameras().at(i).space_matrix;
            //bal_problem.cameras().at(i).space_matrix_intrinsics = bal_problem.cameras().at(i).space_matrix;

            bal_problem.cameras().at(i).space_matrix_intrinsics.normalize();
        }
//@Simon: TRY
        Mat4 H_tilde;
        H_tilde.setZero();
        H_tilde.row(0) = bal_problem.cameras().at(0).space_matrix_intrinsics.row(0);
        H_tilde.row(1) = bal_problem.cameras().at(0).space_matrix_intrinsics.row(1);
        H_tilde.row(2) = bal_problem.cameras().at(0).space_matrix_intrinsics.row(2);
        H_tilde(3,3) = 1;
        bal_problem.h_euclidean().H = H_tilde.inverse();
        //bal_problem.h_euclidean().H = MatX::Identity(4,4);

        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            bal_problem.cameras().at(i).space_matrix_intrinsics =  bal_problem.cameras().at(i).space_matrix_intrinsics * bal_problem.h_euclidean().H;
            //bal_problem.cameras().at(i).space_matrix_intrinsics.normalize();

        }
        std::cout << "bal_problem.cameras().at(0).space_matrix_intrinsics = " << bal_problem.cameras().at(0).space_matrix_intrinsics << "\n";

        for (int i = 0; i < bal_problem.landmarks().size(); i++) {
            bal_problem.landmarks().at(i).p_w_homogeneous = H_tilde * bal_problem.landmarks().at(i).p_w_homogeneous;
            //bal_problem.landmarks().at(i).p_w_homogeneous = bal_problem.landmarks().at(i).p_w_homogeneous/bal_problem.landmarks().at(i).p_w_homogeneous[3];
        }
    }

    template <typename Scalar>
    void optimize_metric_upgrade(BalProblem<Scalar>& bal_problem,
                               const SolverOptions& solver_options,
                               SolverSummary& summary) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);
        TbbConcurrencyObserver concurrency_observer;
        Timer timer_total;

        // preprocessor time includes allocating Linearizor (allocating landmark
        // blocks, factor grouping, etc); everything until the minimizer loop starts.
        Timer timer_preprocessor;

        using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;


        using Vec3 = Mat<Scalar, 3, 1>;
        using Mat3 = Mat<Scalar, 3, 3>;
        using Mat4 = Mat<Scalar, 4, 4>;
        using Mat34 = Mat<Scalar, 3, 4>;
        using Mat43 = Mat<Scalar, 4, 3>;

        const Scalar min_lambda(1.0 / solver_options.max_trust_region_radius);
        const Scalar max_lambda(1.0 / solver_options.min_trust_region_radius);
        const Scalar vee_factor(solver_options.vee_factor);
        const Scalar initial_vee(solver_options.initial_vee);

        //const int max_lm_iter = solver_options.max_num_iterations;
        const int max_lm_iter = 500;

        //Scalar lambda(1.0 / solver_options.initial_trust_region_radius);
        Scalar lambda(0.0001);
        Scalar lambda_vee(initial_vee);

        // check options are valid and abort otherwise
        check_options(solver_options);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        // minmizer time starts after preprocessing
        summary.preprocessor_time_in_seconds = timer_preprocessor.elapsed();
        Timer timer_minimizer;

        bool terminated = false;
        Scalar relative_error_change = 1;

        bool initialization_metric_upgrade = true;
        for (int it = 0; it <= max_lm_iter && !terminated;) {
            IterationSummary it_summary;
            it_summary.iteration = it;
            linearizor->start_iteration(&it_summary);

            Timer timer_iteration;

            // TODO: avoid recomputation of error if we call linearizor->linearize()
            // anyway (only for iteration 0 we can call compute_error instead)...
            ResidualInfo ri;
//@Simon: we initialize the plane at infinity with linear formulation
            if (initialization_metric_upgrade) {
                linearizor->compute_plane_linearly();

                //bal_problem.h_euclidean().plan_infinity << 1,1,1;

//@SImon: following lines appear in MATLAB files... a typo?
                Mat4 H_tilde;
                H_tilde.setZero();
                H_tilde.row(0) = bal_problem.cameras().at(0).space_matrix_intrinsics.row(0);
                H_tilde.row(1) = bal_problem.cameras().at(0).space_matrix_intrinsics.row(1);
                H_tilde.row(2) = bal_problem.cameras().at(0).space_matrix_intrinsics.row(2);
                H_tilde(3,3) = 1;
                bal_problem.h_euclidean().H = H_tilde.inverse(); //@Simon: try
                //bal_problem.h_euclidean().H = MatX::Identity(4,4); //@Simon: try
                //@Simon:initial version:
                bal_problem.h_euclidean().Ht << MatX::Identity(3,3), bal_problem.h_euclidean().plan_infinity.transpose();
                //bal_problem.h_euclidean().H << MatX::Identity(3,3), bal_problem.h_euclidean().plan_infinity.homogeneous().transpose();
                //@SImon: try, to be in line with Pollefeys' paper:
                //Vec3 K = Vec3(bal_problem.cameras().at(0).intrinsics.getParam()[0], bal_problem.cameras().at(0).intrinsics.getParam()[0],1.0);
                //Mat3 K;
                //K.setZero();
                //K(0,0) = bal_problem.cameras().at(0).intrinsics.getParam()[0];
                //K(1,1) = bal_problem.cameras().at(0).intrinsics.getParam()[0];
                //K(2,2) = 1;
                //bal_problem.h_euclidean().Ht << K, bal_problem.h_euclidean().plan_infinity.transpose();
                //std::cout << "initial Ht = \n";
                //std::cout << bal_problem.h_euclidean().Ht << "\n";
                for (int i = 0; i < bal_problem.cameras().size(); i++) {
//@Simon: try
                    //bal_problem.cameras().at(i).space_matrix_intrinsics =  bal_problem.cameras().at(i).space_matrix_intrinsics * bal_problem.h_euclidean().H;
//
                    //bal_problem.cameras().at(i).Ht << MatX::Identity(3,3), bal_problem.h_euclidean().plan_infinity.transpose();

                    bal_problem.cameras().at(i).PH = bal_problem.cameras().at(i).space_matrix_intrinsics * bal_problem.h_euclidean().Ht;
                    bal_problem.cameras().at(i).PHHP = bal_problem.cameras().at(i).PH * bal_problem.cameras().at(i).PH.transpose();
                    //bal_problem.cameras().at(i).PHHP.normalize();
                    bal_problem.cameras().at(i).alpha = 1.0;
                    //bal_problem.cameras().at(i).alpha = bal_problem.cameras().at(i).PHHP.trace() / (bal_problem.cameras().at(i).PHHP.squaredNorm());

                }
                std::cout << "IN INITIALIZATION \n";
            }
            linearizor->compute_error_metric_upgrade(ri);

            //@Simon: initialize with closed form:
            //for (int i = 0; i < bal_problem.cameras().size(); i++){
            //    bal_problem.cameras().at(i).alpha = bal_problem.cameras().at(i).PHHP.completeOrthogonalDecomposition().pseudoInverse();
            //}

            if (initialization_metric_upgrade) {
                for (int i = 0; i < bal_problem.cameras().size(); i++) {
                    bal_problem.cameras().at(i).alpha = bal_problem.cameras().at(i).PHHP.trace() / (bal_problem.cameras().at(i).PHHP.squaredNorm());
                    //bal_problem.cameras().at(i).alpha = 1.0 / (bal_problem.cameras().at(i).PHHP.norm());

                }
            }

            //@Simon: compute best alpha
            //if (initialization_metric_upgrade) {
            //    for (int i = 0; i < bal_problem.cameras().size(); i++) {
            //        bal_problem.cameras().at(i).alpha = bal_problem.cameras().at(i).PHHP.trace() / (bal_problem.cameras().at(i).PHHP.squaredNorm());
            //    }
            //}
            initialization_metric_upgrade = false;
            std::cout << "Iteration {}, {}\n"_format(
                    it, error_summary_oneline(
                            ri, solver_options.use_projection_validity_check()));

            CHECK(ri.is_numerically_valid)
                            << "did not expect numerical failure during linearization";

            // iteration 0 is just error evaluation and logging
            if (it == 0) {
                linearizor->finish_iteration();
                it_summary.cost = ri;
                it_summary.trust_region_radius = 1 / lambda;
                it_summary.iteration_time_in_seconds = timer_iteration.elapsed();
                it_summary.cumulative_time_in_seconds = timer_total.elapsed();
                it_summary.step_is_successful = true;
                it_summary.step_is_valid = true;
                finish_iteration(summary, it_summary);
                ++it;
                continue;
            }

            //linearizor->linearize_metric_upgrade();
            linearizor->linearize_metric_upgrade_v2(); //@SImon

            std::cout << "\t[INFO] Stage 1 time {:.3f}s.\n"_format(
                    it_summary.stage1_time_in_seconds);

            // Don't limit inner lm iterations (to be in line with ceres)
            constexpr int MAX_INNER_IT = std::numeric_limits<int>::max();

            for (int j = 0; j < MAX_INNER_IT && it <= max_lm_iter && !terminated; j++) {
                if (j > 0) {
                    std::cout << "Iteration {}, backtracking\n"_format(it);

                    it_summary = IterationSummary();
                    it_summary.iteration = it;
                    linearizor->start_iteration(&it_summary);

                    timer_iteration.reset();
                }

                // TODO@demmeln(LOW, Niko): This duplicates logic that is already in
                // IterationSummary with cost and cost_change. Consider the best way to
                // have a general way for linearizor to access this info instead of
                // passing `relative_error_change` to `solve(...)` which is very specific
                // to the current Hybrid solver.

                // dampen and solve linear system

                VecX inc = linearizor->solve_metric_upgrade(lambda, relative_error_change);
                std::cout << "\t[INFO] Stage 2 time {:.3f}s.\n"_format(
                        it_summary.stage2_time_in_seconds);

                std::cout << "\t[CG] Summary: {} Time {:.3f}s. Time per iteration "
                             "{:.3f}s\n"_format(
                        it_summary.linear_solver_message,
                        it_summary.solve_reduced_system_time_in_seconds,
                        it_summary.solve_reduced_system_time_in_seconds /
                        it_summary.linear_solver_iterations);

                // TODO: cleanly abort linear solver on numerical issue

                // TODO: add to log: gradient norm, step norm

                if (!inc.array().isFinite().all()) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout
                            << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                               "{:.3f}s, total_time: {:.3f}s\n"
                               ""_format(
                                    reason,
                                    "Numeric issues when computing increment (contains NaNs)",
                                    lambda, it_summary.linear_solver_iterations, iteration_time,
                                    cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }

                    continue;
                }
                bal_problem.backup_metric_upgrade();
                //bal_problem.backup(); //@Simon: why??? perhaps to keep in mind the current values, in case the update in linearizor->apply gives a larger error
                //@Simon: equivalent to closed_form
                bal_problem.h_euclidean().plan_infinity += inc;
                std::cout << "inc of plan_infinity = " << inc << "\n";
                //Mat43 Ht;
                //bal_problem.h_euclidean().Ht << MatX::Identity(3,3), bal_problem.h_euclidean().plan_infinity.transpose();
                bal_problem.h_euclidean().Ht << MatX::Identity(3,3), bal_problem.h_euclidean().plan_infinity.transpose();

                //@Simon: try:
                //Mat3 K;
                //K.setZero();
                //K(0,0) = bal_problem.cameras().at(0).intrinsics.getParam()[0];
                //K(1,1) = bal_problem.cameras().at(0).intrinsics.getParam()[0];
                //K(2,2) = 1;
                //std::cout << "f(0) = " << bal_problem.cameras().at(0).intrinsics.getParam()[0] << "\n";
                //bal_problem.h_euclidean().Ht << K, bal_problem.h_euclidean().plan_infinity.transpose();
                //Ht.setZero();
                //Ht.template block<3,3>(0,0) = MatX::Identity(3,3);
                //Ht.row(3) = bal_problem.h_euclidean().plan_infinity.transpose();
                //Mat4 H_tmp;
                //H_tmp.setZero();
                //H_tmp.template block<4,3>(0,0) = Ht;
                //H_tmp(3,3) = 1;
                for (int i = 0; i < bal_problem.cameras().size(); i++) {
                    bal_problem.cameras().at(i).PH = bal_problem.cameras().at(i).space_matrix_intrinsics * bal_problem.h_euclidean().Ht;
                    bal_problem.cameras().at(i).PHHP =  bal_problem.cameras().at(i).PH *  bal_problem.cameras().at(i).PH.transpose();
                    //bal_problem.cameras().at(i).PHHP.normalize();
                    //bal_problem.cameras().at(i).alpha = bal_problem.cameras().at(i).PHHP.trace() / (bal_problem.cameras().at(i).PHHP.squaredNorm());
                    //bal_problem.cameras().at(i).alpha = 1.0 / (bal_problem.cameras().at(i).PHHP.norm());

                }
                //std::cout << "bal_problem.h_euclidean().Ht = " << bal_problem.h_euclidean().Ht << "\n";
                //std::cout << "in general,  alpha *  bal_problem.cameras().at(5).PHHP =  \n";
                //std::cout << bal_problem.cameras().at(5).alpha * bal_problem.cameras().at(5).PHHP << "\n";
                //@Simon: try to derive alpha iteratively:
                linearizor->estimate_alpha(std::move(inc), lambda);

                //bal_problem.h_euclidean().H = H_tmp; //@Simon: try

                ResidualInfo ri2;
                linearizor->compute_error_metric_upgrade(ri2);

                Scalar l_diff = 0; //@Simon: perhaps delete l_diff

                it_summary.cost = ri2;
                relative_error_change =
                        std::abs(ri.all.error - ri2.all.error) / ri.all.error;

                if (!ri2.is_numerically_valid) {
                    it_summary.step_is_valid = false;
                    it_summary.step_is_successful = false;
                    std::cout << "\t[EVAL] failed to evaluate cost: {}"_format(
                            error_summary_oneline(
                                    ri2, solver_options.use_projection_validity_check()));
                } else {
                    // compute "ri - ri2", depending on 'optimized_cost' config
                    Scalar f_diff =
                            compute_cost_decrease(ri, ri2, solver_options.optimized_cost);

                    // ... only in case of ERROR_VALID_AVG, do we need to normalize the
                    // model cost by the number of observations. The model assumes the
                    // number of valid residuals remains unchanged, so we simply divide the
                    // difference by the number of observations before the update.
                    if (solver_options.optimized_cost ==
                        SolverOptions::OptimizedCost::ERROR_VALID_AVG) {
                        l_diff /= ri.valid.num_obs;
                    }

                    //Scalar step_quality = f_diff / l_diff;
                    Scalar step_quality = f_diff;// / l_diff;

                    std::cout << "\t[EVAL] f_diff {:.4e} l_diff {:.4e} "
                                 "step_quality {:.4e} ri1 {:.4e} ri2 {:.4e}\n"
                                 ""_format(f_diff, l_diff, step_quality, ri.valid.error,
                                           ri2.valid.error);

                    it_summary.relative_decrease = step_quality;

                    //it_summary.step_is_valid = l_diff > 0; //@Simon: TRY WITHOUT THIS CONDITION
                    it_summary.step_is_valid = true;
                    it_summary.step_is_successful =
                            it_summary.step_is_valid &&
                            //step_quality > solver_options.min_relative_decrease &&
                            f_diff > 0; //@Simon: add this condition
                }

                if (it_summary.step_is_successful) {
                    ROOTBA_ASSERT(it_summary.step_is_valid);

                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::cout << "\t[Success] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    //lambda *= Scalar(std::max(
                    //        1.0 / 3, 1 - std::pow(2 * it_summary.relative_decrease - 1, 3)));
                    lambda *= Scalar(1.0/2);
                    //lambda *= Scalar(1.0/1.5);
                    lambda = std::max(min_lambda, lambda);

                    lambda_vee = initial_vee;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    finish_iteration(summary, it_summary);

                    it++;

                    // check function tolerance
                    if (function_tolerance_reached(summary.iterations.back(),
                                                   solver_options, summary.message)) {
                        terminated = true;
                        summary.termination_type = CONVERGENCE;
                    }

                    // stop inner lm loop
                    break;
                } else {
                    const double iteration_time = timer_iteration.elapsed();
                    const double cumulative_time = timer_total.elapsed();

                    std::string reason = it_summary.step_is_valid ? "Reject" : "Invalid";

                    std::cout << "\t[{}] {}, lambda: {:.1e}, cg_iter: {}, it_time: "
                                 "{:.3f}s, total_time: {:.3f}s\n"
                                 ""_format(reason,
                                           format_new_error_info(
                                                   ri2, solver_options.optimized_cost),
                                           lambda, it_summary.linear_solver_iterations,
                                           iteration_time, cumulative_time);

                    lambda = lambda_vee * lambda;
                    lambda_vee *= vee_factor;

                    linearizor->finish_iteration();
                    it_summary.trust_region_radius = 1 / lambda;
                    it_summary.iteration_time_in_seconds = iteration_time;
                    it_summary.cumulative_time_in_seconds = cumulative_time;
                    it_summary.step_is_successful = false;
                    finish_iteration(summary, it_summary);

                    //bal_problem.restore(); /// here we restore if it has been rejected
                    bal_problem.restore_metric_upgrade();
                    it++;

                    if (lambda > max_lambda) {
                        terminated = true;
                        summary.termination_type = NO_CONVERGENCE;
                        summary.message =
                                "Solver did not converge and reached maximum "
                                "damping lambda of {}"_format(max_lambda);
                    }
                }
            }
        }

        if (!terminated) {
            summary.termination_type = NO_CONVERGENCE;
            summary.message =
                    "Solver did not converge after maximum number of "
                    "{} iterations"_format(max_lm_iter);
        }
        Mat4 H2;
        H2.setZero();
        //@Simon: initial
        H2.template block<3,3>(0,0) = MatX::Identity(3,3);
        H2.row(3) = bal_problem.h_euclidean().plan_infinity.homogeneous().transpose();
        //@Simon: try:
        //H2(0,0) = bal_problem.cameras().at(0).intrinsics.getParam()[0] * bal_problem.cameras().at(0).intrinsics.getParam()[0];
        //H2(1,1) = bal_problem.cameras().at(0).intrinsics.getParam()[0] * bal_problem.cameras().at(0).intrinsics.getParam()[0];
        //H2(2,2) = 1;
        //@SImon: Je Hyeong
        bal_problem.h_euclidean().H = bal_problem.h_euclidean().H * H2;
        std::cout << "final:    H = " << bal_problem.h_euclidean().H << "\n";

        //@Simon:
        //bal_problem.h_euclidean().H = H2;

        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            //@Simon: initial
            bal_problem.cameras().at(i).space_matrix_intrinsics = bal_problem.cameras().at(i).space_matrix_intrinsics * H2;

            //bal_problem.cameras().at(i).space_matrix_intrinsics.normalize();
            //@Simon: try
            //bal_problem.cameras().at(i).alpha = (bal_problem.cameras().at(i).space_matrix_intrinsics * bal_problem.cameras().at(i).space_matrix_intrinsics.transpose()).trace() / (bal_problem.cameras().at(i).space_matrix_intrinsics * bal_problem.cameras().at(i).space_matrix_intrinsics.transpose() ).squaredNorm();
        }

        for (int i = 0; i < bal_problem.landmarks().size(); i++) {
            bal_problem.landmarks().at(i).p_w_homogeneous = bal_problem.h_euclidean().H.inverse() * bal_problem.landmarks().at(i).p_w_homogeneous;
            bal_problem.landmarks().at(i).p_w = bal_problem.landmarks().at(i).p_w_homogeneous.template head(3) / bal_problem.landmarks().at(i).p_w_homogeneous(3);
        }
        ResidualInfo ri;

        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            Mat3 P_tmp = bal_problem.cameras().at(i).space_matrix_intrinsics.template block<3,3>(0,0);
            if (P_tmp.determinant() < 0) {
                bal_problem.cameras().at(i).space_matrix_intrinsics = - bal_problem.cameras().at(i).space_matrix_intrinsics * sqrt(bal_problem.cameras().at(i).alpha);
            }
            else {
                bal_problem.cameras().at(i).space_matrix_intrinsics = bal_problem.cameras().at(i).space_matrix_intrinsics * sqrt(bal_problem.cameras().at(i).alpha);
            }
            //std::cout << "space_matrix_intrinsics = " << bal_problem.cameras().at(i).space_matrix_intrinsics << "\n";
        }

        //@Simon: add chiraliry constraint
        int neg_depth = 0;
        int pos_depth = 0;
        for (int i = 0; i < bal_problem.landmarks().size(); i++) {
            for (const auto& [cam_idx, obs] : bal_problem.landmarks()[i].obs) {
                if (bal_problem.cameras().at(cam_idx).space_matrix_intrinsics.row(2)* bal_problem.landmarks().at(i).p_w.homogeneous()< 0) {
                    neg_depth += 1;
                }
                else {
                    pos_depth +=1;
                }
            }
        }
        std::cout << "neg_depth = " << neg_depth << "\n";
        std::cout << "pos_depth = " << pos_depth << "\n";
        if (neg_depth > pos_depth) {
            for (int i = 0; i < bal_problem.cameras().size(); i++) {
                bal_problem.cameras().at(i).space_matrix_intrinsics.col(3) = - bal_problem.cameras().at(i).space_matrix_intrinsics.col(3);
            }
            for (int i = 0; i < bal_problem.landmarks().size(); i++) {
                bal_problem.landmarks().at(i).p_w = -bal_problem.landmarks().at(i).p_w;
            }
        }

        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            Vec3 K = Vec3(bal_problem.cameras().at(i).intrinsics.getParam()[0], bal_problem.cameras().at(i).intrinsics.getParam()[0],1.0);
            bal_problem.cameras().at(i).space_matrix = K.asDiagonal() * bal_problem.cameras().at(i).space_matrix_intrinsics;
            //bal_problem.cameras().at(i).space_matrix = bal_problem.cameras().at(i).space_matrix_intrinsics;

        }
        for (int i =0; i< bal_problem.landmarks().size(); i++) {
            bal_problem.landmarks().at(i).p_w_homogeneous = bal_problem.landmarks().at(i).p_w.homogeneous();// / bal_problem.landmarks().at(i).p_w_homogeneous(3);

        }
        //linearizor->compute_error_projective_space_homogeneous(ri, false);
        linearizor->compute_error_projective_space_homogeneous_RpOSE(ri,false);
        std::cout << "DEBUG 3 Iteration, {}\n"_format(
                error_summary_oneline(
                        ri, solver_options.use_projection_validity_check()));

        //@Simon: derive closest perspective camera
        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            Mat3 R = bal_problem.cameras().at(i).space_matrix_intrinsics.template block<3,3>(0,0);

            Eigen::JacobiSVD<Mat3> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
            //Eigen::JacobiSVD<Mat3, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
            R = svd.matrixU() * svd.matrixV().transpose();
            std::cout << "R.singularValues() = " << svd.singularValues() << "\n";
            if (R.determinant() < 0) {
                Mat3 U_tmp;
                U_tmp.col(0) = svd.matrixU().col(0);
                U_tmp.col(1) = svd.matrixU().col(1);
                U_tmp.col(2) = -svd.matrixU().col(2);
                //U_tmp.col(2) = svd.matrixU().col(2);
                R = U_tmp * svd.matrixV().transpose();
                std::cout << "R.determinant < 0" << "\n";
            }
            //std::cout << "before SVD:  " << bal_problem.cameras().at(i).space_matrix_intrinsics.template block<3,3>(0,0) << "\n";
            //std::cout << "after SVD:  " << R << "\n";
            bal_problem.cameras().at(i).space_matrix_intrinsics.template block<3,3>(0,0) = R;
            //bal_problem.cameras().at(i).space_matrix_intrinsics.template block<3,1>(0,3) = - R * bal_problem.cameras().at(i).space_matrix_intrinsics.col(3);
        }

        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            Vec3 K = Vec3(bal_problem.cameras().at(i).intrinsics.getParam()[0], bal_problem.cameras().at(i).intrinsics.getParam()[0],1.0);
            bal_problem.cameras().at(i).space_matrix = K.asDiagonal() * bal_problem.cameras().at(i).space_matrix_intrinsics;
            //bal_problem.cameras().at(i).space_matrix =  bal_problem.cameras().at(i).space_matrix_intrinsics;

        }
        for (int i =0; i< bal_problem.landmarks().size(); i++) {
            bal_problem.landmarks().at(i).p_w_homogeneous = bal_problem.landmarks().at(i).p_w.homogeneous();// / bal_problem.landmarks().at(i).p_w_homogeneous(3);

        }
        //linearizor->compute_error_projective_space_homogeneous(ri, false);
        //linearizor->compute_error_projective_space_homogeneous_RpOSE(ri,false);
        linearizor->compute_error_projective_space_homogeneous_RpOSE_test_rotation(ri,false);
        std::cout << "DEBUG 4 Iteration, {}\n"_format(
                error_summary_oneline(
                        ri, solver_options.use_projection_validity_check()));

        for (int i = 0; i < bal_problem.cameras().size(); i++) {
            Mat3 R = bal_problem.cameras().at(i).space_matrix_intrinsics.template block<3,3>(0,0);
            Vec3 t = bal_problem.cameras().at(i).space_matrix_intrinsics.template block<3,1>(0,3);
            //Scalar theta = acos((R.trace() - 1.0)/2);
            //Eigen::Matrix<Scalar, 3, 1> omega;
            //omega(0) = R(2,1) - R(1,2);
            //omega(1) = R(0,2) - R(2,0);
            //omega(2) = R(1,0) - R(0,1);
            //omega *= 1/(2 * sin(theta));
            //std::cout << "l4753 ok \n";
            //Eigen::Matrix<Scalar, 3, 3> R_f = Eigen::AngleAxis<Scalar>(theta, omega).toRotationMatrix();     // PI/2 rotation along z axis
            Sophus::SE3<Scalar> T_c(R,t);
            //T_c.so3() = R_f;
            //T_c.translation() = t;
            //Sophus::SE3<Scalar> T_c(R,t);
            //bal_problem.cameras().at(i).T_c_w = T_c;
            //std::cout << "l4759 ok \n";
            //std::cout << "T_c.matrix() = " << T_c.matrix() << "\n";
            //std::cout << "T_c    t = " << T_c.translation() << "\n";
            //bal_problem.cameras().at(i).T_c_w = Sophus::se3_logd(T_c);
            bal_problem.cameras().at(i).T_c_w = T_c;
        }


        summary.minimizer_time_in_seconds = timer_minimizer.elapsed();
        summary.postprocessor_time_in_seconds = 0;  // currently no postprocessing
        summary.total_time_in_seconds = timer_total.elapsed();

        summary.num_threads_given = solver_options.num_threads;
        summary.num_threads_used = concurrency_observer.get_peak_concurrency();

        //lambda_old_ = lambda;

        finish_solve(summary, solver_options);

        std::cout << "Final Cost: {}\n"_format(error_summary_oneline(
                summary.final_cost, solver_options.use_projection_validity_check()));
        std::cout << "{}: {}\n"_format(
                magic_enum::enum_name(summary.termination_type), summary.message);
        std::cout.flush();
    }
    // namespace

    template <typename Scalar>
    void radial_estimation(BalProblem<Scalar>& bal_problem, const SolverOptions& solver_options,
                           SolverSummary& summary,
                           Timer<std::chrono::high_resolution_clock> timer_total) {
        ScopedTbbThreadLimit thread_limit(solver_options.num_threads);

        summary = SolverSummary();
        summary.num_linear_solves = 0;
        summary.num_residual_evaluations = 0;
        summary.num_jacobian_evaluations = 0;

        std::unique_ptr<Linearizor<Scalar>> linearizor =
                Linearizor<Scalar>::create(bal_problem, solver_options, &summary);

        linearizor->radial_estimate();


    }


}  // namespace

template <typename Scalar>
void bundle_adjust_manual(BalProblem<Scalar>& bal_problem,
                          const SolverOptions& solver_options,
                          SolverSummary* output_solver_summary,
                          PipelineTimingSummary* output_timing_summary, SolverSummary* output_solver_summary_tmp) {
  OwnOrReference<SolverSummary> solver_summary(output_solver_summary);
  //SolverSummary* output_solver_summary_tmp;
  OwnOrReference<SolverSummary> solver_summary_tmp(output_solver_summary_tmp);
  //optimize_lm_ours(bal_problem, solver_options, *solver_summary);

  //@Simon: first step: linear VARPRO
  Scalar lambda_old_;
  Timer timer_total;
  if (solver_options.varpro == SolverOptions::VarProType::AFFINE) {
      optimize_lm_ours_affine_space(bal_problem, solver_options, *solver_summary, lambda_old_);
  }
  else if (solver_options.varpro == SolverOptions::VarProType::HOMO) {
      optimize_lm_ours_pOSE_riemannian_manifold(bal_problem, solver_options, *solver_summary, lambda_old_);
      create_homogeneous_landmark(bal_problem);
      //optimize_lm_ours_pOSE_homogeneous(bal_problem, solver_options, *solver_summary, lambda_old_);
  }
  else if (solver_options.varpro == SolverOptions::VarProType::ROSE) {
      optimize_lm_ours_pOSE_rOSE(bal_problem, solver_options, *solver_summary, lambda_old_);
      create_homogeneous_landmark(bal_problem);
  }
  else if (solver_options.varpro == SolverOptions::VarProType::EXPOSE) {
      optimize_lm_ours_expOSE(bal_problem, solver_options, *solver_summary, *solver_summary_tmp, lambda_old_, timer_total);
      create_homogeneous_landmark(bal_problem);
  }
  else if (solver_options.varpro == SolverOptions::VarProType::RPOSE) {
      optimize_lm_ours_RpOSE(bal_problem, solver_options, *solver_summary, *solver_summary_tmp, lambda_old_, timer_total);

      optimize_lm_ours_RpOSE_refinement(bal_problem, solver_options, *solver_summary, *solver_summary_tmp, lambda_old_, timer_total, 100.0);
      //optimize_lm_ours_RpOSE_refinement(bal_problem, solver_options, *solver_summary, *solver_summary_tmp, lambda_old_, timer_total, 100.0);

      optimize_lm_ours_RpOSE_ML(bal_problem, solver_options, *solver_summary, *solver_summary_tmp, lambda_old_, timer_total);

      //radial_estimation(bal_problem,solver_options, *solver_summary, timer_total);

      create_homogeneous_landmark(bal_problem);


  }
  else {
      optimize_lm_ours_pOSE(bal_problem, solver_options, *solver_summary, *solver_summary_tmp, lambda_old_, timer_total);
      //create_homogeneous_landmark(bal_problem);
  }
    //optimize_lm_ours_affine_space(bal_problem, solver_options, *solver_summary, lambda_old_);

//@Simon: second step: nonlinear VARPRO
    //if (solver_options.joint == true) {
        optimize_homogeneous_joint_RpOSE(bal_problem, solver_options, *solver_summary, timer_total);
    //}
    //else{
    //    optimize_lm_ours_projective_refinement(bal_problem, solver_options, *solver_summary,lambda_old_);
    //    //optimize_homogeneous_joint(bal_problem, solver_options, *solver_summary);
    //}

//@Simon: third step: metric upgrade
    create_metric_intrinsics(bal_problem);
    optimize_metric_upgrade(bal_problem, solver_options, *solver_summary);

//@Simon: finally, we can solve the traditional BA
    //solver_options.optimized_cost == SolverOptions::OptimizedCost::ERROR_VALID;
    //optimize_lm_ours(bal_problem, solver_options, *solver_summary);



  if (output_timing_summary) {
    output_timing_summary->optimize_time =
        solver_summary->total_time_in_seconds;
  }

  LOG(INFO) << solver_summary->message;

  // TODO: Print summary similar to ceres: oneline, or full
}

#ifdef ROOTBA_INSTANTIATIONS_FLOAT
template void bundle_adjust_manual(
    BalProblem<float>& bal_problem, const SolverOptions& solver_options,
    SolverSummary* output_solver_summary,
    PipelineTimingSummary* output_timing_summary, SolverSummary* output_solver_summary_tmp);
#endif

#ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template void bundle_adjust_manual(
    BalProblem<double>& bal_problem, const SolverOptions& solver_options,
    SolverSummary* output_solver_summary,
    PipelineTimingSummary* output_timing_summary, SolverSummary* output_solver_summary_tmp);
#endif

}  // namespace rootba
