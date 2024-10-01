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

#include "rootba_povar/bal/ba_log_options.hpp"
#include "rootba_povar/bal/bal_residual_options.hpp"
#include "rootba_povar/options/visitable_options.hpp"

namespace rootba_povar {

/// Main options to configure the BA solver. Many options correspond to Ceres'
/// solver options.
struct SolverOptions : public VisitableOptions<SolverOptions> {
  // ////////////////////////////////////////////////////////////
  // enumerations
  // ////////////////////////////////////////////////////////////

  // which cost to consider in LM to check error reduction
  WISE_ENUM_CLASS_MEMBER(
      OptimizedCost,   //!
      (ERROR, 0),      //!< sum of all squared residuals
      ERROR_VALID,     //!< sum of valid squared residuals (positive z)
      ERROR_VALID_AVG  //!< mean of valid squared residuals
  );

  // which linearization to use for manual solver
  WISE_ENUM_CLASS_MEMBER(
      SolverType,
      // VarProj with classical Preconditioned Conjugate Gradients
      PCG,
      // Power Schur Complement for Scalable Bundle Adjustment (PoBA) https://arxiv.org/abs/2204.12834
      POWER_SCHUR_COMPLEMENT,
      // Power VarProj https://arxiv.org/abs/2405.05079
      POWER_VARPROJ,
      // Direct cholesky factorization
      CHOLESKY);

    WISE_ENUM_CLASS_MEMBER(
            SolverTypeRiemannian,
    // PoBA (https://arxiv.org/abs/2204.12834) with Riemannian manifold framework
            RIPOBA,
    // Preconditioned conjugate gradients with Riemannian manifold framework
            RIPCG);

  // see ceres::PreconditionerType
  // (rootba solvers and manual SC solvers don't implement all types)
  WISE_ENUM_CLASS_MEMBER(PreconditionerType, (IDENTITY, 0), JACOBI,
                         SCHUR_JACOBI, CLUSTER_JACOBI, CLUSTER_TRIDIAGONAL);

  // see ceres::LinearSolverType
  WISE_ENUM_CLASS_MEMBER(LinearSolverType, (DENSE_NORMAL_CHOLESKY, 0), DENSE_QR,
                         SPARSE_NORMAL_CHOLESKY, DENSE_SCHUR, SPARSE_SCHUR,
                         ITERATIVE_SCHUR, CGNR);

  BEGIN_VISITABLES(SolverOptions);

  // ////////////////////////////////////////////////////////////
  // common options
  // ////////////////////////////////////////////////////////////

  // solver type
  VISITABLE_META(SolverType, solver_type_step_1,
                 init(SolverType::POWER_VARPROJ)
                     .help("Solver type; 'POWER_VARPROJ' for Power Variable Projection, "
                           "'PCG' for Variable Projection with classical PCG,"
                           "'POWER_SCHUR_COMPLEMENT' for Levenberg-Marquardt with Power SC"
                           "'CHOLESKY' for Variable Projection with classical Cholesky factorization."));

    // solver type
    VISITABLE_META(SolverTypeRiemannian, solver_type_step_2,
                   init(SolverTypeRiemannian::RIPOBA)
                           .help("Solver type; 'RIPOBA' for Power Bundle Adjustment with Riemannian manifold framework, "
                                 "'RIPCG' for classical PCG with Riemannian manifold framework."));

  // verbosity
  VISITABLE_META(
      int, verbosity_level,
      init(2).range(0, 2).help("Output verbosity level. 0: silent, 1: brief "
                               "report (one line), 2: full report"));

  // debug output
  VISITABLE_META(bool, debug,
                 init(false).help("if true, print out additional info all "
                                  "around; may slow down runtime"));

  // num threads
  VISITABLE_META(int, num_threads,
                 init(0).range(0, 1000).help(
                     "number of threads to use for optimization. 0 means the "
                     "system should determine a suitable number "
                     "(e.g. number of virutal cores available)"));

  // residual formulation options
  VISITABLE(BalResidualOptions, residual);

  VISITABLE_META(double, alpha, init(0.01).range(0,1).help("weight in front of affine part, in pOSE optimization"));

  // iteration log
  VISITABLE(BaLogOptions, log);

  // optimized cost in lm (for ceres we can only do Error and ErrorValid)
  VISITABLE_META(
      OptimizedCost, optimized_cost,
      init(OptimizedCost::ERROR)
          .help("Which cost to consider for the 'cost decreased?' check in "
                "Levenberg-Marqardt. ERROR considers all residuals, "
                "ERROR_VALID ignores residuals with negative z, and "
                "ERROR_VALID_AVG compares the average of over valid residuals "
                "(only for non-ceres solvers)."));

  /// returns false for optimized_cost == Error, else true
  bool use_projection_validity_check() const;

  // ////////////////////////////////////////////////////////////
  // manual solver / ceres options
  // ////////////////////////////////////////////////////////////

  // maximum number of iterations
  VISITABLE_META(
      int, max_num_iterations_step_1,
      init(50).range(0, 10000).help("maximum number of solver iterations for pOSE optimization (0 "
                                    "means just initialize and exit)"));

    // maximum number of iterations
    VISITABLE_META(
            int, max_num_iterations_step_2,
            init(50).range(0, 10000).help("maximum number of solver iterations for joint homogeneous optimization (0 "
                                          "means just initialize and exit)"));

  // minimum relative decrease (ceres default: 1e-3)
  VISITABLE_META(double, min_relative_decrease,
                 init(0).help("Lower cound for the relative decrease before a "
                              "step is accepted (see Ceres)."));

  // trust region radius (ceres defaults init/min/max: 1e4/1e-32/1e16)
  VISITABLE_META(double, initial_trust_region_radius,
                 init(1e4)
                     .range(1e-10, 1e16)
                     .logscale()
                     .help("Determines the initial damping (see Ceres)."));
  VISITABLE_META(double, min_trust_region_radius,
                 init(1e-32)
                     .range(1e-32, 1e16)
                     .logscale()
                     .help("Optimization terminates if the trust region radius "
                           "becomes smaller than this value (see Ceres)."));
  VISITABLE_META(
      double, max_trust_region_radius,
      init(1e16)
          .range(1e-16, 1e16)
          .logscale()
          .help("Defines the minimum damping we always add (see Ceres)."));

  // limits for LM diagonal (ceres defaults min/max: 1e-6/1e32)
  VISITABLE_META(double, min_lm_diagonal,
                 init(1e-6).range(1e-32, 1).logscale().help(
                     "Currently only affects Ceres."));
  VISITABLE_META(double, max_lm_diagonal,
                 init(1e32).range(1, 1e32).logscale().help(
                     "Currently only affects Ceres."));

  // linear solver iteration limits (ceres default min/max: 0/500)
  VISITABLE_META(
      int, min_linear_solver_iterations,
      init(0).help(
          "Minimum number of iterations for which the linear solver should "
          "run, even if the convergence criterion is satisfied (see Ceres)."));
  VISITABLE_META(int, max_linear_solver_iterations,
                 init(500).help("Maximum number of iterations for which the "
                                "linear solver should run (see Ceres)."));

  // linear solver forcing sequence (ceres default: 1e-1)
  VISITABLE_META(
      double, eta,
      init(1e-2).help(
          "Forcing sequence parameter. The truncated Newton solver uses this "
          "number to control the relative accuracy with which the Newton step "
          "is computed. This constant is passed to ConjugateGradientsSolver "
          "which uses it to terminate the iterations when (Q_i - Q_{i-1})/Q_i "
          "< eta/i (see Ceres)."));

  VISITABLE_META(
      double, r_tolerance,
      init(-1).help("Experimental, only used in power_sc and hybrid_sc"));

  VISITABLE_META(
      double, power_order,
      init(2).help("Only used in explicit_power_schur")
      );

  // jacobian scaling (ceres default: true)
  VISITABLE_META(
      bool, jacobi_scaling,
      init(true).help(
          "Use Jacobian scaling (see Ceres); note that unlike Ceres, our "
          "manual solvers consider the additional parameter "
          "jacobi_scaling_epsilon; moreover, the recompute the scale in every "
          "iteration, where ceres computes it only once in the beginning."));


  // jacobian scaling epsilon
  VISITABLE_META(
      double, jacobi_scaling_epsilon,
      init(0.0).help(
          "additional option for manual solvers: use 1/(eps + norm(diag)) to "
          "scale Jacobians; Ceres always uses eps == 1; a value 0 means "
          "'floating point epsilon' (different for float and double)."));

  // preconditioner type
  // ceres has default JACOBI, but we default to SCHUR_JACOBI
  VISITABLE_META(
      PreconditionerType, preconditioner_type,
      init(PreconditionerType::SCHUR_JACOBI)
          .help("Which preconditioner to use for PCG (see Ceres). Valid values "
                "for QR solver: JACOBI, SCHUR_JACOBI; valid values for SC "
                "solver: JACOBI, SCHUR_JACOBI; valid values for Ceres: see "
                "Ceres."));

  // termination tolerances
  // ceres defaults func/grad/param: 1e-6/1e-10/1e-8
  VISITABLE_META(double, function_tolerance,
                 init(1e-6).help("(new_cost - old_cost) < function_tolerance * "
                                 "old_cost; (see Ceres)"));
  VISITABLE_META(double, gradient_tolerance, init(0).help("only for Ceres"));
  VISITABLE_META(double, parameter_tolerance, init(0).help("only for Ceres"));

  VISITABLE_META(bool, check_gradients, init(false).help("only for Ceres"));
  VISITABLE_META(double, gradient_check_relative_precision,
                 init(1e-8).help("only for Ceres"));
  VISITABLE_META(double, gradient_check_numeric_derivative_relative_step_size,
                 init(1e-6).help("only for Ceres"));

  // ////////////////////////////////////////////////////////////
  // manual solver options
  // ////////////////////////////////////////////////////////////

  //@Simon: TODO: implement float version
  //VISITABLE_META(
  //    bool, use_double,
  //    init(true).help("if false, use float instead of double (only manual)"));


  VISITABLE_META(
      bool, cache_hessian_blocks,
      init(false).help("Precompute Jp'Jl for computing preconditioner and "
                       "right_mulitply (only implicit and factor sc solvers; "
                       "only for SCHUR_JACOBI preconditioner)"));

  VISITABLE_META(bool, jp_t_jl_on_the_fly, init(false).help("Experimental"));
  VISITABLE_META(bool, reallocate_cache, init(false).help("Experimental"));

  VISITABLE_META(
      bool, merge_factor,
      init(true).help("Try to merge the remaining non-factor landmarks into "
                      "the exisitng factors group as the last phase."));

  VISITABLE_META(
      int, power_sc_iterations,
      init(10).help("Number of inner iterations of Power Schur Complement."));

  VISITABLE_META(
      int, max_factor_size,
      init(-1).help("Maximum number of cameras of grouped factors."));

  VISITABLE_META(
      double, initial_vee,
      init(2.0).help("initial decrease factor for trust region update during "
                     "Levenberg-Marquardt; Ceres uses fixed value of 2.0."));
  VISITABLE_META(
      double, vee_factor,
      init(2.0).help("update of decrease factor for trust region update during "
                     "Levenberg-Marquardt; Ceres uses fixed value of 2.0."));

  END_VISITABLES;
};

}  // namespace rootba_povar
