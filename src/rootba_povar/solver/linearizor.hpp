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

#include "rootba_povar/bal/bal_problem.hpp"
#include "rootba_povar/bal/residual_info.hpp"
#include "rootba_povar/bal/solver_options.hpp"
#include "rootba_povar/solver/solver_summary.hpp"

namespace rootba_povar {

// TODO: can we find a better name than Linearizor? Linearization is already
// taken for the more low-level API in LinearizationQR and LinearizationSC.
template <typename Scalar_>
class Linearizor {
 public:
  using Scalar = Scalar_;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

 public:
  // factory method
  static std::unique_ptr<Linearizor<Scalar>> create(
      BalProblem<Scalar>& bal_problem, const SolverOptions& options,
      SolverSummary* summary = nullptr);
  static std::unique_ptr<Linearizor<Scalar>> create_homogeneous(
            BalProblem<Scalar>& bal_problem, const SolverOptions& options,
            SolverSummary* summary = nullptr);

  virtual ~Linearizor() = default;

  virtual void start_iteration(IterationSummary* it_summary = nullptr) = 0;

  virtual void compute_error_pOSE(ResidualInfo& ri, bool initialization_varproj) = 0;
  virtual void compute_error_homogeneous(ResidualInfo& ri, bool initialization_varproj) = 0;

  virtual void initialize_varproj_lm_pOSE(Scalar alpha, bool initialization_varproj) = 0;

  virtual void linearize_pOSE(Scalar alpha) = 0;
  virtual void linearize_projective_space_homogeneous() = 0;

  virtual VecX solve_joint(Scalar lambda, Scalar relative_error_change) = 0;
  virtual  VecX solve(const SolverOptions& solver_options, Scalar lambda, Scalar relative_error_change) = 0;

  virtual Scalar apply_joint(VecX&& inc) = 0;
  virtual  Scalar apply(const SolverOptions& solver_options, Scalar alpha, VecX&& inc) = 0;

  virtual void finish_iteration() = 0;
};

}  // namespace rootba_povar
