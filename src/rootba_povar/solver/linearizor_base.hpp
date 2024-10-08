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

#include "rootba_povar/cg/conjugate_gradient.hpp"
#include "rootba_povar/solver/linearizor.hpp"

namespace rootba_povar {

template <typename Scalar_>
class LinearizorBase : public Linearizor<Scalar_> {
 public:
  using Scalar = Scalar_;
  using Base = Linearizor<Scalar>;

  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

 public:  // public interface
  LinearizorBase(BalProblem<Scalar>& bal_problem, const SolverOptions& options,
                 SolverSummary* summary = nullptr);

  void start_iteration(IterationSummary* it_summary = nullptr) override;


    void compute_error_pOSE(ResidualInfo& ri, bool initialization_varproj) override;
    void compute_error_homogeneous(ResidualInfo& ri, bool initialization_varproj) override;

    void initialize_varproj_lm_pOSE(Scalar_ alpha, bool initialization_varproj) override;

    void finish_iteration() override;

 protected:  // helpers
    Scalar get_effective_jacobi_scaling_epsilon();

  // solve H(-x) = b with PCG
  typename ConjugateGradientsSolver<Scalar>::Summary pcg(
      const LinearOperator<Scalar>& H_pp, const VecX& b_p,
      std::unique_ptr<Preconditioner<Scalar>>&& preconditioner, VecX& xref);

    typename ConjugateGradientsSolver<Scalar>::Summary pcg_joint(
            const LinearOperator<Scalar>& H_pp, const VecX& b_p,
            std::unique_ptr<Preconditioner<Scalar>>&& preconditioner, VecX& xref);

 protected:
  SolverOptions options_;
  BalProblem<Scalar>& bal_problem_;

  // summary pointers are optional (e.g. for unit testing we can skip them)
  SolverSummary* summary_ = nullptr;
  IterationSummary* it_summary_ = nullptr;
};

}  // namespace rootba_povar
