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

#include <glog/logging.h>

#include "rootba_povar/bal/ba_log_utils.hpp"
#include "rootba_povar/bal/bal_app_options.hpp"
#include "rootba_povar/cli/bal_cli_utils.hpp"
#include "rootba_povar/solver/bal_bundle_adjustment.hpp"

int main(int argc, char** argv) {
    using namespace rootba_povar;

    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    // parse cli and load config
    BalAppOptions options;
    if (!parse_bal_app_arguments(
            "Solve BAL problem with solver determined by config.", argc, argv,
            options)) {
        return 1;
    }

    // print options
    if (options.solver.verbosity_level >= 2) {
        LOG(INFO) << "Options:\n" << options;
    }

//  if (options.solver.solver_type == SolverOptions::SolverType::CERES) {
//    DatasetSummary dataset_summary;
//    PipelineTimingSummary timing_summary;
//    BaLog log;
//
//    // load data
//    BalProblem<double> bal_problem = load_normalized_bal_problem<double>(
//        options.dataset, &dataset_summary, &timing_summary);
//    log_summary(log.static_data.timing, timing_summary);
//
//    // run ceres solver (also updates `static_.timing`)
//    bundle_adjust_ceres(bal_problem, options.solver, &log);
//
//    // log summary (`static_.solver`, `static_.timing` and `iterations` are
//    // already filled)
//    log_summary(log.static_data.problem_info, dataset_summary);
//    log.save_json(options.solver.log);
//  } else {
    BalPipelineSummary summary;
    BalPipelineSummary summary_tmp;

    // load dataset
    auto bal_problem = load_normalized_bal_problem<double>(
            options.dataset, &summary.dataset, &summary.timing);

    // run solver
    bundle_adjust_manual(bal_problem, options.solver, &summary.solver,
                         &summary.timing, &summary_tmp.solver);

    // postprocess
    bal_problem.postprocress(options.dataset, &summary.timing);

    // log summary
    BaLog log;
    log_summary(log, summary);
    log.save_json(options.solver.log);
//   }

  return 0;
}
