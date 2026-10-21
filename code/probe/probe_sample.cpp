//   Probe: does the analytical network Jacobian of the newer OpenNN change the feasibility
//   repair, or only its cost?
//
//   The probe runs exactly the specification-only task of inverse_cases.cpp (draw random points
//   in the constrained domain, repair each, keep those made feasible), against the newer library
//   instead of the pinned one, in two arms of the SAME binary:
//
//     arm "fd"         RAI_FORCE_NUMERICAL_JACOBIAN=1, the finite-difference Jacobian of the pin
//     arm "analytical" the analytical network Jacobian added in 26347f3e7
//
//   One binary and one switch, so the two arms cannot differ in compiler, flags, machine or
//   library version: they differ only in how the constraint Jacobian is formed. The pinned
//   build is the third arm, already measured in out/runs_<case>.csv, and it also carries the
//   other changes of the interval, so it answers a different question.
//
//   The repair settings are made explicit rather than left to the defaults, because the default
//   feasibility_rounds changed from 3 (pin) to 1 between the two versions: left alone, the arms
//   would differ in budget and not in derivative.
//
//   Cost is reported in three raw counts and never in one pre-weighted total, so that any later
//   weighting can be applied without running anything again:
//     forwards    inference rows through Network::calculate_outputs (the paper's unit)
//     jac_calls   analytical Jacobians formed, which never reach calculate_outputs
//     jac_sweeps  layer sweeps those cost, charged as 1 recomputation + 1 per network output
//
//   Usage (from code/):  probe\bin\probe_sample.exe <case> <seed> <starts> [constraint|band]
//   Env: RAI_FORCE_NUMERICAL_JACOBIAN (set: fd arm), RAI_FEASIBILITY_ROUNDS (default 3).

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>

#include "opennn/core/random_utilities.h"
#include "opennn/network/network.h"

#include "cases.h"

namespace opennn
{
extern std::atomic<long long> rai_forward_rows;
extern std::atomic<long long> rai_jacobian_calls;
extern std::atomic<long long> rai_jacobian_sweeps;
}

using namespace opennn;
using namespace rai;
namespace fs = std::filesystem;

namespace
{

// The same exposure of the repair step as inverse_cases.cpp, on the newer protected name
// (solve_system became get_feasible_points).
class FeasibleSampler : public ResponseOptimization
{
public:
    using ResponseOptimization::ResponseOptimization;

    MatrixR sample(Index starts, Index& accepted)
    {
        const pair<VectorR, VectorR> domain = calculate_domain();
        MatrixR rows(starts, domain.first.size() + network->get_outputs_number());
        accepted = 0;
        for (Index i = 0; i < starts; ++i)
        {
            const auto [input, output] = get_feasible_points(calculate_random_input(domain));
            if (input.size() == 0) continue;
            rows.row(accepted).head(input.size()) = input.transpose();
            rows.row(accepted).tail(output.size()) = output.transpose();
            ++accepted;
        }
        rows.conservativeResize(accepted, Eigen::NoChange);
        return rows;
    }

private:
    MatrixR single_optimization() override { return {}; }
    MatrixR multi_optimization() override { return {}; }
};


void write_rows(const fs::path& path, const MatrixR& rows, Network& network)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    std::vector<std::string> names;
    for (const auto& n : network.get_input_feature_names()) names.push_back(n);
    for (const auto& n : network.get_output_feature_names()) names.push_back(n);
    for (size_t j = 0; j < names.size(); ++j) file << names[j] << (j + 1 < names.size() ? "," : "\n");
    file << std::setprecision(9);
    for (Index i = 0; i < rows.rows(); ++i)
        for (Index j = 0; j < rows.cols(); ++j) file << rows(i, j) << (j + 1 < rows.cols() ? "," : "\n");
}

}


int main(int argc, char* argv[])
{
    try
    {
        std::cout << std::unitbuf;
        if (argc < 4) throw std::runtime_error("usage: probe_sample <case> <seed> <starts> [statement]");

        const std::string name = argv[1];
        const unsigned seed = unsigned(std::stoul(argv[2]));
        const Index starts = std::stoll(argv[3]);
        const std::string statement = argc > 4 ? argv[4] : "constraint";

        const bool numerical = std::getenv("RAI_FORCE_NUMERICAL_JACOBIAN") != nullptr;
        const char* rounds_text = std::getenv("RAI_FEASIBILITY_ROUNDS");
        const Index rounds = rounds_text ? std::stoll(rounds_text) : 3;
        const std::string arm = numerical ? "fd" : "analytical";

        const Problem problem = make_problem(name);
        Network network(fs::path("models")/(name + ".json"));

        set_seed(seed);
        FeasibleSampler sampler(&network);
        sampler.set_feasibility_rounds(rounds);
        sampler.set_feasibility_evaluations(50);
        apply(problem, sampler, statement == "band" ? Formulation::Band : Formulation::Constraint, {});

        rai_forward_rows = 0;
        rai_jacobian_calls = 0;
        rai_jacobian_sweeps = 0;

        const auto start = std::chrono::steady_clock::now();
        Index accepted = 0;
        const MatrixR rows = sampler.sample(starts, accepted);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        write_rows(fs::path("out/probe")/std::format("{}_{}_s{}.csv", name, arm, seed), rows, network);

        const fs::path log = "out/probe/runs.csv";
        const bool fresh = !fs::exists(log);
        std::ofstream file(log, std::ios::app);
        if (fresh) file << "case,arm,statement,seed,starts,rows,forwards,jac_calls,jac_sweeps,rounds,seconds\n";
        file << std::format("{},{},{},{},{},{},{},{},{},{},{:.3f}\n", name, arm, statement, seed, starts,
                            rows.rows(), rai_forward_rows.load(), rai_jacobian_calls.load(),
                            rai_jacobian_sweeps.load(), rounds, seconds);

        std::cout << std::format("{} {} seed {}: {}/{} feasible ({:.1f}%), {} forwards, "
                                 "{} jacobians ({} sweeps), {:.1f} s\n",
                                 name, arm, seed, accepted, starts, 100.0*double(accepted)/double(starts),
                                 rai_forward_rows.load(), rai_jacobian_calls.load(),
                                 rai_jacobian_sweeps.load(), seconds);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
