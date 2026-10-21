//   Runs the inverse-design problems of cases.h through the pinned OpenNN library.
//
//   Usage (from code/):
//     bin\inverse_cases.exe problems
//         writes out/problems.json: every problem statement plus the design box of each
//         network, so the Python baselines state exactly the same problems.
//     bin\inverse_cases.exe sample <case> <seed> <starts> [constraint|band]
//         specification-only problem (no objective): draws <starts> random points in the
//         domain and applies the library's feasibility repair to each, keeping those it
//         makes feasible. -> out/sample/<case>[_band]_s<seed>.csv
//     bin\inverse_cases.exe solve <case> <dc|ga> <constraint|band|objective> <single|bi> <seed>
//                                 [iterations=10] [points=200] [sampling_rounds=50]
//         DomainContraction (dc) or GeneticResponse (ga); the specification as an equality
//         constraint, as a band around the target (the catalogue case, see cases.h), or as a
//         Fixed objective (any other formulation name; the name "objective-matched" labels
//         objective runs given a larger iterations/points setting so that they spend the
//         budget of the constraint runs, run_matched.py); the one- or two-objective task.
//         -> out/solve/<case>_<solver>_<formulation>_<task>_s<seed>.csv
//
//   Every run appends one line to out/runs_<case>.csv with the rows returned, the surrogate
//   forward passes spent (counted inside Network::calculate_outputs, see
//   opennn_pin/README.md) and the wall-clock time.

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>

#include "opennn/core/random_utilities.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/scaling_layer.h"
#include "opennn/registry.h"
#include "opennn/response_optimization/domain_contraction.h"
#include "opennn/response_optimization/genetic_response.h"

#include "cases.h"

namespace opennn { extern std::atomic<long long> rai_forward_rows; }

using namespace opennn;
using namespace rai;
namespace fs = std::filesystem;

namespace
{

// Exposes the repair step both library solvers apply to every candidate, with nothing
// around it: a random point in the constrained domain, repaired or discarded.
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
            const auto [input, output] = solve_system(calculate_random_input(domain));
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


std::vector<std::string> column_names(Network& network)
{
    std::vector<std::string> names;
    for (const auto& n : network.get_input_feature_names()) names.push_back(n);
    for (const auto& n : network.get_output_feature_names()) names.push_back(n);
    return names;
}


void write_rows(const fs::path& path, const MatrixR& rows, const std::vector<std::string>& names)
{
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    for (size_t j = 0; j < names.size(); ++j) file << names[j] << (j + 1 < names.size() ? "," : "\n");
    file << std::setprecision(9);
    for (Index i = 0; i < rows.rows(); ++i)
        for (Index j = 0; j < rows.cols(); ++j) file << rows(i, j) << (j + 1 < rows.cols() ? "," : "\n");
}


void log_run(const std::string& name, const std::string& line)
{
    const fs::path path = "out/runs_" + name + ".csv";
    const bool fresh = !fs::exists(path);
    std::ofstream file(path, std::ios::app);
    if (fresh) file << "case,mode,solver,formulation,task,seed,starts,rows,forwards,seconds,status\n";
    file << line << "\n";
}


std::string json_escape(const std::string& s)
{
    std::string out;
    for (const char c : s) out += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
    return out;
}


void write_problems()
{
    fs::create_directories("out");
    std::ofstream file("out/problems.json");
    file << "{\n";
    const std::vector<std::string> cases = {"concrete", "hea", "steel", "kernel"};
    for (size_t k = 0; k < cases.size(); ++k)
    {
        const Problem p = make_problem(cases[k]);
        Network network(fs::path("models")/(p.name + ".json"));
        const auto* scaling = static_cast<const Scaling*>(network.get_first(LayerType::Scaling));
        const VectorR lower = scaling->get_minimums(), upper = scaling->get_maximums();

        file << std::format("  \"{}\": {{\n    \"specification\": \"{}\", \"target\": {}, \"band\": {},\n",
                            p.name, p.specification, p.target, p.band);
        file << "    \"inputs\": [";
        const auto inputs = network.get_input_feature_names();
        for (size_t j = 0; j < inputs.size(); ++j) file << (j ? ", " : "") << "\"" << inputs[j] << "\"";
        file << "],\n    \"outputs\": [";
        const auto outputs = network.get_output_feature_names();
        for (size_t j = 0; j < outputs.size(); ++j) file << (j ? ", " : "") << "\"" << outputs[j] << "\"";
        file << "],\n    \"lower\": [";
        for (Index j = 0; j < lower.size(); ++j) file << (j ? ", " : "") << std::format("{:.9g}", lower(j));
        file << "],\n    \"upper\": [";
        for (Index j = 0; j < upper.size(); ++j) file << (j ? ", " : "") << std::format("{:.9g}", upper(j));
        file << "],\n    \"constraints\": [\n";
        for (size_t i = 0; i < p.constraints.size(); ++i)
        {
            const auto& c = p.constraints[i];
            file << std::format("      {{\"expression\": \"{}\", \"condition\": \"{}\", \"values\": [",
                                json_escape(c.expression), condition_name(c.condition));
            for (size_t v = 0; v < c.values.size(); ++v) file << (v ? ", " : "") << c.values[v];
            file << "]}" << (i + 1 < p.constraints.size() ? "," : "") << "\n";
        }
        const auto write_objectives = [&](const char* key, const std::vector<ObjectiveStatement>& list, bool last)
        {
            file << "    \"" << key << "\": [";
            for (size_t i = 0; i < list.size(); ++i)
                file << (i ? ", " : "") << std::format("{{\"expression\": \"{}\", \"sense\": \"{}\"}}",
                                                       json_escape(list[i].expression), sense_name(list[i].sense));
            file << "]" << (last ? "\n" : ",\n");
        };
        file << "    ],\n";
        write_objectives("single", p.single, false);
        write_objectives("bi", p.bi, true);
        file << "  }" << (k + 1 < cases.size() ? "," : "") << "\n";
    }
    file << "}\n";
    std::cout << "wrote out/problems.json\n";
}

}


int main(int argc, char* argv[])
{
    try
    {
        std::cout << std::unitbuf;
        if (argc < 2) throw std::runtime_error("usage: see the header of inverse_cases.cpp");
        const std::string mode = argv[1];

        if (mode == "problems")
        {
            write_problems();
            return 0;
        }

        const std::string name = argv[2];
        const Problem problem = make_problem(name);
        Network network(fs::path("models")/(name + ".json"));
        const std::vector<std::string> names = column_names(network);

        if (mode == "sample")
        {
            const unsigned seed = unsigned(std::stoul(argv[3]));
            const Index starts = std::stoll(argv[4]);
            const std::string statement = argc > 5 ? argv[5] : "constraint";
            set_seed(seed);

            FeasibleSampler sampler(&network);
            apply(problem, sampler,
                  statement == "band" ? Formulation::Band : Formulation::Constraint, {});

            rai_forward_rows = 0;
            const auto start = std::chrono::steady_clock::now();
            Index accepted = 0;
            const MatrixR rows = sampler.sample(starts, accepted);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

            const std::string suffix = statement == "band" ? "_band" : "";
            write_rows(fs::path("out/sample")/std::format("{}{}_s{}.csv", name, suffix, seed), rows, names);
            log_run(name, std::format("{},sample,repair,{},none,{},{},{},{},{:.3f},ok", name, statement, seed,
                                starts, rows.rows(), rai_forward_rows.load(), seconds));
            std::cout << std::format("{} sample seed {}: {}/{} feasible, {} forwards, {:.1f} s\n", name, seed,
                                     accepted, starts, rai_forward_rows.load(), seconds);
            return 0;
        }

        if (mode == "solve")
        {
            const std::string solver = argv[3], formulation_name = argv[4], task = argv[5];
            const unsigned seed = unsigned(std::stoul(argv[6]));
            const Index iterations = argc > 7 ? std::stoll(argv[7]) : 10;
            const Index points = argc > 8 ? std::stoll(argv[8]) : 200;
            const Index rounds = argc > 9 ? std::stoll(argv[9]) : 50;
            const Formulation formulation = formulation_name == "constraint" ? Formulation::Constraint
                                          : formulation_name == "band"      ? Formulation::Band
                                                                             : Formulation::Objective;
            std::vector<ObjectiveStatement> objectives = task == "single" ? problem.single : problem.bi;
            // Exploration hook: RAI_OBJECTIVES="expr|Minimize;expr|Maximize" replaces the task's objectives.
            if (const char* override_text = std::getenv("RAI_OBJECTIVES"))
            {
                objectives.clear();
                std::stringstream list(override_text);
                std::string item;
                while (std::getline(list, item, ';'))
                {
                    const auto bar = item.find('|');
                    objectives.push_back({item.substr(0, bar),
                                          item.substr(bar + 1) == "Maximize" ? Sense::Maximize : Sense::Minimize});
                }
            }
            if (objectives.empty()) throw std::runtime_error(name + " has no " + task + " task");

            set_seed(seed);
            std::unique_ptr<ResponseOptimization> optimization;
            if (solver == "dc") optimization = std::make_unique<DomainContraction>(&network);
            else optimization = std::make_unique<GeneticResponse>(&network);
            optimization->set_iterations_number(iterations);
            optimization->set_points_number(points);
            optimization->set_sampling_rounds(rounds);
            apply(problem, *optimization, formulation, objectives);

            rai_forward_rows = 0;
            const auto start = std::chrono::steady_clock::now();
            MatrixR rows;
            std::string status = "ok";
            try { rows = optimization->perform_response_optimization(); }
            catch (const std::exception& e) { status = std::string("failed: ") + e.what(); }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

            for (char& c : status) if (c == ',' || c == '\n') c = ';';
            const std::string tag = std::format("{}_{}_{}_{}_s{}", name, solver, formulation_name, task, seed);
            write_rows(fs::path("out/solve")/(tag + ".csv"), rows, names);
            log_run(name, std::format("{},solve,{},{},{},{},{},{},{},{:.3f},{}", name, solver, formulation_name, task, seed,
                                iterations*points, rows.rows(), rai_forward_rows.load(), seconds, status));
            std::cout << std::format("{}: {} rows, {} forwards, {:.1f} s, {}\n", tag, rows.rows(),
                                     rai_forward_rows.load(), seconds, status);
            return 0;
        }

        throw std::runtime_error("unknown mode " + mode);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
