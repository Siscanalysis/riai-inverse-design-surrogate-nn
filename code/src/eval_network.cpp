//   Evaluates a saved network on a case's data and split, and exports it for the baselines.
//
//   Usage (from code/):  bin\eval_network.exe <concrete|steel|kernel> <model.json>
//
//   Loads the network with the pinned OpenNN, reads data/<case>.csv and data/<case>_split.csv,
//   and reports, per output and per subset, the true coefficient of determination
//   (1 - SSres/SStot) and the squared correlation, which are not the same number: OpenNN's
//   own "determination" is the second one. Writes the Python export and a predictions file
//   beside the model, so surrogates.py can drive exactly this network.

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/network/network.h"
#include "opennn/network/model_expression.h"

using namespace opennn;
namespace fs = std::filesystem;

namespace
{

std::vector<std::string> split(const std::string& text, char separator)
{
    std::vector<std::string> tokens;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, separator)) tokens.push_back(token);
    return tokens;
}


std::vector<std::string> read_roles(const fs::path& path, Index rows)
{
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    std::vector<std::string> roles(size_t(rows), "None");
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const auto tokens = split(line, ',');
        const Index row = std::stoll(tokens.at(0)) - 2;
        std::string role = tokens.at(1);
        while (!role.empty() && (role.back() == '\r' || role.back() == '\n' || role.back() == ' ')) role.pop_back();
        roles.at(size_t(row)) = role;
    }
    return roles;
}


std::pair<double, double> fit(const std::vector<double>& target, const std::vector<double>& output)
{
    const Index n = Index(target.size());
    double target_mean = 0.0, output_mean = 0.0;
    for (Index i = 0; i < n; ++i) { target_mean += target[size_t(i)]; output_mean += output[size_t(i)]; }
    target_mean /= double(n); output_mean /= double(n);

    double sse = 0.0, sst = 0.0, covariance = 0.0, target_variance = 0.0, output_variance = 0.0;
    for (Index i = 0; i < n; ++i)
    {
        const double t = target[size_t(i)], o = output[size_t(i)];
        sse += (o - t)*(o - t);
        sst += (t - target_mean)*(t - target_mean);
        covariance += (t - target_mean)*(o - output_mean);
        target_variance += (t - target_mean)*(t - target_mean);
        output_variance += (o - output_mean)*(o - output_mean);
    }
    const double r = covariance/std::sqrt(target_variance*output_variance);
    return {sst > 0.0 ? 1.0 - sse/sst : NAN, r*r};
}

}


int main(int argc, char* argv[])
{
    try
    {
        if (argc < 3) throw std::runtime_error("usage: eval_network <case> <model.json>");
        const std::string name = argv[1];
        const fs::path model_path = argv[2];

        Network network(model_path);

        TabularDataset dataset(fs::path("data")/(name + ".csv"), ",", true, false);
        const MatrixR data = dataset.get_data();
        const Index targets_number = (name == "concrete" || name == "kernel") ? 2 : 4;
        const Index inputs_number = data.cols() - targets_number;

        const auto roles = read_roles(fs::path("data")/(name + "_split.csv"), data.rows());
        const MatrixR inputs = data.leftCols(inputs_number);
        const MatrixR outputs = network.calculate_outputs(inputs);

        std::cout << std::format("{}: {} inputs -> {} outputs, {} samples\n",
                                 name, inputs_number, targets_number, data.rows());

        for (const std::string& subset : {"Training", "Validation", "Testing"})
        {
            for (Index j = 0; j < targets_number; ++j)
            {
                std::vector<double> target, output;
                for (Index i = 0; i < data.rows(); ++i)
                    if (roles[size_t(i)] == subset)
                    {
                        target.push_back(double(data(i, inputs_number + j)));
                        output.push_back(double(outputs(i, j)));
                    }
                if (target.empty()) continue;
                const auto [determination, squared_correlation] = fit(target, output);
                std::cout << std::format("  {:10} output {}: true R2 {:.4f}   squared correlation {:.4f}   (n = {})\n",
                                         subset, j, determination, squared_correlation, target.size());
            }
        }

        const fs::path stem = model_path.parent_path()/model_path.stem();
        ModelExpression(&network).save(stem.string() + "_surrogate.py", ModelExpression::ProgrammingLanguage::Python);

        std::ofstream predictions(stem.string() + "_predictions.csv");
        predictions << std::setprecision(9);
        for (Index i = 0; i < inputs.rows(); ++i)
        {
            for (Index j = 0; j < inputs.cols(); ++j) predictions << inputs(i, j) << ",";
            for (Index j = 0; j < outputs.cols(); ++j)
                predictions << outputs(i, j) << (j + 1 < outputs.cols() ? "," : "\n");
        }
        std::cout << "wrote " << stem.string() << "_surrogate.py and _predictions.csv\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
