//   Trains the surrogate network of one case with the pinned OpenNN.
//
//   Usage (from code/):  bin\train_surrogates.exe <concrete|hea|steel|cnc> [neurons,list] [seeds]
//
//   Roles come from data/<case>_split.csv (grouped a-priori split, prep_data.py). For every
//   hidden-layer size in the list and every seed the network is trained with the OpenNN
//   QuasiNewton optimizer on the mean squared error (RAI_OPTIMIZER overrides it) plus an L2 penalty and early stopping on the
//   validation set. The (size, seed) pair with the highest mean validation R2 is kept.
//   R2 is the true coefficient of determination, 1 - SSres/SStot, per target.
//
//   After selection the design box is set to the full-table range of each input (the
//   scaling layer otherwise stores the training-split range, and the response optimizer
//   takes its box from there), and the output clamp to the full-table range of each
//   target. The forward pass is unchanged by the first (mean/std scaling) and only bounds
//   the prediction by the second.
//
//   Writes models/<case>.json (+ .bin), models/<case>_surrogate.py (OpenNN Python export),
//   models/<case>_predictions.csv (all rows, for the Python cross-check) and
//   out/surrogate_<case>.txt / .csv (selection log and metrics).

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>

#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/evaluation/evaluation.h"
#include "opennn/models/models.h"
#include "opennn/network/layers/clamping_layer.h"
#include "opennn/network/layers/scaling_layer.h"
#include "opennn/network/model_expression.h"
#include "opennn/registry.h"
#include "opennn/training/training.h"

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


std::vector<SampleRole> read_roles(const fs::path& path, Index rows)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    std::string line;
    std::getline(file, line);
    std::vector<SampleRole> roles(size_t(rows), SampleRole::None);
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const auto tokens = split(line, ',');
        const Index row = std::stoll(tokens.at(0)) - 2;
        const std::string role = tokens.at(1).substr(0, tokens.at(1).find_last_not_of("\r\n ") + 1);
        roles.at(size_t(row)) = role == "Training" ? SampleRole::Training
                              : role == "Validation" ? SampleRole::Validation : SampleRole::Testing;
    }
    return roles;
}


VectorR r2_per_target(const MatrixR& targets, const MatrixR& outputs)
{
    VectorR r2(targets.cols());
    for (Index j = 0; j < targets.cols(); ++j)
    {
        const double mean = double(targets.col(j).mean());
        double sse = 0.0, sst = 0.0;
        for (Index i = 0; i < targets.rows(); ++i)
        {
            sse += std::pow(double(outputs(i, j)) - double(targets(i, j)), 2);
            sst += std::pow(double(targets(i, j)) - mean, 2);
        }
        r2(j) = float(sst > 0.0 ? 1.0 - sse/sst : NAN);
    }
    return r2;
}


struct Fit
{
    Index neurons = 0;
    unsigned seed = 0;
    Index epochs = 0;
    VectorR train_r2, validation_r2, test_r2;
    float mean_validation() const { return validation_r2.mean(); }
};


std::string join(const VectorR& v)
{
    std::string s;
    for (Index j = 0; j < v.size(); ++j) s += std::format("{}{:.4f}", j ? ";" : "", v(j));
    return s;
}


std::unique_ptr<TabularDataset> load_dataset(const std::string& name)
{
    auto owned = std::make_unique<TabularDataset>(fs::path("data")/(name + ".csv"), ",", true, false);
    TabularDataset& dataset = *owned;

    // Targets are the last columns of data/<case>.csv (prep_data.py).
    const Index targets_number = name == "cnc" || name == "concrete" ? 2 : name == "hea" ? 1 : name == "steel" ? 4 : -1;
    if (targets_number < 0) throw std::runtime_error("unknown case " + name);

    const Index variables_number = ssize(dataset.get_variables());
    for (Index i = 0; i < variables_number; ++i)
        dataset.set_variable_role(i, i < variables_number - targets_number ? VariableRole::Input : VariableRole::Target);

    dataset.set_variable_scalers("MeanStandardDeviation");
    dataset.set_shape(VariableRole::Input, {dataset.get_features_number(VariableRole::Input)});
    dataset.set_shape(VariableRole::Target, {dataset.get_features_number(VariableRole::Target)});

    const auto roles = read_roles(fs::path("data")/(name + "_split.csv"), dataset.get_samples_number());
    for (Index i = 0; i < ssize(roles); ++i) dataset.set_sample_role(i, roles[size_t(i)]);

    return owned;
}


Fit train_one(TabularDataset& dataset, std::unique_ptr<ApproximationNetwork>& owned, Index neurons, unsigned seed)
{
    set_seed(seed);
    owned = std::make_unique<ApproximationNetwork>(dataset.get_input_shape(), Shape{neurons}, dataset.get_target_shape());
    ApproximationNetwork& network = *owned;

    Training training(&network, &dataset);
    if (const char* optimizer = std::getenv("RAI_OPTIMIZER")) training.set_optimization_algorithm(optimizer);
    else training.set_optimization_algorithm("QuasiNewton");
    training.get_optimization_algorithm()->set_display(false);
    training.get_loss()->set_regularization("L2");
    training.get_loss()->set_regularization_weight(1e-3f);
    training.get_optimization_algorithm()->set_maximum_validation_failures(500);
    training.get_optimization_algorithm()->set_maximum_epochs(5000);
    const TrainingResult result = training.train();

    const Evaluation evaluation(&network, &dataset);
    Fit fit{neurons, seed};
    fit.epochs = result.get_epochs_number();
    for (const auto& [role, out] : {std::pair{"Training", &fit.train_r2},
                                    std::pair{"Validation", &fit.validation_r2},
                                    std::pair{"Testing", &fit.test_r2}})
    {
        const auto [targets, outputs] = evaluation.get_targets_and_outputs(role);
        *out = r2_per_target(targets, outputs);
    }
    return fit;
}

}


int main(int argc, char* argv[])
{
    try
    {
        if (argc < 2) throw std::runtime_error("usage: train_surrogates <concrete|hea|steel|cnc> [neurons,list] [seeds]");
        const std::string name = argv[1];
        std::vector<Index> grid;
        for (const auto& t : split(argc > 2 ? argv[2] : "2,4,8,16,32", ',')) grid.push_back(std::stoll(t));
        const unsigned seeds = argc > 3 ? unsigned(std::stoul(argv[3])) : 5;

        fs::create_directories("models");
        fs::create_directories("out");

        const auto owned_dataset = load_dataset(name);
        TabularDataset& dataset = *owned_dataset;
        const Index inputs_number = dataset.get_features_number(VariableRole::Input);
        const Index targets_number = dataset.get_features_number(VariableRole::Target);

        std::ofstream log(fs::path("out")/("surrogate_" + name + ".csv"));
        log << "neurons,seed,train_r2,validation_r2,test_r2,mean_validation_r2\n";

        Fit best;
        best.validation_r2 = VectorR::Constant(targets_number, -1e9f);

        for (const Index neurons : grid)
            for (unsigned seed = 1; seed <= seeds; ++seed)
            {
                std::unique_ptr<ApproximationNetwork> trial;
                const Fit fit = train_one(dataset, trial, neurons, seed);
                log << std::format("{},{},{},{},{},{:.4f}\n", neurons, seed, join(fit.train_r2),
                                   join(fit.validation_r2), join(fit.test_r2), fit.mean_validation());
                log.flush();
                std::cout << std::format("{} n={:2d} s={} epochs {} val {} test {}\n", name, neurons, seed, fit.epochs,
                                         join(fit.validation_r2), join(fit.test_r2)) << std::flush;
                if (fit.mean_validation() > best.mean_validation()) best = fit;
            }

        // Retrain the selected pair (deterministic given the seed) and finish the model.
        std::unique_ptr<ApproximationNetwork> owned_network;
        const Fit final_fit = train_one(dataset, owned_network, best.neurons, best.seed);
        ApproximationNetwork& network = *owned_network;

        const MatrixR data = dataset.get_data();
        auto* scaling = static_cast<Scaling*>(network.get_first(LayerType::Scaling));
        std::vector<Descriptives> descriptives = scaling->get_descriptives();
        for (Index j = 0; j < inputs_number; ++j)
        {
            descriptives[size_t(j)].minimum = data.col(j).minCoeff();
            descriptives[size_t(j)].maximum = data.col(j).maxCoeff();
        }
        scaling->set_descriptives(descriptives);

        if (auto* clamping = dynamic_cast<Clamping*>(network.get_layer("clamping_layer").get()))
            for (Index j = 0; j < targets_number; ++j)
            {
                clamping->set_lower_bound(j, data.col(inputs_number + j).minCoeff());
                clamping->set_upper_bound(j, data.col(inputs_number + j).maxCoeff());
            }

        network.set_input_names(dataset.get_feature_names(VariableRole::Input));
        network.set_output_names(dataset.get_feature_names(VariableRole::Target));

        const Evaluation evaluation(&network, &dataset);
        Fit clamped{best.neurons, best.seed};
        for (const auto& [role, out] : {std::pair{"Training", &clamped.train_r2},
                                        std::pair{"Validation", &clamped.validation_r2},
                                        std::pair{"Testing", &clamped.test_r2}})
        {
            const auto [targets, outputs] = evaluation.get_targets_and_outputs(role);
            *out = r2_per_target(targets, outputs);
        }

        network.save(fs::path("models")/(name + ".json"));
        ModelExpression(&network).save(fs::path("models")/(name + "_surrogate.py"),
                                       ModelExpression::ProgrammingLanguage::Python);

        const MatrixR inputs = data.leftCols(inputs_number);
        const MatrixR outputs = network.calculate_outputs(inputs);
        std::ofstream predictions(fs::path("models")/(name + "_predictions.csv"));
        predictions << std::setprecision(9);
        for (Index i = 0; i < inputs.rows(); ++i)
        {
            for (Index j = 0; j < inputs.cols(); ++j) predictions << inputs(i, j) << ",";
            for (Index j = 0; j < outputs.cols(); ++j) predictions << outputs(i, j) << (j + 1 < outputs.cols() ? "," : "\n");
        }

        std::ofstream report(fs::path("out")/("surrogate_" + name + ".txt"));
        report << std::format("case {}: {} inputs -> {} targets; grid {}; seeds 1-{}\n", name, inputs_number,
                              targets_number, argc > 2 ? argv[2] : "2,4,8,16,32", seeds)
               << std::format("selected: {} hidden neurons, seed {} (mean validation R2 {:.4f})\n",
                              best.neurons, best.seed, best.mean_validation())
               << "retrained R2 train/validation/test: " << join(final_fit.train_r2) << " | "
               << join(final_fit.validation_r2) << " | " << join(final_fit.test_r2) << "\n"
               << "after full-range box + clamp, R2 train/validation/test: " << join(clamped.train_r2) << " | "
               << join(clamped.validation_r2) << " | " << join(clamped.test_r2) << "\n";
        std::cout << "\n" << std::ifstream(fs::path("out")/("surrogate_" + name + ".txt")).rdbuf();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
