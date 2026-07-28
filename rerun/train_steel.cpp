// RIAI Case C - train the low-alloy steel surrogate.
//
//   data/steel_multi_fulldata.csv : 14 inputs -> 4 outputs
//     inputs  : C Si Mn P S Ni Cr Mo Cu V Al N Nb_Ta temperature_C
//     outputs : proof_stress_MPa tensile_strength_MPa elongation_pct reduction_area_pct
//
// Mirrors the benchmark_datasets baseline_trainer recipe (TabularDataset ->
// ApproximationNetwork -> TrainingStrategy -> save JSON+BIN). Seeded for a
// reproducible net. Reports per-output test R^2 (calculate_determination).
//
// Writes models/steel_multi.json (+ sibling .bin with the weights).

#include "opennn/tabular_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/bounding_layer.h"
#include "opennn/neural_network.h"
#include "opennn/training_strategy.h"
#include "opennn/testing_analysis.h"
#include "opennn/variable.h"
#include "opennn/random_utilities.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace opennn;
using namespace std;

int main(int argc, char** argv)
{
    set_seed(1);   // reproducible surrogate

    const filesystem::path csv = (argc > 1) ? filesystem::path(argv[1])
                                            : filesystem::path("data/steel_multi_fulldata.csv");
    const int hidden = (argc > 2) ? stoi(argv[2]) : 32;
    const int epochs = (argc > 3) ? stoi(argv[3]) : 2000;

    cout << unitbuf << fixed << setprecision(4);
    cout << "Training steel surrogate from " << csv.string()
         << "  (hidden=" << hidden << ", epochs=" << epochs << ")\n";

    TabularDataset dataset(csv, ",", /*has_header=*/true, /*text=*/false);

    const vector<string> names = {
        "C","Si","Mn","P","S","Ni","Cr","Mo","Cu","V","Al","N","Nb_Ta","temperature_C",
        "proof_stress_MPa","tensile_strength_MPa","elongation_pct","reduction_area_pct"
    };
    const Index n_in = 14, n_out = 4;

    for (Index i = 0; i < n_in; ++i) {
        dataset.set_variable_role(i, "Input");
        dataset.set_variable_type(i, VariableType::Numeric);
    }
    for (Index i = 0; i < n_out; ++i) {
        dataset.set_variable_role(n_in + i, "Target");
        dataset.set_variable_type(n_in + i, VariableType::Numeric);
    }

    dataset.impute_missing_values_statistic(TabularDataset::MissingValuesMethod::Mean);
    dataset.set_shape("Input",  { dataset.get_features_number("Input")  });
    dataset.set_shape("Target", { dataset.get_features_number("Target") });
    dataset.set_default_variable_scalers();
    dataset.split_samples_random(0.6f, 0.2f, 0.2f);

    cout << "Samples: " << dataset.get_samples_number()
         << "  inputs: "  << dataset.get_features_number("Input")
         << "  targets: " << dataset.get_features_number("Target") << "\n";

    Shape hidden_shape; hidden_shape.push_back(hidden);
    ApproximationNetwork net(dataset.get_input_shape(), hidden_shape, dataset.get_target_shape());
    if (Bounding* b = dynamic_cast<Bounding*>(net.get_first("Bounding"))) b->set_bounding_method("NoBounding");

    TrainingStrategy strategy(&net, &dataset);
    strategy.get_optimization_algorithm()->set_display(true);
    strategy.get_optimization_algorithm()->set_maximum_epochs(epochs);
    strategy.train();

    // per-output test R^2
    TestingAnalysis ta(&net, &dataset);
    const MatrixR X_te = dataset.get_data("Testing", "Input");
    const MatrixR T_te = dataset.get_data("Testing", "Target");
    const MatrixR O_te = net.calculate_outputs(X_te);
    cout << "\nTest R^2 per output (n_test=" << T_te.rows() << "):\n";
    for (Index j = 0; j < n_out; ++j) {
        VectorR t(T_te.rows()), o(O_te.rows());
        for (Index i = 0; i < T_te.rows(); ++i) { t(i) = T_te(i, j); o(i) = O_te(i, j); }
        const float r2 = ta.calculate_determination(o, t);
        cout << "  " << left << setw(22) << names[(size_t)(n_in + j)] << " R^2 = " << r2 << "\n";
    }

    filesystem::create_directories("models");
    net.save("models/steel_multi.json");
    cout << "\nSaved models/steel_multi.json (+ .bin)\n";
    return 0;
}
