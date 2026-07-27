// RIAI Case B - HEA hardness, fixed-target inverse design on the composition simplex.
//
// Surrogate: 10 atomic fractions (Al,Co,Cr,Cu,Fe,Mn,Mo,Ni,Ti,V) -> Vickers hardness HV.
// Constraint: sum of fractions = 1 (composition simplex, affine equality).
//
// FIX HV = target -> recover the feasible ALLOY-COMPOSITION manifold, then rank by
// analytic-Jacobian robustness and select via get_robust_point (centrality<->robustness).
// Dumps out/hea_manifold.csv for the parallel-coordinates / Isomap figure.

#include "opennn/neural_network.h"
#include "opennn/network_differential.h"
#include "opennn/response_optimization.h"
#include "opennn/response_constraints.h"   // ComparisonOperator
#include "opennn/random_utilities.h"       // set_seed: reproducible manuscript numbers

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace opennn;
using namespace std;

int main(int argc, char** argv)
{
    cout << unitbuf;   // flush every << so a crash localizes to the last printed line
    set_seed(1);   // fixed seed => reproducible manuscript numbers
    const filesystem::path model_path = (argc > 1) ? filesystem::path(argv[1])
                                                   : filesystem::path("models/hea_10elt.json");
    filesystem::create_directories("out");
    cout << fixed << setprecision(4);
    cout << "Loading HEA surrogate: " << model_path.string() << "\n";
    NeuralNetwork net(model_path);

    const vector<Variable> in_vars  = net.get_input_variables();
    const vector<Variable> out_vars = net.get_output_variables();
    const Index n_in = (Index)in_vars.size();
    const string hv  = out_vars.front().name;
    cout << "Inputs: " << n_in << "  Output: " << hv << "\n";

    ResponseOptimization opt(&net);
    const auto domain = opt.get_original_domain("Input");

    // reachable HV range by sampling
    const MatrixR si = opt.calculate_random_inputs(domain, 2000);
    const MatrixR so = opt.calculate_outputs(si);
    vector<float> vals((size_t)so.rows());
    for (Index i = 0; i < so.rows(); ++i) vals[(size_t)i] = so(i, 0);
    ranges::sort(vals);
    const float lo = vals.front(), hi = vals.back();
    const float HV_target = (argc > 2) ? stof(argv[2]) : vals[vals.size() * 3 / 5];  // upper-mid, reachable
    cout << "Reachable HV ~ [" << lo << ", " << hi << "], target = " << HV_target << "\n";

    // composition simplex: sum of the 10 fractions == 1
    string simplex;
    for (Index c = 0; c < n_in; ++c) simplex += (c ? " + " : "") + in_vars[(size_t)c].name;
    opt.set_formula_constraint(simplex, ComparisonOperator::EqualTo, 1.0f, 1.0f);
    cout << "Simplex constraint: " << simplex << " = 1\n";

    opt.set_objective(hv, ResponseOptimization::Sense::Fixed, HV_target);
    opt.set_relative_tolerance(1e-2f);
    // The HEA simplex has a very low feasible ratio; with the default threshold the solver
    // escalates adaptive oversampling unboundedly and crashes the pinned library. Lower the
    // ratio threshold and cap the total evaluations so oversampling stays bounded (as in baselines).
    opt.set_min_feasible_ratio(1e-4f);
    opt.set_iterations(8);
    opt.set_evaluations_number(48000);
    opt.set_max_total_evaluations(500000);

    cout << "\n=== FIXED HV = " << HV_target << " -> alloy-composition manifold ===\n";
    MatrixR M = opt.perform_response_optimization();
    if (M.rows() == 0) { cerr << "No feasible solution.\n"; return 2; }
    cout << "Manifold solutions: " << M.rows() << "\n";

    // simplex residual check
    double max_simplex_err = 0.0;
    for (Index r = 0; r < M.rows(); ++r) {
        double s = 0.0; for (Index c = 0; c < n_in; ++c) s += M(r, c);
        max_simplex_err = max(max_simplex_err, fabs(s - 1.0));
    }
    cout << "  max |sum(x)-1| over manifold = " << max_simplex_err << "\n";

    // robustness via analytic Jacobian (scalar output)
    vector<float> sens((size_t)M.rows(), 0.0f);
    NetworkDifferential diff; bool jac = true;
    try { diff.build(net); } catch (const exception& e) { jac = false; cout << "  (Jacobian unavailable: " << e.what() << ")\n"; }
    if (jac) {
        VectorR span(n_in);
        for (Index c = 0; c < n_in; ++c) span(c) = domain.superior_frontier(c) - domain.inferior_frontier(c);
        const VectorR cot = VectorR::Ones(1);
        for (Index r = 0; r < M.rows(); ++r) {
            const VectorR x = M.row(r).leftCols(n_in).transpose();
            const VectorR g = diff.vjp(x, cot);
            float s = 0.0f; for (Index c = 0; c < n_in; ++c) s += (g(c) * span(c)) * (g(c) * span(c));
            sens[(size_t)r] = sqrt(s);
        }
        Index mr = 0; for (Index r = 1; r < M.rows(); ++r) if (sens[(size_t)r] < sens[(size_t)mr]) mr = r;
        cout << "  most-robust alloy (row " << mr << ", sens " << sens[(size_t)mr] << "): x=[";
        for (Index c = 0; c < n_in; ++c) cout << (c ? ", " : "") << M(mr, c);
        cout << "] -> HV=" << M(mr, n_in) << "\n";
    }

    // per-input diversity (spread / domain span)
    cout << "\n--- diversity per fraction (range / domain) ---\n";
    for (Index c = 0; c < n_in; ++c) {
        float mn = M(0, c), mx = M(0, c);
        for (Index r = 0; r < M.rows(); ++r) { mn = min(mn, M(r, c)); mx = max(mx, M(r, c)); }
        const float sp = domain.superior_frontier(c) - domain.inferior_frontier(c);
        cout << "  " << left << setw(4) << in_vars[(size_t)c].name << right
             << "  [" << mn << ", " << mx << "]  fill=" << (sp > 1e-9f ? (mx - mn) / sp * 100.0f : 0.0f) << "%\n";
    }

    // selection dial. wallMargin = worst-case normalized distance to the nearest domain wall
    // (1=centre, 0=on a wall): low margin => point at the edge of the training support, where
    // the surrogate extrapolates/saturates. Prefer a more central point (lower balance) to hedge.
    auto wall_margin = [&](const VectorR& row) -> float {
        float worst = 1.0f;
        for (Index c = 0; c < n_in; ++c) {
            const float half = 0.5f * (domain.superior_frontier(c) - domain.inferior_frontier(c));
            if (half < 1e-9f) continue;
            const float m = min(row(c) - domain.inferior_frontier(c),
                                domain.superior_frontier(c) - row(c)) / half;
            worst = min(worst, max(0.0f, min(1.0f, m)));
        }
        return worst;
    };
    cout << "\n--- get_robust_point(front, balance) ---\n";
    for (const float b : { 0.0f, 0.5f, 1.0f }) {
        const auto [idx, row] = opt.get_robust_point(M, b);
        cout << "  balance=" << b << " -> row " << idx << "  HV=" << row(n_in)
             << "  sens=" << (jac ? sens[(size_t)idx] : 0.0f)
             << "  wallMargin=" << wall_margin(row) << "\n";
    }

    // dump
    {
        ofstream csv("out/hea_manifold.csv");
        for (Index c = 0; c < n_in; ++c) csv << in_vars[(size_t)c].name << ",";
        csv << hv << ",sensitivity\n";
        for (Index r = 0; r < M.rows(); ++r) {
            for (Index c = 0; c < n_in; ++c) csv << M(r, c) << ",";
            csv << M(r, n_in) << "," << sens[(size_t)r] << "\n";
        }
        cout << "\n  wrote out/hea_manifold.csv (" << M.rows() << " rows)\n";
    }

    cout << "Case B done.\n";
    return 0;
}
