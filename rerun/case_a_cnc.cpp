// RIAI Case A - CNC turning AISI 304, mixed-sense response optimization.
//
// Surrogate: (cutting_speed vc, feed f, depth_of_cut ap, tool_wear_VB) -> (Ra, F).
// MRR = vc * f * ap is analytic (exact).
//
// Demonstrates, on ONE problem:
//   * FIXED  (equality)  target Ra = spec  -> recover the feasible recipe MANIFOLD
//   * MINIMIZE F (cutting force)           -> objective along the manifold
//   * MAXIMIZE MRR (productivity)          -> competing objective (analytic)
//   * robustness (analytic df/dx) + centrality selection via get_robust_point
//   * MULTI-TARGET: FIX Ra & FIX F -> intersection sub-manifold
//
// Outputs CSVs into out/ for the manuscript figures.

#include "opennn/neural_network.h"
#include "opennn/network_differential.h"
#include "opennn/response_optimization.h"

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

static Index find_output(const vector<Variable>& outs, const string& name)
{
    for (Index i = 0; i < (Index)outs.size(); ++i)
        if (outs[(size_t)i].name == name) return i;
    return -1;
}

// span-weighted robustness of a chosen output (cotangent picks the output row).
static vector<float> robustness(NeuralNetwork& net, const MatrixR& results,
                                Index n_in, Index n_out, Index out_col,
                                const ResponseOptimization::Domain& domain)
{
    vector<float> sens((size_t)results.rows(), 0.0f);
    NetworkDifferential diff;
    try { diff.build(net); }
    catch (const exception& e) { cout << "  (analytic Jacobian unavailable: " << e.what() << ")\n"; return sens; }

    VectorR span(n_in);
    for (Index c = 0; c < n_in; ++c) span(c) = domain.superior_frontier(c) - domain.inferior_frontier(c);

    VectorR cotangent = VectorR::Zero(n_out);
    cotangent(out_col) = 1.0f;

    for (Index r = 0; r < results.rows(); ++r)
    {
        const VectorR x = results.row(r).leftCols(n_in).transpose();
        const VectorR g = diff.vjp(x, cotangent);
        float s = 0.0f;
        for (Index c = 0; c < n_in; ++c) s += (g(c) * span(c)) * (g(c) * span(c));
        sens[(size_t)r] = sqrt(s);
    }
    return sens;
}

int main(int argc, char** argv)
{
    const filesystem::path model_path = (argc > 1) ? filesystem::path(argv[1])
                                                   : filesystem::path("models/cnc_multi.json");
    const float Ra_target = (argc > 2) ? stof(argv[2]) : 1.6f;   // N7 surface-finish class

    filesystem::create_directories("out");
    cout << unitbuf << fixed << setprecision(4);   // flush every << so a crash localizes
    cout << "Loading CNC surrogate: " << model_path.string() << "\n";
    NeuralNetwork net;
    net.load(model_path);

    // --- direct sanity probe: does the net vary with feed (the dominant Ra driver)? ---
    {
        cout << "[probe] direct net.calculate_outputs on manual inputs (vc=350, ap=0.5, VB=0):\n";
        for (const float fd : { 0.07f, 0.10f, 0.13f }) {
            MatrixR x(1, 4); x(0,0) = 350.0f; x(0,1) = fd; x(0,2) = 0.5f; x(0,3) = 0.0f;
            const MatrixR y = net.calculate_outputs(x);
            cout << "   feed=" << fd << " -> [" << y(0,0) << ", " << y(0,1) << "]\n";
        }
    }

    const vector<Variable> in_vars  = net.get_input_variables();
    const vector<Variable> out_vars = net.get_output_variables();
    const Index n_in  = (Index)in_vars.size();
    const Index n_out = (Index)out_vars.size();
    cout << "Inputs: " << n_in << "  Outputs: " << n_out << "\n";
    for (const auto& v : in_vars)  cout << "  in : " << v.name << "\n";
    for (const auto& v : out_vars) cout << "  out: " << v.name << "\n";

    const Index ra = find_output(out_vars, "Ra");
    const Index ff = find_output(out_vars, "F");
    if (ra < 0 || ff < 0) { cerr << "Expected outputs Ra and F.\n"; return 1; }

    ResponseOptimization opt(&net);
    const auto domain = opt.get_original_domain("Input");

    // Pick a REACHABLE Ra target from the surrogate's own output range (an interior
    // percentile), unless one is passed explicitly. Ra=1.6 (N7 ceiling) sits at the
    // extreme of the reachable range, so as an *equality* target it yields a near-empty
    // feasible set; a mid-range finish spec gives a genuine manifold.
    float Ra_used = Ra_target;
    {
        const MatrixR si = opt.calculate_random_inputs(domain, 3000);
        const MatrixR so = opt.calculate_outputs(si);
        vector<float> rv((size_t)so.rows());
        for (Index i = 0; i < so.rows(); ++i) rv[(size_t)i] = so(i, ra);
        ranges::sort(rv);
        const float ra_lo_s = rv.front(), ra_hi_s = rv.back();
        cout << "Reachable Ra ~ [" << ra_lo_s << ", " << ra_hi_s << "] um\n";
        if (argc <= 2) Ra_used = rv[rv.size() * 55 / 100];   // 55th percentile, interior & reachable
        cout << "Ra equality target = " << Ra_used << " um"
             << (argc > 2 ? " (user)" : " (interior percentile)") << "\n";
    }

    // export input domain box (centrality reference)
    {
        ofstream d("out/cnc_domain.csv");
        d << "input,min,max\n";
        for (Index c = 0; c < n_in; ++c)
            d << in_vars[(size_t)c].name << "," << domain.inferior_frontier(c)
              << "," << domain.superior_frontier(c) << "\n";
    }

    // MRR helper: inputs order is [cutting_speed, feed, depth_of_cut, tool_wear_VB]
    auto mrr_of = [&](const MatrixR& R, Index r) -> float {
        return R(r, 0) * R(r, 1) * R(r, 2);
    };

    // ================= RUN 1: FIXED Ra -> feasible recipe MANIFOLD =================
    cout << "\n=== RUN 1: FIXED Ra = " << Ra_used << " um -> feasible manifold ===\n";
    opt.clear_objectives();
    opt.set_objective("Ra", ResponseOptimization::Sense::Fixed, Ra_used);
    opt.set_relative_tolerance(1e-2f);
    opt.set_iterations(8);
    opt.set_evaluations_number(4000);

    MatrixR M = opt.perform_response_optimization();
    if (M.rows() == 0) { cerr << "No feasible solution (Ra target may be unreachable).\n"; return 2; }
    cout << "Manifold solutions: " << M.rows() << "\n";

    const vector<float> sensRa = robustness(net, M, n_in, n_out, ra, domain);

    // stats
    float ra_lo = M(0, n_in + ra), ra_hi = ra_lo, f_lo = M(0, n_in + ff), f_hi = f_lo;
    float mrr_lo = mrr_of(M, 0), mrr_hi = mrr_lo;
    for (Index r = 0; r < M.rows(); ++r) {
        ra_lo = min(ra_lo, M(r, n_in + ra)); ra_hi = max(ra_hi, M(r, n_in + ra));
        f_lo  = min(f_lo,  M(r, n_in + ff)); f_hi  = max(f_hi,  M(r, n_in + ff));
        const float m = mrr_of(M, r); mrr_lo = min(mrr_lo, m); mrr_hi = max(mrr_hi, m);
    }
    cout << "  Ra  on manifold : [" << ra_lo << ", " << ra_hi << "] (target " << Ra_used << ")\n";
    cout << "  F   trade-off   : [" << f_lo << ", " << f_hi << "] N   (MINIMIZE)\n";
    cout << "  MRR trade-off   : [" << mrr_lo << ", " << mrr_hi << "]   (MAXIMIZE)\n";

    // dump manifold with F, MRR, robustness
    {
        ofstream csv("out/cnc_manifold.csv");
        for (Index c = 0; c < n_in; ++c) csv << in_vars[(size_t)c].name << ",";
        csv << "Ra,F,MRR,sensitivity_Ra\n";
        for (Index r = 0; r < M.rows(); ++r) {
            for (Index c = 0; c < n_in; ++c) csv << M(r, c) << ",";
            csv << M(r, n_in + ra) << "," << M(r, n_in + ff) << ","
                << mrr_of(M, r) << "," << sensRa[(size_t)r] << "\n";
        }
        cout << "  wrote out/cnc_manifold.csv\n";
    }

    // robustness + centrality selection.
    // margin = worst-case normalized distance to the nearest domain wall (1=box centre,
    // 0=on a wall). It quantifies extrapolation/saturation risk of the surrogate: points near
    // a wall sit at the edge of the training support, where the net extrapolates and saturates.
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
    cout << "\n--- get_robust_point(front, balance) : centrality <-> robustness ---\n";
    for (const float b : { 0.0f, 0.5f, 1.0f }) {
        const auto [idx, row] = opt.get_robust_point(M, b);
        cout << "  balance=" << b << " -> row " << idx
             << "  Ra=" << row(n_in + ra) << "  F=" << row(n_in + ff)
             << "  MRR=" << mrr_of(M, idx) << "  sensRa=" << sensRa[(size_t)idx]
             << "  wallMargin=" << wall_margin(row) << "\n";
    }

    // ============ RUN 2: mixed-sense solver FIX Ra + MINIMIZE F ============
    cout << "\n=== RUN 2: FIX Ra=" << Ra_used << " + MINIMIZE F (solver) ===\n";
    opt.clear_objectives();
    opt.set_objective("Ra", ResponseOptimization::Sense::Fixed, Ra_used);
    opt.set_objective("F",  ResponseOptimization::Sense::Minimize);
    opt.set_relative_tolerance(1e-2f);
    opt.set_iterations(8);
    opt.set_evaluations_number(4000);
    MatrixR MF = opt.perform_response_optimization();
    if (MF.rows() > 0) {
        // report the lowest-F recipe
        Index best = 0;
        for (Index r = 1; r < MF.rows(); ++r) if (MF(r, n_in + ff) < MF(best, n_in + ff)) best = r;
        cout << "  min-F recipe: F=" << MF(best, n_in + ff) << "  Ra=" << MF(best, n_in + ra)
             << "  MRR=" << mrr_of(MF, best) << "  x=[";
        for (Index c = 0; c < n_in; ++c) cout << (c ? ", " : "") << MF(best, c);
        cout << "]\n";
        ofstream csv("out/cnc_fixRa_minF.csv");
        for (Index c = 0; c < n_in; ++c) csv << in_vars[(size_t)c].name << ",";
        csv << "Ra,F,MRR\n";
        for (Index r = 0; r < MF.rows(); ++r) {
            for (Index c = 0; c < n_in; ++c) csv << MF(r, c) << ",";
            csv << MF(r, n_in + ra) << "," << MF(r, n_in + ff) << "," << mrr_of(MF, r) << "\n";
        }
        cout << "  wrote out/cnc_fixRa_minF.csv (" << MF.rows() << " rows)\n";
    } else cout << "  (no solution)\n";

    // ============ RUN 3: MULTI-TARGET  FIX Ra & FIX F ============
    // pick an F target inside the manifold's F range (median-ish)
    const float F_target = 0.5f * (f_lo + f_hi);
    cout << "\n=== RUN 3: MULTI-TARGET FIX Ra=" << Ra_used << " & FIX F=" << F_target << " ===\n";
    opt.clear_objectives();
    opt.set_objective("Ra", ResponseOptimization::Sense::Fixed, Ra_used);
    opt.set_objective("F",  ResponseOptimization::Sense::Fixed, F_target);
    opt.set_relative_tolerance(2e-2f);
    opt.set_iterations(8);
    opt.set_evaluations_number(6000);
    MatrixR MT = opt.perform_response_optimization();
    cout << "  intersection solutions: " << MT.rows() << "\n";
    if (MT.rows() > 0) {
        ofstream csv("out/cnc_multitarget.csv");
        for (Index c = 0; c < n_in; ++c) csv << in_vars[(size_t)c].name << ",";
        csv << "Ra,F,MRR\n";
        for (Index r = 0; r < MT.rows(); ++r) {
            for (Index c = 0; c < n_in; ++c) csv << MT(r, c) << ",";
            csv << MT(r, n_in + ra) << "," << MT(r, n_in + ff) << "," << mrr_of(MT, r) << "\n";
        }
        cout << "  wrote out/cnc_multitarget.csv\n";
    }

    cout << "\nCase A done.\n";
    return 0;
}
