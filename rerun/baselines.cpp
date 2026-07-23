// RIAI baselines - honest comparison with the SAME repair given to the baselines.
//
// For each case (cnc | hea) we compare, at an equal total evaluation budget and an
// identical feasibility criterion (output equality band + input constraints):
//
//   1. reject_norepair : uniform box sampling + band filter, NO repair.
//   2. reject_repair   : perform_response_optimization with iterations=1 -> one round
//                        of sample + (affine input + output) repair + filter. Same
//                        repair operators as ours; no iterative domain contraction.
//   3. ours            : perform_response_optimization with iterations=8 (contraction).
//
// Metrics per method: feasible count, feasibility rate (feasible / budget), input-
// constraint satisfaction (simplex residual for HEA), diversity (mean per-input fill
// of the domain), robustness of the feasible set (analytic Jacobian S, lower=better),
// and wall-clock. Written to out/baselines_<case>.csv.

#include "opennn/neural_network.h"
#include "opennn/network_differential.h"
#include "opennn/response_optimization.h"
#include "opennn/response_constraints.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace opennn;
using namespace std;
using Clock = chrono::steady_clock;

struct Case {
    string name, model;
    string target_output;        // fixed output name
    bool   simplex = false;      // HEA: sum of inputs == 1
    int    pctile = 55;          // reachable target percentile
};

static Index out_index(const vector<Variable>& outs, const string& n) {
    for (Index i = 0; i < (Index)outs.size(); ++i) if (outs[(size_t)i].name == n) return i;
    return -1;
}

struct Metrics {
    string method;
    long budget = 0;
    long feasible = 0;
    double feas_rate = 0;
    double simplex_resid = 0;   // mean |sum(x)-1| over returned inputs (HEA)
    double mean_fill = 0;       // mean per-free-input range/domain over feasible set
    double mean_S = 0, min_S = 0;
    double seconds = 0;
};

// span-weighted robustness S for the fixed output, over a set of input rows.
static void robustness_stats(NetworkDifferential& diff, const MatrixR& X, Index n_in,
                             Index n_out, Index out_col, const VectorR& span,
                             double& mean_S, double& min_S)
{
    mean_S = 0; min_S = 1e30;
    if (X.rows() == 0) { min_S = 0; return; }
    VectorR cot = VectorR::Zero(n_out); cot(out_col) = 1.0f;
    for (Index r = 0; r < X.rows(); ++r) {
        const VectorR x = X.row(r).transpose();
        const VectorR g = diff.vjp(x, cot);
        double s = 0; for (Index c = 0; c < n_in; ++c) s += (g(c)*span(c))*(g(c)*span(c));
        s = sqrt(s);
        mean_S += s; min_S = min(min_S, s);
    }
    mean_S /= (double)X.rows();
}

static double mean_fill_of(const MatrixR& X, const ResponseOptimization::Domain& dom, Index n_in) {
    if (X.rows() == 0) return 0;
    double acc = 0; int free = 0;
    for (Index c = 0; c < n_in; ++c) {
        float lo = X(0,c), hi = X(0,c);
        for (Index r = 0; r < X.rows(); ++r) { lo = min(lo, X(r,c)); hi = max(hi, X(r,c)); }
        const float sp = dom.superior_frontier(c) - dom.inferior_frontier(c);
        if (sp > 1e-9f) { acc += (hi-lo)/sp; ++free; }
    }
    return free ? acc/free*100.0 : 0.0;
}

int main(int argc, char** argv)
{
    const string which = (argc > 1) ? argv[1] : "cnc";
    Case C;
    if (which == "hea") C = { "hea", "models/hea_10elt.json", "HV", true, 60 };
    else                C = { "cnc", "models/cnc_multi.json", "Ra", false, 55 };

    filesystem::create_directories("out");
    cout << unitbuf << fixed << setprecision(4);
    cout << "=== BASELINES (" << C.name << ") ===\nLoading " << C.model << "\n";
    NeuralNetwork net; net.load(C.model);
    const auto in_vars = net.get_input_variables();
    const auto out_vars = net.get_output_variables();
    const Index n_in = (Index)in_vars.size(), n_out = (Index)out_vars.size();
    const Index ok = out_index(out_vars, C.target_output);

    ResponseOptimization opt(&net);
    const auto domain = opt.get_original_domain("Input");
    VectorR span(n_in);
    for (Index c = 0; c < n_in; ++c) span(c) = domain.superior_frontier(c) - domain.inferior_frontier(c);
    NetworkDifferential diff; diff.build(net);

    // reachable target + common band (identical feasibility test for all methods)
    const MatrixR si = opt.calculate_random_inputs(domain, 4000);
    const MatrixR so = opt.calculate_outputs(si);
    vector<float> ov((size_t)so.rows());
    for (Index i = 0; i < so.rows(); ++i) ov[(size_t)i] = so(i, ok);
    sort(ov.begin(), ov.end());
    const float o_lo = ov.front(), o_hi = ov.back();
    const float target = ov[ov.size() * C.pctile / 100];
    const float rel_tol = 1e-2f;
    const float band = rel_tol * (o_hi - o_lo);            // common half-width
    const float simplex_tol = 1e-4f;
    cout << C.target_output << " reachable [" << o_lo << ", " << o_hi << "], target=" << target
         << ", band=+/-" << band << "\n";

    auto simplex_resid = [&](const MatrixR& X) {
        if (!C.simplex || X.rows() == 0) return 0.0;
        double acc = 0; for (Index r = 0; r < X.rows(); ++r) {
            double s = 0; for (Index c = 0; c < n_in; ++c) s += X(r,c); acc += fabs(s - 1.0);
        } return acc / X.rows();
    };
    // keep only rows satisfying the COMMON feasibility test (band + simplex if applicable)
    auto refilter = [&](const MatrixR& X, const MatrixR& Y) {
        vector<Index> keep;
        for (Index r = 0; r < X.rows(); ++r) {
            if (fabs(Y(r, ok) - target) > band) continue;
            if (C.simplex) { double s=0; for (Index c=0;c<n_in;++c) s+=X(r,c); if (fabs(s-1.0) > simplex_tol) continue; }
            keep.push_back(r);
        }
        MatrixR Xk((Index)keep.size(), n_in);
        for (size_t i = 0; i < keep.size(); ++i) Xk.row((Index)i) = X.row(keep[i]);
        return Xk;
    };

    const long BUDGET = 48000;   // 8 iters x 6000 evals; also the one-shot budget (equal total)
    vector<Metrics> results;

    auto record = [&](const string& m, const MatrixR& Xfeas, long budget, double secs, double sresid) {
        Metrics M; M.method = m; M.budget = budget; M.feasible = Xfeas.rows();
        M.feas_rate = budget ? (double)Xfeas.rows()/budget : 0;
        M.simplex_resid = sresid;
        M.mean_fill = mean_fill_of(Xfeas, domain, n_in);
        robustness_stats(diff, Xfeas, n_in, n_out, ok, span, M.mean_S, M.min_S);
        M.seconds = secs;
        results.push_back(M);
        cout << "  " << left << setw(16) << m << right
             << "  feasible=" << setw(5) << M.feasible
             << "  rate=" << setw(8) << M.feas_rate
             << "  fill%=" << setw(7) << M.mean_fill
             << "  meanS=" << setw(9) << M.mean_S
             << "  minS=" << setw(9) << M.min_S
             << "  simplex=" << setw(8) << M.simplex_resid
             << "  " << M.seconds << "s\n";
    };

    // ---- 1. rejection, NO repair ----
    {
        auto t0 = Clock::now();
        const MatrixR X = opt.calculate_random_inputs(domain, BUDGET);
        const MatrixR Y = opt.calculate_outputs(X);
        const MatrixR Xf = refilter(X, Y);
        double secs = chrono::duration<double>(Clock::now()-t0).count();
        record("reject_norepair", Xf, BUDGET, secs, simplex_resid(X));
    }

    // ---- 2. rejection + repair (1 iteration) ----
    {
        auto t0 = Clock::now();
        opt.clear_objectives(); opt.clear_formula_constraints();
        if (C.simplex) {
            string sx; for (Index c=0;c<n_in;++c) sx += (c?" + ":"") + in_vars[(size_t)c].name;
            opt.set_formula_constraint(sx, ComparisonOperator::EqualTo, 1.0f, 1.0f);
        }
        opt.set_objective(C.target_output, ResponseOptimization::Sense::Fixed, target);
        opt.set_relative_tolerance(rel_tol);
        opt.set_min_feasible_ratio(1e-4f);
        opt.set_iterations(1);
        opt.set_evaluations_number((int)BUDGET);
        opt.set_max_total_evaluations((Index)(BUDGET * 8));   // cap oversampling runaway
        MatrixR M;
        try { M = opt.perform_response_optimization(); }
        catch (const exception& e) { cout << "  [reject_repair failed: " << e.what() << "]\n"; }
        MatrixR X = (M.rows() ? M.leftCols(n_in).eval() : MatrixR(0, n_in));
        MatrixR Y = (M.rows() ? M.rightCols(n_out).eval() : MatrixR(0, n_out));
        MatrixR Xf = refilter(X, Y);
        double secs = chrono::duration<double>(Clock::now()-t0).count();
        record("reject_repair", Xf, BUDGET, secs, simplex_resid(X));
    }

    // ---- 3. ours (iterative contraction), same total budget ----
    {
        auto t0 = Clock::now();
        opt.clear_objectives(); opt.clear_formula_constraints();
        if (C.simplex) {
            string sx; for (Index c=0;c<n_in;++c) sx += (c?" + ":"") + in_vars[(size_t)c].name;
            opt.set_formula_constraint(sx, ComparisonOperator::EqualTo, 1.0f, 1.0f);
        }
        opt.set_objective(C.target_output, ResponseOptimization::Sense::Fixed, target);
        opt.set_relative_tolerance(rel_tol);
        opt.set_min_feasible_ratio(1e-4f);
        const int iters = 8;
        opt.set_iterations(iters);
        opt.set_evaluations_number((int)(BUDGET / iters));
        opt.set_max_total_evaluations((Index)(BUDGET * 8));
        MatrixR M;
        try { M = opt.perform_response_optimization(); }
        catch (const exception& e) { cout << "  [ours failed: " << e.what() << "]\n"; }
        MatrixR X = (M.rows() ? M.leftCols(n_in).eval() : MatrixR(0, n_in));
        MatrixR Y = (M.rows() ? M.rightCols(n_out).eval() : MatrixR(0, n_out));
        MatrixR Xf = refilter(X, Y);
        double secs = chrono::duration<double>(Clock::now()-t0).count();
        record("ours_iterative", Xf, BUDGET, secs, simplex_resid(X));
    }

    // ---- dump ----
    ofstream csv("out/baselines_" + C.name + ".csv");
    csv << "method,budget,feasible,feas_rate,simplex_resid,mean_fill_pct,mean_S,min_S,seconds\n";
    for (const auto& M : results)
        csv << M.method << "," << M.budget << "," << M.feasible << "," << M.feas_rate << ","
            << M.simplex_resid << "," << M.mean_fill << "," << M.mean_S << "," << M.min_S << ","
            << M.seconds << "\n";
    cout << "wrote out/baselines_" << C.name << ".csv\n";
    return 0;
}
