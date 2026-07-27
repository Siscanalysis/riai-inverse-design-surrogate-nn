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
// best residual to the exact target |f_k(x)-c|, and wall-clock. Written to
// out/baselines_<case>.csv.
//
// Extra CNC-only experiments:
//   - Exact target (no tolerance interval): sweep the equality band toward zero and count
//     how many solutions AT the target each method returns. Rejection collapses to 0;
//     IDC's contraction stays populated. -> out/baselines_cnc_exact.csv.
//   - Mixed-sense (fix Ra + minimize F): best feasible F, ours vs rejection.

#include "opennn/neural_network.h"
#include "opennn/network_differential.h"
#include "opennn/response_optimization.h"
#include "opennn/response_constraints.h"
#include "opennn/model_expression.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
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
    long true_evals = 0;        // actual surrogate forward passes (incl. repair)
    long feasible = 0;
    double feas_rate = 0;
    double simplex_resid = 0;   // mean |sum(x)-1| over returned inputs (HEA)
    double mean_fill = 0;       // mean per-free-input range/domain over feasible set
    double mean_S = 0, min_S = 0;
    double best_resid = 0;      // min |f_k(x)-c| the method achieves (EXACT target, no band)
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
    // export the surrogate as a standalone Python function for the pymoo baselines
    try { ModelExpression(&net).save("models/" + C.name + "_surrogate.py",
                                     ModelExpression::ProgrammingLanguage::Python); }
    catch (const exception& e) { cout << "  [python export failed: " << e.what() << "]\n"; }
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

    // Shared HARD budget: every method gets EXACTLY this many true surrogate forwards.
    // ours uses iterations=6 so it CANNOT early-stop before the last iteration
    // (min_iterations=4 in the library => earliest stop is i=5, which is the final
    // iteration when max_iterations=6) => ours spends the whole budget, same as the rest.
    const long BUDGET = 12000;   // 6 iters x 2000 evals; also the one-shot budget (equal total)
    vector<Metrics> results;

    auto record = [&](const string& m, const MatrixR& Xfeas, long budget, long tevals, double secs, double sresid, double bresid) {
        Metrics M; M.method = m; M.budget = budget; M.true_evals = tevals; M.feasible = Xfeas.rows();
        M.feas_rate = tevals ? (double)Xfeas.rows()/tevals : 0;   // rate over TRUE evals (fair)
        M.simplex_resid = sresid;
        M.best_resid = bresid;
        M.mean_fill = mean_fill_of(Xfeas, domain, n_in);
        robustness_stats(diff, Xfeas, n_in, n_out, ok, span, M.mean_S, M.min_S);
        M.seconds = secs;
        results.push_back(M);
        // dump the feasible inputs so the extrapolation analysis (wall margin + distance to
        // training data) can compare the WHOLE distribution per method, not just a summary.
        {
            ofstream fx("out/feasible_" + C.name + "_" + m + ".csv");
            for (Index c = 0; c < n_in; ++c) fx << (c ? "," : "") << in_vars[(size_t)c].name;
            fx << "\n";
            for (Index r = 0; r < Xfeas.rows(); ++r) {
                for (Index c = 0; c < n_in; ++c) fx << (c ? "," : "") << Xfeas(r, c);
                fx << "\n";
            }
        }
        cout << "  " << left << setw(16) << m << right
             << "  feas=" << setw(5) << M.feasible
             << "  trueEvals=" << setw(7) << M.true_evals
             << "  rate=" << setw(8) << M.feas_rate
             << "  fill%=" << setw(7) << M.mean_fill
             << "  minS=" << setw(9) << M.min_S
             << "  bestResid=" << setw(10) << M.best_resid
             << "  simplex=" << setw(7) << M.simplex_resid
             << "  " << M.seconds << "s\n";
    };
    // best |f_k(x)-c| over a set of output rows (the method's closest approach to the EXACT target)
    auto best_resid_of = [&](const MatrixR& Y, Index col) {
        double b = 1e30;
        for (Index r = 0; r < Y.rows(); ++r) b = min(b, (double)fabs(Y(r, col) - target));
        return (Y.rows() ? b : 0.0);
    };

    // ---- 1. rejection, NO repair ----
    {
        auto t0 = Clock::now();
        const MatrixR X = opt.calculate_random_inputs(domain, BUDGET);
        const MatrixR Y = opt.calculate_outputs(X);
        const MatrixR Xf = refilter(X, Y);
        double secs = chrono::duration<double>(Clock::now()-t0).count();
        // exact-target residual: closest any of the 12000 uniform samples gets to c (no band).
        record("reject_norepair", Xf, BUDGET, BUDGET, secs, simplex_resid(X), best_resid_of(Y, ok));
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
        record("reject_repair", Xf, BUDGET, (long)opt.get_evaluations_used(), secs, simplex_resid(X), best_resid_of(Y, ok));
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
        const int iters = 6;                 // spends the FULL budget (see BUDGET note)
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
        record("ours_iterative", Xf, BUDGET, (long)opt.get_evaluations_used(), secs, simplex_resid(X), best_resid_of(Y, ok));
    }

    // ---- Exact target (no tolerance interval): populated solution set at a vanishing band ----
    // The equality band used above is only a COLLECTION device to gather many manifold points.
    // If the specification is a single EXACT value, the decisive question is not "how close does
    // the single best point get" (with a low-dimensional output a uniform sampler may get one
    // lucky near-hit) but "how many solutions AT the exact value does the method still return".
    // Hitting a measure-zero target by chance is vanishingly unlikely, so as the band shrinks the
    // rejection set collapses toward zero. IDC needs no user tolerance: domain contraction
    // concentrates the whole budget onto the target, so it keeps returning a populated exact set.
    // Same 12000-eval budget; we sweep the tolerance toward zero and count returned solutions.
    // Restricted to the box-domain case (CNC): on the HEA simplex the exact-target manifold is a
    // measure-zero set in 10-D that even IDC cannot populate at a vanishing band within budget, so
    // an exact-target claim there would be dishonest; the simplex/repair story is made separately.
    if (!C.simplex) {
        opt.clear_objectives(); opt.clear_formula_constraints();   // pure uniform reference (no constraint-aware sampling)
        // one uniform reference sample at the same budget, reused for every band width
        const MatrixR Xu = opt.calculate_random_inputs(domain, BUDGET);
        const MatrixR Yu = opt.calculate_outputs(Xu);
        const double range = (double)(o_hi - o_lo);
        double reject_best = 1e30;
        for (Index r = 0; r < Yu.rows(); ++r) reject_best = min(reject_best, (double)fabs(Yu(r, ok) - target));

        ofstream ex("out/baselines_" + C.name + "_exact.csv");
        ex << "rel_tol,band,reject_feasible,reject_fill_pct,reject_best_resid,"
              "ours_feasible,ours_fill_pct,ours_best_resid,ours_true_evals\n";
        cout << "\n-- Exact target (no tolerance interval): solutions returned AT the target, "
             << "12000 evals (fill% = diversity of the returned set) --\n";
        cout << "   " << left << setw(9) << "rel_tol" << setw(12) << "band"
             << setw(10) << "rej_feas" << setw(9) << "rej_fill"
             << setw(11) << "ours_feas" << setw(10) << "ours_fill" << setw(12) << "ours_resid" << right << "\n";
        for (float tol : {1e-2f, 1e-3f, 1e-4f, 1e-5f, 1e-6f}) {
            const double band_tol = (double)tol * range;
            long rej_feas = 0;
            vector<Index> rej_rows;
            for (Index r = 0; r < Yu.rows(); ++r) {
                if (fabs(Yu(r, ok) - target) > band_tol) continue;
                if (C.simplex) { double s=0; for (Index c=0;c<n_in;++c) s+=Xu(r,c); if (fabs(s-1.0) > simplex_tol) continue; }
                ++rej_feas; rej_rows.push_back(r);
            }
            MatrixR Xrej((Index)rej_rows.size(), n_in);
            for (size_t i = 0; i < rej_rows.size(); ++i) Xrej.row((Index)i) = Xu.row(rej_rows[i]);
            const double rej_fill = mean_fill_of(Xrej, domain, n_in);

            opt.clear_objectives(); opt.clear_formula_constraints();
            if (C.simplex) {
                string sx; for (Index c=0;c<n_in;++c) sx += (c?" + ":"") + in_vars[(size_t)c].name;
                opt.set_formula_constraint(sx, ComparisonOperator::EqualTo, 1.0f, 1.0f);
            }
            opt.set_objective(C.target_output, ResponseOptimization::Sense::Fixed, target);
            opt.set_relative_tolerance(tol);                 // -> exact as tol -> 0
            opt.set_min_feasible_ratio(1e-6f);
            opt.set_iterations(6); opt.set_evaluations_number((int)(BUDGET/6));
            opt.set_max_total_evaluations((Index)(BUDGET*8));
            MatrixR Me;
            try { Me = opt.perform_response_optimization(); }
            catch (const exception& e) { cout << "   [ours exact tol=" << tol << " failed: " << e.what() << "]\n"; }
            double ours_best = 1e30; for (Index r = 0; r < Me.rows(); ++r) ours_best = min(ours_best, (double)fabs(Me(r, n_in+ok) - target));
            if (!Me.rows()) ours_best = 0;
            const MatrixR Xours = (Me.rows() ? Me.leftCols(n_in).eval() : MatrixR(0, n_in));
            const double ours_fill = mean_fill_of(Xours, domain, n_in);
            const long ev = (long)opt.get_evaluations_used();
            cout << "   " << left << setw(9) << tol << setw(12) << band_tol
                 << setw(10) << rej_feas << setw(9) << rej_fill
                 << setw(11) << Me.rows() << setw(10) << ours_fill << setw(12) << ours_best << right << "\n";
            ex << tol << "," << band_tol << "," << rej_feas << "," << rej_fill << "," << (rej_feas ? reject_best : 0.0)
               << "," << Me.rows() << "," << ours_fill << "," << ours_best << "," << ev << "\n";
        }
        cout << "   (uniform sampler best single residual = " << reject_best
             << "; below that band the rejection set is empty.)\n";
    }

    // ---- Mixed-sense optimization (CNC only): FIX Ra + MINIMIZE F ----
    // The pure fixed-target above has nothing to optimize, so an optimizer cannot beat a
    // sampler. Here there IS a secondary objective (min F), and ours (contraction toward the
    // objective) should reach a lower best-feasible F than rejection (uniform sampling) at
    // equal/less true budget.
    if (!C.simplex) {
        const Index ff = out_index(out_vars, "F");
        if (ff >= 0) {
            auto best_feasible_F = [&](const MatrixR& X) {
                if (!X.rows()) return 1e30;
                const MatrixR Y = opt.calculate_outputs(X);
                double m = 1e30;
                for (Index r = 0; r < X.rows(); ++r)
                    if (fabs(Y(r, ok) - target) <= band) m = min(m, (double)Y(r, ff));
                return m;
            };
            const MatrixR Xr = opt.calculate_random_inputs(domain, BUDGET);
            const double f_reject = best_feasible_F(Xr);

            opt.clear_objectives(); opt.clear_formula_constraints();
            opt.set_objective(C.target_output, ResponseOptimization::Sense::Fixed, target);
            opt.set_objective("F", ResponseOptimization::Sense::Minimize);
            opt.set_relative_tolerance(rel_tol); opt.set_min_feasible_ratio(1e-4f);
            opt.set_iterations(6); opt.set_evaluations_number((int)(BUDGET/6));
            opt.set_max_total_evaluations((Index)(BUDGET*8));
            MatrixR Mo;
            try { Mo = opt.perform_response_optimization(); }
            catch (const exception& e) { cout << "  [ours mixed-sense failed: " << e.what() << "]\n"; }
            const long ev_ours = (long)opt.get_evaluations_used();
            double f_ours = 1e30;
            for (Index r = 0; r < Mo.rows(); ++r)
                if (fabs(Mo(r, n_in + ok) - target) <= band) f_ours = min(f_ours, (double)Mo(r, n_in + ff));

            cout << "\n-- Mixed-sense (fix Ra + minimize F): best feasible F (lower=better) --\n";
            cout << "   rejection (sampler): F=" << f_reject << "  (evals " << BUDGET << ")\n";
            cout << "   ours (optimizer):    F=" << f_ours   << "  (trueEvals " << ev_ours << ")\n";
            ofstream mf("out/baselines_" + C.name + "_mixedsense.csv");
            mf << "method,best_F,evals\nrejection," << f_reject << "," << BUDGET
               << "\nours," << f_ours << "," << ev_ours << "\n";
        }
    }

    // ---- dump ----
    ofstream csv("out/baselines_" + C.name + ".csv");
    csv << "method,budget,true_evals,feasible,feas_rate,simplex_resid,mean_fill_pct,mean_S,min_S,best_resid,seconds\n";
    for (const auto& M : results)
        csv << M.method << "," << M.budget << "," << M.true_evals << "," << M.feasible << "," << M.feas_rate << ","
            << M.simplex_resid << "," << M.mean_fill << "," << M.mean_S << "," << M.min_S << ","
            << M.best_resid << "," << M.seconds << "\n";
    cout << "wrote out/baselines_" << C.name << ".csv\n";
    return 0;
}
