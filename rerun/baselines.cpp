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

// ------------------------------------------------------------------------------------
// Shared repair used by NSGA-II: box clip, and (for the simplex) the closed-form min-norm
// projection onto {sum(x)=1} intersected with the box, applied iteratively with clipping
// and residual redistribution. This is exactly the affine repair for a single equality
// (a = 1): Dx_i = -e * a_i/||a||^2 = -e/|active|, then clip and freeze saturated vars.
// ------------------------------------------------------------------------------------
static void repair_x(VectorR& x, const VectorR& lo, const VectorR& up, bool simplex)
{
    const Index n = x.size();
    for (Index i = 0; i < n; ++i) x(i) = min(up(i), max(lo(i), x(i)));
    if (!simplex) return;
    for (int pass = 0; pass < 64; ++pass) {
        double s = 0; for (Index i = 0; i < n; ++i) s += x(i);
        const double e = s - 1.0;
        if (fabs(e) < 1e-9) break;
        // active = not saturated in the direction that would fix the residual
        vector<Index> active;
        for (Index i = 0; i < n; ++i) {
            if (e > 0 && x(i) > lo(i) + 1e-9f) active.push_back(i);
            else if (e < 0 && x(i) < up(i) - 1e-9f) active.push_back(i);
        }
        if (active.empty()) break;
        const double d = e / (double)active.size();
        for (Index i : active) x(i) = min(up(i), max(lo(i), (float)(x(i) - d)));
    }
}

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

// ------------------------------------------------------------------------------------
// NSGA-II (real-coded, SBX crossover + polynomial mutation, fast non-dominated sort +
// crowding). It optimizes the PROBLEM's own objectives, NOT robustness: for the CNC it is a
// genuine bi-objective search (minimize force F, maximize MRR) with the fixed-Ra equality
// handled as a constraint via Deb's constrained domination; for the HEA it is a single-
// objective feasibility search (drive HV to the band) on the simplex. Offspring get the same
// input repair (repair_x) each generation. Robustness is NEVER an objective here; like ours,
// it is selected POST-HOC on the returned front (record()/get_robust_point). Returns the final
// population inputs (P x n_in); the caller applies the common feasibility filter.
// ------------------------------------------------------------------------------------
static MatrixR nsga2_repair(NeuralNetwork& net,
                            ResponseOptimization& opt,
                            const ResponseOptimization::Domain& dom,
                            Index n_in, Index n_out, Index ok, Index ff, float target, float band,
                            bool cnc, bool simplex, const VectorR& lo, const VectorR& up,
                            int P, int G, long& evals_out)
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double eta_c = 15.0, eta_m = 20.0, pm = 1.0 / (double)n_in;

    auto mrr_of = [&](const VectorR& x) { return (double)x(0) * x(1) * x(2); };
    // batch objectives (both MINIMIZED) + constraint violation for a set of rows.
    //   CNC : f1 = F (minimize), f2 = -MRR (i.e. maximize MRR), cv = max(0, |Ra - c| - band).
    //   HEA : f1 = |HV - c| (feasibility), f2 = 0, cv = 0 (simplex handled by repair).
    auto eval_pop = [&](vector<VectorR>& X, vector<double>& f1, vector<double>& f2, vector<double>& cv) {
        const int m = (int)X.size();
        MatrixR M((Index)m, n_in);
        for (int i = 0; i < m; ++i) M.row(i) = X[i].transpose();
        const MatrixR Y = net.calculate_outputs(M);
        f1.resize(m); f2.resize(m); cv.resize(m);
        for (int i = 0; i < m; ++i) {
            if (cnc) {
                f1[i] = Y(i, ff);
                f2[i] = -mrr_of(X[i]);
                cv[i] = max(0.0, (double)fabs(Y(i, ok) - target) - (double)band);
            } else {
                f1[i] = fabs(Y(i, ok) - target);
                f2[i] = 0.0;
                cv[i] = 0.0;
            }
        }
    };
    // Deb's constrained domination: a feasible point beats an infeasible one; among two
    // infeasible, less violation wins; among two feasible (equal cv), Pareto on the objectives.
    auto dominates = [](double a1, double a2, double acv, double b1, double b2, double bcv) {
        if (acv != bcv) return acv < bcv;
        return a1 <= b1 && a2 <= b2 && (a1 < b1 || a2 < b2);
    };
    // fast non-dominated sort -> rank per index
    auto nds = [&](const vector<double>& f1, const vector<double>& f2, const vector<double>& cv, vector<int>& rank) {
        const int m = (int)f1.size();
        rank.assign(m, 0);
        vector<vector<int>> S(m); vector<int> n(m, 0); vector<int> F;
        for (int p = 0; p < m; ++p) {
            for (int q = 0; q < m; ++q) if (p != q) {
                if (dominates(f1[p], f2[p], cv[p], f1[q], f2[q], cv[q])) S[p].push_back(q);
                else if (dominates(f1[q], f2[q], cv[q], f1[p], f2[p], cv[p])) n[p]++;
            }
            if (n[p] == 0) { rank[p] = 0; F.push_back(p); }
        }
        int r = 0;
        while (!F.empty()) {
            vector<int> next;
            for (int p : F) for (int q : S[p]) if (--n[q] == 0) { rank[q] = r + 1; next.push_back(q); }
            r++; F = next;
        }
    };
    auto crowding = [&](const vector<int>& idx, const vector<double>& f1,
                        const vector<double>& f2, vector<double>& cd) {
        for (int i : idx) cd[i] = 0;
        for (int obj = 0; obj < 2; ++obj) {
            const vector<double>& f = (obj == 0) ? f1 : f2;
            vector<int> s = idx;
            sort(s.begin(), s.end(), [&](int a, int b){ return f[a] < f[b]; });
            cd[s.front()] = cd[s.back()] = 1e30;
            const double range = f[s.back()] - f[s.front()] + 1e-12;
            for (size_t k = 1; k + 1 < s.size(); ++k)
                cd[s[k]] += (f[s[k+1]] - f[s[k-1]]) / range;
        }
    };

    // init
    vector<VectorR> pop(P);
    { MatrixR X0 = opt.calculate_random_inputs(dom, P);
      for (int i = 0; i < P; ++i) { VectorR x = X0.row(i).transpose(); repair_x(x, lo, up, simplex); pop[i] = x; } }
    vector<double> f1, f2, cv; eval_pop(pop, f1, f2, cv);
    long evals = P;

    vector<int> rank(P); vector<double> cd(P, 0);
    nds(f1, f2, cv, rank);
    { vector<int> all(P); for (int i=0;i<P;++i) all[i]=i; crowding(all, f1, f2, cd); }

    auto better = [&](int a, int b) {                       // crowded-comparison operator
        if (rank[a] != rank[b]) return rank[a] < rank[b];
        return cd[a] > cd[b];
    };
    auto tourn = [&]() { int a = rng()%P, b = rng()%P; return better(a,b) ? a : b; };

    for (int gen = 0; gen < G; ++gen) {
        // offspring
        vector<VectorR> Q(P);
        for (int i = 0; i < P; i += 2) {
            const VectorR& p1 = pop[tourn()]; const VectorR& p2 = pop[tourn()];
            VectorR c1 = p1, c2 = p2;
            for (Index j = 0; j < n_in; ++j) {                 // SBX
                if (U(rng) <= 0.5 && fabs(p1(j)-p2(j)) > 1e-12) {
                    double u = U(rng);
                    double beta = (u <= 0.5) ? pow(2*u, 1.0/(eta_c+1)) : pow(1.0/(2*(1-u)), 1.0/(eta_c+1));
                    double a = 0.5*((1+beta)*p1(j) + (1-beta)*p2(j));
                    double b = 0.5*((1-beta)*p1(j) + (1+beta)*p2(j));
                    c1(j) = (float)a; c2(j) = (float)b;
                }
            }
            auto mutate = [&](VectorR& x) {                    // polynomial mutation
                for (Index j = 0; j < n_in; ++j) if (U(rng) < pm) {
                    double u = U(rng), rng_j = up(j)-lo(j);
                    double delta = (u < 0.5) ? pow(2*u, 1.0/(eta_m+1)) - 1
                                             : 1 - pow(2*(1-u), 1.0/(eta_m+1));
                    x(j) = (float)(x(j) + delta * rng_j);
                }
            };
            mutate(c1); mutate(c2);
            repair_x(c1, lo, up, simplex); repair_x(c2, lo, up, simplex);
            Q[i] = c1; if (i+1 < P) Q[i+1] = c2;
        }
        vector<double> qf1, qf2, qcv; eval_pop(Q, qf1, qf2, qcv); evals += P;

        // combine parents + offspring, elitist truncation to P
        vector<VectorR> Rp = pop; Rp.insert(Rp.end(), Q.begin(), Q.end());
        vector<double> rf1 = f1; rf1.insert(rf1.end(), qf1.begin(), qf1.end());
        vector<double> rf2 = f2; rf2.insert(rf2.end(), qf2.begin(), qf2.end());
        vector<double> rcv = cv; rcv.insert(rcv.end(), qcv.begin(), qcv.end());
        vector<int> rrank; nds(rf1, rf2, rcv, rrank);
        const int M = (int)Rp.size();
        vector<double> rcd(M, 0);
        // group by rank
        int maxr = 0; for (int v : rrank) maxr = max(maxr, v);
        vector<vector<int>> fronts(maxr+1);
        for (int i = 0; i < M; ++i) fronts[rrank[i]].push_back(i);
        vector<int> sel;
        for (int r = 0; r <= maxr && (int)sel.size() < P; ++r) {
            crowding(fronts[r], rf1, rf2, rcd);
            vector<int> fr = fronts[r];
            sort(fr.begin(), fr.end(), [&](int a, int b){ return rcd[a] > rcd[b]; });
            for (int i : fr) { if ((int)sel.size() >= P) break; sel.push_back(i); }
        }
        vector<VectorR> np(P); vector<double> nf1(P), nf2(P), ncv(P);
        for (int i = 0; i < P; ++i) { np[i]=Rp[sel[i]]; nf1[i]=rf1[sel[i]]; nf2[i]=rf2[sel[i]]; ncv[i]=rcv[sel[i]]; }
        pop.swap(np); f1.swap(nf1); f2.swap(nf2); cv.swap(ncv);
        nds(f1, f2, cv, rank);
        { vector<int> all(P); for (int i=0;i<P;++i) all[i]=i; crowding(all, f1, f2, cd); }
    }
    evals_out = evals;
    MatrixR out((Index)P, n_in);
    for (int i = 0; i < P; ++i) out.row(i) = pop[i].transpose();
    return out;
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

    auto record = [&](const string& m, const MatrixR& Xfeas, long budget, long tevals, double secs, double sresid) {
        Metrics M; M.method = m; M.budget = budget; M.true_evals = tevals; M.feasible = Xfeas.rows();
        M.feas_rate = tevals ? (double)Xfeas.rows()/tevals : 0;   // rate over TRUE evals (fair)
        M.simplex_resid = sresid;
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
             << "  simplex=" << setw(7) << M.simplex_resid
             << "  " << M.seconds << "s\n";
    };

    // ---- 1. rejection, NO repair ----
    {
        auto t0 = Clock::now();
        const MatrixR X = opt.calculate_random_inputs(domain, BUDGET);
        const MatrixR Y = opt.calculate_outputs(X);
        const MatrixR Xf = refilter(X, Y);
        double secs = chrono::duration<double>(Clock::now()-t0).count();
        record("reject_norepair", Xf, BUDGET, BUDGET, secs, simplex_resid(X));
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
        record("reject_repair", Xf, BUDGET, (long)opt.get_evaluations_used(), secs, simplex_resid(X));
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
        record("ours_iterative", Xf, BUDGET, (long)opt.get_evaluations_used(), secs, simplex_resid(X));
    }

    // ---- 4. NSGA-II with the SAME repair ----
    {
        auto t0 = Clock::now();
        VectorR lo = domain.inferior_frontier, up = domain.superior_frontier;
        const Index ff_nsga = out_index(out_vars, "F");
        const int P = 100;
        const int G = max(1, (int)(BUDGET / P) - 1);   // ~ equal total eval budget
        long ev = 0;
        MatrixR X;
        try { X = nsga2_repair(net, opt, domain, n_in, n_out, ok, ff_nsga, target, band,
                               /*cnc=*/!C.simplex, C.simplex, lo, up, P, G, ev); }
        catch (const exception& e) { cout << "  [nsga2 failed: " << e.what() << "]\n"; }
        MatrixR Xf(0, n_in);
        double sres = 0;
        if (X.rows()) { MatrixR Y = opt.calculate_outputs(X); Xf = refilter(X, Y); sres = simplex_resid(X); }
        double secs = chrono::duration<double>(Clock::now()-t0).count();
        record("nsga2_repair", Xf, ev ? ev : BUDGET, ev ? ev : BUDGET, secs, sres);
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
    csv << "method,budget,true_evals,feasible,feas_rate,simplex_resid,mean_fill_pct,mean_S,min_S,seconds\n";
    for (const auto& M : results)
        csv << M.method << "," << M.budget << "," << M.true_evals << "," << M.feasible << "," << M.feas_rate << ","
            << M.simplex_resid << "," << M.mean_fill << "," << M.mean_S << "," << M.min_S << ","
            << M.seconds << "\n";
    cout << "wrote out/baselines_" << C.name << ".csv\n";
    return 0;
}
