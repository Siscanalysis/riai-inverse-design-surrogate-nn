// RIAI Case C - Inverse alloy design on NIMS low-alloy steels (mixed-sense).
//
// Surrogate: (C Si Mn P S Ni Cr Mo Cu V Al N Nb_Ta temperature_C)
//            -> (proof_stress, tensile_strength, elongation, reduction_area).
//
// Room-temperature alloy design: temperature_C is FIXED at 27 C (an EqualTo box),
// so the 13 composition levers are the design variables.
//
// Demonstrates, on ONE problem:
//   * FIXED (equality) tensile_strength = target  -> recover the composition MANIFOLD
//   * BI-OBJECTIVE over the manifold: MAXIMIZE elongation (ductility) vs MINIMIZE
//     an affine alloying-COST proxy -> ductility-cost Pareto front
//   * NONLINEAR weldability constraint  CEV = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15 <= 0.45
//       -> exercises the nonlinear (Gauss-Newton) repair operator
//   * AFFINE cost budget  3Ni+2Cr+8Mo+12V+1.5Cu <= budget
//       -> exercises the affine repair operator
//   * robustness (analytic d elongation/dx) + centrality selection via get_robust_point
//   * MULTI-TARGET: FIX tensile & FIX elongation -> intersection sub-manifold
//
// Writes CSVs into out/ for the manuscript figures/tables.

#include "opennn/neural_network.h"
#include "opennn/network_differential.h"
#include "opennn/response_optimization.h"
#include "opennn/response_constraints.h"   // ComparisonOperator
#include "opennn/random_utilities.h"

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
static Index find_input(const vector<Variable>& ins, const string& name)
{
    for (Index i = 0; i < (Index)ins.size(); ++i)
        if (ins[(size_t)i].name == name) return i;
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
    set_seed(1);   // reproducible manuscript numbers
    const filesystem::path model_path = (argc > 1) ? filesystem::path(argv[1])
                                                   : filesystem::path("models/steel_multi.json");
    const float T_design = 27.0f;   // room-temperature alloy design

    filesystem::create_directories("out");
    cout << unitbuf << fixed << setprecision(4);
    cout << "Loading steel surrogate: " << model_path.string() << "\n";
    NeuralNetwork net;
    net.load(model_path);

    const vector<Variable> in_vars  = net.get_input_variables();
    const vector<Variable> out_vars = net.get_output_variables();
    const Index n_in  = (Index)in_vars.size();
    const Index n_out = (Index)out_vars.size();
    cout << "Inputs: " << n_in << "  Outputs: " << n_out << "\n";

    const Index i_temp = find_input(in_vars, "temperature_C");
    const Index i_Ni = find_input(in_vars, "Ni"), i_Cr = find_input(in_vars, "Cr");
    const Index i_Mo = find_input(in_vars, "Mo"), i_Cu = find_input(in_vars, "Cu");
    const Index i_V  = find_input(in_vars, "V");
    const Index i_C  = find_input(in_vars, "C"),  i_Mn = find_input(in_vars, "Mn");
    const Index uts = find_output(out_vars, "tensile_strength_MPa");
    const Index elo = find_output(out_vars, "elongation_pct");
    const Index yld = find_output(out_vars, "proof_stress_MPa");
    if (uts < 0 || elo < 0 || i_temp < 0) { cerr << "Missing expected variables.\n"; return 1; }

    // analytic helpers on an input row
    auto cost_of = [&](const MatrixR& R, Index r) -> float {
        return 3.0f*R(r,i_Ni) + 2.0f*R(r,i_Cr) + 8.0f*R(r,i_Mo) + 12.0f*R(r,i_V) + 1.5f*R(r,i_Cu);
    };
    auto ceq_of = [&](const MatrixR& R, Index r) -> float {
        return R(r,i_C) + R(r,i_Mn)/6.0f + (R(r,i_Cr)+R(r,i_Mo)+R(r,i_V))/5.0f + (R(r,i_Ni)+R(r,i_Cu))/15.0f;
    };

    ResponseOptimization opt(&net);
    const auto domain = opt.get_original_domain("Input");

    // Pick a REACHABLE tensile target at T=27 (interior percentile), unless passed explicitly.
    float UTS_target = (argc > 2) ? stof(argv[2]) : -1.0f;
    {
        MatrixR si = opt.calculate_random_inputs(domain, 4000);
        for (Index i = 0; i < si.rows(); ++i) si(i, i_temp) = T_design;   // evaluate at room T
        const MatrixR so = opt.calculate_outputs(si);
        vector<float> uv((size_t)so.rows());
        for (Index i = 0; i < so.rows(); ++i) uv[(size_t)i] = so(i, uts);
        ranges::sort(uv);
        cout << "Reachable UTS at " << T_design << " C ~ [" << uv.front() << ", " << uv.back() << "] MPa\n";
        if (UTS_target < 0) UTS_target = uv[uv.size() * 55 / 100];   // 55th percentile, interior
        cout << "UTS equality target = " << UTS_target << " MPa"
             << (argc > 2 ? " (user)" : " (interior percentile)") << "\n";
    }

    // export input domain box (centrality reference)
    {
        ofstream d("out/steel_domain.csv");
        d << "input,min,max\n";
        for (Index c = 0; c < n_in; ++c)
            d << in_vars[(size_t)c].name << "," << domain.inferior_frontier(c)
              << "," << domain.superior_frontier(c) << "\n";
    }

    // solver budget (box domain; keep oversampling BOUNDED so nonlinear repair stays cheap).
    auto configure = [&](float tol, int evals) {
        opt.set_relative_tolerance(tol);
        opt.set_iterations(8);
        opt.set_evaluations_number(evals);
        opt.set_min_feasible_ratio(1e-3f);
        opt.set_max_total_evaluations((Index)evals * 6);
    };

    // ============ RUN 1: FIXED UTS @ T=27 + weldability -> composition MANIFOLD ============
    cout << "\n=== RUN 1: FIX UTS=" << UTS_target << " MPa @ T=27 C + weldability CEV<=0.45 -> manifold ===\n";
    opt.clear_objectives();
    opt.clear_formula_constraints();
    opt.set_objective("tensile_strength_MPa", ResponseOptimization::Sense::Fixed, UTS_target);
    opt.set_objective("temperature_C",        ResponseOptimization::Sense::Fixed, T_design);
    // nonlinear weldability cap (CEV in [0, 0.45]; CEV>=0 by construction) -> Gauss-Newton repair
    opt.set_formula_constraint("C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15",
                               ComparisonOperator::Between, 0.0f, 0.45f);
    configure(1e-2f, 6000);

    MatrixR M = opt.perform_response_optimization();
    if (M.rows() == 0) { cerr << "No feasible solution (UTS target may be unreachable at T=27).\n"; return 2; }
    cout << "Manifold solutions: " << M.rows() << "\n";

    const vector<float> sensElo = robustness(net, M, n_in, n_out, elo, domain);

    // stats over the manifold
    float uts_lo=M(0,n_in+uts), uts_hi=uts_lo, elo_lo=M(0,n_in+elo), elo_hi=elo_lo;
    float cost_lo=cost_of(M,0), cost_hi=cost_lo, ceq_lo=ceq_of(M,0), ceq_hi=ceq_lo;
    for (Index r = 0; r < M.rows(); ++r) {
        uts_lo=min(uts_lo,M(r,n_in+uts)); uts_hi=max(uts_hi,M(r,n_in+uts));
        elo_lo=min(elo_lo,M(r,n_in+elo)); elo_hi=max(elo_hi,M(r,n_in+elo));
        const float cc=cost_of(M,r); cost_lo=min(cost_lo,cc); cost_hi=max(cost_hi,cc);
        const float ce=ceq_of(M,r);  ceq_lo=min(ceq_lo,ce);  ceq_hi=max(ceq_hi,ce);
    }
    cout << "  UTS on manifold : [" << uts_lo << ", " << uts_hi << "] (target " << UTS_target << ")\n";
    cout << "  elongation      : [" << elo_lo << ", " << elo_hi << "] %   (MAXIMIZE)\n";
    cout << "  alloying cost   : [" << cost_lo << ", " << cost_hi << "]    (MINIMIZE)\n";
    cout << "  CEV weldability : [" << ceq_lo << ", " << ceq_hi << "] (cap 0.45)\n";

    // dump manifold with elongation, cost, CEV, robustness
    {
        ofstream csv("out/steel_manifold.csv");
        for (Index c = 0; c < n_in; ++c) csv << in_vars[(size_t)c].name << ",";
        csv << "proof_stress,tensile_strength,elongation,reduction_area,cost,CEV,sensitivity_elong\n";
        for (Index r = 0; r < M.rows(); ++r) {
            for (Index c = 0; c < n_in; ++c) csv << M(r, c) << ",";
            csv << M(r,n_in+yld) << "," << M(r,n_in+uts) << "," << M(r,n_in+elo) << ","
                << M(r,n_in+find_output(out_vars,"reduction_area_pct")) << ","
                << cost_of(M,r) << "," << ceq_of(M,r) << "," << sensElo[(size_t)r] << "\n";
        }
        cout << "  wrote out/steel_manifold.csv\n";
    }

    // centrality (wall margin) + get_robust_point selection
    auto wall_margin = [&](const VectorR& row) -> float {
        float worst = 1.0f;
        for (Index c = 0; c < n_in; ++c) {
            if (c == i_temp) continue;    // temperature is FIXED at 27 (design decision), not a free lever:
                                          // it sits on the domain floor, so it would pin the margin to 0.
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
             << "  elong=" << row(n_in+elo) << "  UTS=" << row(n_in+uts)
             << "  cost=" << cost_of(M,idx) << "  sensElo=" << sensElo[(size_t)idx]
             << "  wallMargin=" << wall_margin(row) << "\n";
    }

    // ============ RUN 2: ductility-cost PARETO front over the manifold ============
    // non-dominated set: maximize elongation, minimize cost (both computed above).
    cout << "\n=== RUN 2: ductility-cost Pareto front over the fixed-UTS manifold ===\n";
    {
        const Index n = M.rows();
        vector<char> dominated((size_t)n, 0);
        for (Index i = 0; i < n; ++i) {
            const float ei = M(i,n_in+elo), ci = cost_of(M,i);
            for (Index j = 0; j < n; ++j) {
                if (i == j) continue;
                const float ej = M(j,n_in+elo), cj = cost_of(M,j);
                if (ej >= ei && cj <= ci && (ej > ei || cj < ci)) { dominated[(size_t)i] = 1; break; }
            }
        }
        vector<Index> front;
        for (Index i = 0; i < n; ++i) if (!dominated[(size_t)i]) front.push_back(i);
        sort(front.begin(), front.end(), [&](Index a, Index b){ return cost_of(M,a) < cost_of(M,b); });
        cout << "  Pareto-optimal alloys: " << front.size() << " of " << n << " manifold points\n";

        ofstream csv("out/steel_pareto.csv");
        for (Index c = 0; c < n_in; ++c) csv << in_vars[(size_t)c].name << ",";
        csv << "tensile_strength,elongation,cost,CEV,sensitivity_elong\n";
        for (Index idx : front) {
            for (Index c = 0; c < n_in; ++c) csv << M(idx, c) << ",";
            csv << M(idx,n_in+uts) << "," << M(idx,n_in+elo) << "," << cost_of(M,idx) << ","
                << ceq_of(M,idx) << "," << sensElo[(size_t)idx] << "\n";
        }
        cout << "  wrote out/steel_pareto.csv\n";
        if (!front.empty()) {
            const Index lean = front.front(), duct = front.back();
            cout << "  lean-cost end : cost=" << cost_of(M,lean) << "  elong=" << M(lean,n_in+elo) << "%\n";
            cout << "  max-duct end  : cost=" << cost_of(M,duct) << "  elong=" << M(duct,n_in+elo) << "%\n";
        }
    }

    // ============ RUN 3: mixed-sense solver FIX UTS + FIX T + AFFINE cost budget + MAX elongation ============
    const float budget = 0.5f * (cost_lo + cost_hi);
    cout << "\n=== RUN 3: FIX UTS + FIX T=27 + cost<=" << budget << " (affine) + MAXIMIZE elongation ===\n";
    opt.clear_objectives();
    opt.clear_formula_constraints();
    opt.set_objective("tensile_strength_MPa", ResponseOptimization::Sense::Fixed, UTS_target);
    opt.set_objective("temperature_C",        ResponseOptimization::Sense::Fixed, T_design);
    opt.set_objective("elongation_pct",       ResponseOptimization::Sense::Maximize);
    opt.set_formula_constraint("C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15",
                               ComparisonOperator::Between, 0.0f, 0.45f);       // nonlinear weldability
    opt.set_formula_constraint("3*Ni + 2*Cr + 8*Mo + 12*V + 1.5*Cu",
                               ComparisonOperator::Between, 0.0f, budget);       // affine cost budget
    configure(1e-2f, 6000);
    MatrixR MF = opt.perform_response_optimization();
    if (MF.rows() > 0) {
        Index best = 0;
        for (Index r = 1; r < MF.rows(); ++r) if (MF(r,n_in+elo) > MF(best,n_in+elo)) best = r;
        cout << "  max-ductility lean alloy: elong=" << MF(best,n_in+elo) << "%  UTS=" << MF(best,n_in+uts)
             << "  cost=" << cost_of(MF,best) << "  CEV=" << ceq_of(MF,best) << "\n";
        ofstream csv("out/steel_fixUTS_maxElong.csv");
        for (Index c = 0; c < n_in; ++c) csv << in_vars[(size_t)c].name << ",";
        csv << "tensile_strength,elongation,cost,CEV\n";
        for (Index r = 0; r < MF.rows(); ++r) {
            for (Index c = 0; c < n_in; ++c) csv << MF(r, c) << ",";
            csv << MF(r,n_in+uts) << "," << MF(r,n_in+elo) << "," << cost_of(MF,r) << "," << ceq_of(MF,r) << "\n";
        }
        cout << "  wrote out/steel_fixUTS_maxElong.csv (" << MF.rows() << " rows)\n";
    } else cout << "  (no solution)\n";

    // ============ RUN 4: MULTI-TARGET  FIX UTS & FIX elongation ============
    const float elo_target = 0.5f * (elo_lo + elo_hi);
    cout << "\n=== RUN 4: MULTI-TARGET FIX UTS=" << UTS_target << " & FIX elongation=" << elo_target << " @ T=27 ===\n";
    opt.clear_objectives();
    opt.clear_formula_constraints();
    opt.set_objective("tensile_strength_MPa", ResponseOptimization::Sense::Fixed, UTS_target);
    opt.set_objective("elongation_pct",       ResponseOptimization::Sense::Fixed, elo_target);
    opt.set_objective("temperature_C",        ResponseOptimization::Sense::Fixed, T_design);
    configure(2e-2f, 8000);
    MatrixR MT = opt.perform_response_optimization();
    cout << "  intersection solutions: " << MT.rows() << "\n";
    if (MT.rows() > 0) {
        ofstream csv("out/steel_multitarget.csv");
        for (Index c = 0; c < n_in; ++c) csv << in_vars[(size_t)c].name << ",";
        csv << "tensile_strength,elongation,cost,CEV\n";
        for (Index r = 0; r < MT.rows(); ++r) {
            for (Index c = 0; c < n_in; ++c) csv << MT(r, c) << ",";
            csv << MT(r,n_in+uts) << "," << MT(r,n_in+elo) << "," << cost_of(MT,r) << "," << ceq_of(MT,r) << "\n";
        }
        cout << "  wrote out/steel_multitarget.csv\n";
    }

    cout << "\nCase C done.\n";
    return 0;
}
