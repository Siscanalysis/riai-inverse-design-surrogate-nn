// The inverse-design problems, stated once with the OpenNN response-optimization API
// so that every driver (and the Python baselines, through out/problems.json) reads the same
// statement.
//
// A problem is a specification (one network output that must equal a target value), the
// other constraints, and the objectives. The specification can be given to the library in
// two ways:
//
//   Formulation::Constraint  add_constraint(output, Equal, {target})   (repaired onto)
//   Formulation::Objective   add_objective(output, Fixed, target)      (scored as -|f - c|)
//   Formulation::Band        add_constraint(output, Between, target*(1 +- band))
//
// The band exists for the catalogue case. When every design variable is restricted to a finite
// set, the attainable responses are a finite set too, and a target that falls between two of
// them makes the equality unsatisfiable however good the solver is. The band is then the
// statement an engineer writes instead, and its half-width has to clear the spacing of the
// attainable values. Problems with continuous levers leave band at zero and never use it.

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "opennn/response_optimization/response_optimization.h"

namespace rai
{

using opennn::ResponseOptimization;
using Sense = ResponseOptimization::Objective::Sense;
using Condition = ResponseOptimization::Constraint::Condition;

enum class Formulation { Constraint, Objective, Band };

struct ConstraintStatement
{
    std::string expression;
    Condition condition;
    std::vector<float> values;
};

struct ObjectiveStatement
{
    std::string expression;
    Sense sense;
};

struct Problem
{
    std::string name;
    std::string specification;          // network output that must equal the target
    float target = 0.0f;
    float band = 0.0f;                  // relative half-width of the band formulation, 0 = unused
    std::vector<ConstraintStatement> constraints;
    std::vector<ObjectiveStatement> single;   // one objective (mixed-sense scalar task)
    std::vector<ObjectiveStatement> bi;       // two objectives (mixed-sense front task), may be empty
};


inline Problem make_problem(const std::string& name)
{
    Problem p;
    p.name = name;

    if (name == "concrete")
    {
        // C30/37 design: target mean strength f_ck + 8 MPa at 28 days, slump class S4.
        p.specification = "CompressiveStrength_MPa";
        p.target = 38.0f;
        // Design within the plant's operating window: each free input between the 5th and
        // 95th percentiles of the plant's 28-day mixes (the full data box lets the network
        // extrapolate into corners the plant never used, e.g. 110 kg/m3 of water).
        p.constraints = {
            {"Cement_kg_m3", Condition::Between, {234.0f, 491.0f}},
            {"FineAgg_kg_m3", Condition::Between, {708.0f, 962.0f}},
            {"CoarseAgg_kg_m3", Condition::Between, {756.0f, 1130.0f}},
            {"Water_kg_m3", Condition::Between, {155.0f, 241.0f}},
            {"WaterReducer_kg_m3", Condition::Between, {0.0f, 8.3f}},
            {"FlyAsh_kg_m3", Condition::Between, {0.0f, 87.0f}},
            {"TestAge_days", Condition::Equal, {28.0f}},
            {"Accelerator_kg_m3", Condition::Equal, {0.0f}},
            {"SilicaFume_kg_m3", Condition::Equal, {0.0f}},
            {"Slump_mm", Condition::Between, {160.0f, 210.0f}},
            {"Water_kg_m3 / (Cement_kg_m3 + FlyAsh_kg_m3)", Condition::LessEqual, {0.55f}},
            // Absolute volume balance: the components of a batchable mix add up to about one
            // cubic metre. Densities [kg/L]: cement 3.15, fly ash 2.30, water 1.00, fine
            // aggregate 2.65, coarse aggregate 2.70, water reducer 1.10. The band is the
            // 5th-95th percentile of the plant's own 28-day mixes, which close the balance
            // only loosely. It replaces the earlier band on the total mass, a weaker condition
            // that admits mixes no plant could batch.
            {"Cement_kg_m3/3.15 + FlyAsh_kg_m3/2.30 + Water_kg_m3/1.00 + FineAgg_kg_m3/2.65"
             " + CoarseAgg_kg_m3/2.70 + WaterReducer_kg_m3/1.10",
             Condition::Between, {970.0f, 1010.0f}},
        };
        p.single = {{"Cement_kg_m3", Sense::Minimize}};
        // Replacing cement by the less reactive fly ash needs more binder for the same strength.
        p.bi = {{"FlyAsh_kg_m3", Sense::Maximize}, {"Cement_kg_m3 + FlyAsh_kg_m3", Sense::Minimize}};
    }
    else if (name == "hea")
    {
        p.specification = "HV";
        p.target = 600.0f;
        p.constraints = {
            {"Al + Co + Cr + Cu + Fe + Mn + Mo + Ni + Ti + V", Condition::Equal, {1.0f}},
            {"Al; Co; Cr; Cu; Fe; Mn; Mo; Ni; Ti; V", Condition::Cardinality, {5.0f}},
        };
        // Rule-of-mixtures density [g/cm3]: sum c_i A_i / sum c_i A_i / rho_i, with the atomic
        // masses [g/mol] and room-temperature densities [g/cm3] of the pure elements.
        p.single = {{"(26.982*Al + 58.933*Co + 51.996*Cr + 63.546*Cu + 55.845*Fe + 54.938*Mn + 95.95*Mo"
                     " + 58.693*Ni + 47.867*Ti + 50.942*V)"
                     " / (26.982*Al/2.70 + 58.933*Co/8.90 + 51.996*Cr/7.19 + 63.546*Cu/8.96 + 55.845*Fe/7.874"
                     " + 54.938*Mn/7.21 + 95.95*Mo/10.28 + 58.693*Ni/8.908 + 47.867*Ti/4.506 + 50.942*V/6.11)",
                     Sense::Minimize}};
    }
    else if (name == "kernel")
    {
        // Latency budget on a tuned CUDA convolution kernel: 8000 us, near the median of the
        // measured configurations. The network predicts the base-10 logarithm of the duration,
        // whose measured range spans 68x, so the specification is stated on that scale; the
        // analysis converts back and reports the residual as a relative error in microseconds.
        p.specification = "log10_duration";
        p.target = 3.9030900f;                      // log10(8000)
        // +-1 % of the budget. The attainable predictions nearest the target are 7996 and
        // 8004 us and they are spaced 2.5 us apart on average, so a per-cent band holds several
        // dozen configurations while a strict equality holds none.
        p.band = 0.01f;
        p.constraints = {
            // Each knob is a finite catalogue, not an interval.
            {"TBX", Condition::AllowedSet, {8.0f, 16.0f, 32.0f, 64.0f}},
            {"TBY", Condition::AllowedSet, {8.0f, 16.0f, 32.0f, 64.0f}},
            {"LOCAL", Condition::AllowedSet, {0.0f, 1.0f, 2.0f}},
            {"WPTX", Condition::AllowedSet, {1.0f, 2.0f, 4.0f, 8.0f}},
            {"WPTY", Condition::AllowedSet, {1.0f, 2.0f, 4.0f, 8.0f}},
            {"VECTOR", Condition::AllowedSet, {1.0f, 2.0f, 4.0f}},
            {"UNROLL_FACTOR", Condition::AllowedSet, {1.0f, 7.0f}},
            {"PADDING", Condition::AllowedSet, {0.0f, 1.0f}},
            // The kernel's own launch rules, recovered from the measured set and verified to
            // contain every one of its 3311 configurations: a block holds at most 1024 threads;
            // the vector width divides the work per thread (both are powers of two, so the
            // divisibility is the inequality); and padding only applies when the kernel caches
            // in shared memory. Together they cut the catalogue from 9216 points to 4680.
            {"TBX * TBY", Condition::LessEqual, {1024.0f}},
            {"WPTX - VECTOR", Condition::GreaterEqual, {0.0f}},
            {"PADDING - LOCAL", Condition::LessEqual, {0.0f}},
        };
        // Among the configurations that meet the budget, the one that co-schedules best.
        p.single = {{"achieved_occupancy", Sense::Maximize}};
        // No second criterion: at a fixed latency the measured counters leave one configuration
        // undominated out of 366, so a front here would be an artefact. The fronts stay in the
        // two cases whose trade-off is physically established.
    }
    else if (name == "steel")
    {
        // Room-temperature design; IIW carbon equivalent as the weldability cap.
        p.specification = "tensile_strength_MPa";
        p.target = 600.0f;
        p.constraints = {
            {"temperature_C", Condition::Equal, {27.0f}},
            {"C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15", Condition::LessEqual, {0.45f}},
        };
        p.single = {{"elongation_pct", Sense::Maximize}};
        // Ductility against the content of the costly alloying elements, at fixed strength.
        p.bi = {{"elongation_pct", Sense::Maximize}, {"Ni + Cr + Mo + Cu + V", Sense::Minimize}};
    }
    else
        throw std::runtime_error("unknown case " + name);

    return p;
}


inline void apply(const Problem& p, ResponseOptimization& optimization, Formulation formulation,
                  const std::vector<ObjectiveStatement>& objectives)
{
    for (const ConstraintStatement& c : p.constraints)
        optimization.add_constraint(c.expression, c.condition, c.values);

    if (formulation == Formulation::Constraint)
        optimization.add_constraint(p.specification, Condition::Equal, {p.target});
    else if (formulation == Formulation::Band)
    {
        if (p.band <= 0.0f) throw std::runtime_error("case " + p.name + " has no band");
        // The band is quoted on the physical quantity, so on a logarithmic specification it is
        // the logarithm of the two ends and not the target plus or minus a fraction of itself.
        const bool logarithmic = p.specification.rfind("log10_", 0) == 0;
        const float value = logarithmic ? std::pow(10.0f, p.target) : p.target;
        const float low = value*(1.0f - p.band), high = value*(1.0f + p.band);
        optimization.add_constraint(p.specification, Condition::Between,
                                    logarithmic ? std::vector<float>{std::log10(low), std::log10(high)}
                                                : std::vector<float>{low, high});
    }
    else
        optimization.add_objective(p.specification, Sense::Fixed, p.target);

    for (const ObjectiveStatement& o : objectives)
        optimization.add_objective(o.expression, o.sense);
}


inline const char* condition_name(Condition c)
{
    switch (c)
    {
    case Condition::Equal: return "Equal";
    case Condition::Between: return "Between";
    case Condition::GreaterEqual: return "GreaterEqual";
    case Condition::LessEqual: return "LessEqual";
    case Condition::Greater: return "Greater";
    case Condition::Less: return "Less";
    case Condition::AllowedSet: return "AllowedSet";
    case Condition::Integer: return "Integer";
    case Condition::Cardinality: return "Cardinality";
    }
    return "?";
}


inline const char* sense_name(Sense s)
{
    return s == Sense::Minimize ? "Minimize" : s == Sense::Maximize ? "Maximize" : "Fixed";
}

}
