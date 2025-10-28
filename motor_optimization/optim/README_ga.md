# PMSM Genetic Algorithm Optimization Module

A robust multi-objective optimization framework for PMSM motor design using NSGA-II (Non-dominated Sorting Genetic Algorithm II). This module finds the complete Pareto front of motor designs, revealing the trade-off between efficiency and cost.

## Overview

The `ga.py` module implements evolutionary optimization with custom operators designed specifically for motor design. Unlike single-objective methods, this approach produces a **Pareto front**: a set of equally optimal designs representing different efficiency-cost trade-offs, enabling informed engineering decisions.

## Features

- **Multi-Objective Optimization**: Simultaneously maximize efficiency AND minimize cost
- **Pareto Front Discovery**: Get 20-100 optimal design alternatives
- **Smart Slot-Pole Repair**: Automatically corrects invalid winding configurations during evolution
- **Custom Mutation Operator**: Bounded Gaussian mutations with geometric constraints
- **Robust Crossover**: Blend crossover (BLX-α) for continuous parameters
- **Design Space Constraints**: Hard bounds prevent physically impossible motors
- **NSGA-II Algorithm**: Industry-standard multi-objective evolutionary algorithm

## Installation

### Dependencies

```bash
pip install deap numpy
```

DEAP (Distributed Evolutionary Algorithms in Python) is the core framework.

## Quick Start

```python
from ga import run_ga
from viz.plots import plot_pareto, top_designs_table

# Run optimization: 30 generations × 80 population = 2400 evaluations
final_pop, pareto_front = run_ga(n_gen=30, pop_size=80)

# Visualize trade-offs
plot_pareto(pareto_front, title="NSGA-II: Efficiency vs Cost")

# Export top designs
df = top_designs_table(pareto_front, k=15)
print(df)

# Access a specific design from Pareto front
best_efficiency_design = max(pareto_front, key=lambda ind: ind.fitness.values[0])
print(f"Slots: {best_efficiency_design[0]}, Poles: {best_efficiency_design[1]}")
print(f"Efficiency: {best_efficiency_design.fitness.values[0]*100:.2f}%")
print(f"Cost: ${best_efficiency_design.fitness.values[1]:.2f}")
```

## Core Functions

### `run_ga(n_gen, pop_size)`

High-level function to execute NSGA-II optimization.

**Parameters:**
- `n_gen` (int, default=30): Number of generations to evolve
- `pop_size` (int, default=80): Population size (individuals per generation)

**Returns:**
- `tuple[list, list]`: 
  - `final_pop`: Complete final population
  - `pareto_front`: Non-dominated individuals (Pareto-optimal designs)

**Computational Cost:**
- Total evaluations ≈ `n_gen × pop_size` 
- Example: 30 gen × 80 pop = 2,400 evaluations (~5-10 minutes)

**Example:**

```python
# Quick test run (2-3 minutes)
pop, pareto = run_ga(n_gen=15, pop_size=50)

# Production run (20-30 minutes)
pop, pareto = run_ga(n_gen=100, pop_size=120)

# Analyze convergence
print(f"Total population: {len(pop)}")
print(f"Pareto front size: {len(pareto)}")
print(f"Pareto fraction: {len(pareto)/len(pop)*100:.1f}%")
```

---

### `evaluate(individual)`

Fitness function for NSGA-II. Converts design parameters to performance metrics.

**Parameters:**
- `individual` (list): Design vector `[S, P, Lmm, dOD, dID, gap, pm_t, wire_d, turns]`

**Returns:**
- `tuple[float, float]`: `(efficiency, cost_usd)`
  - Efficiency: 0.0-1.0 (NSGA-II maximizes first objective)
  - Cost: USD (NSGA-II minimizes second objective)
  - Invalid designs return `(-1e9, 1e9)` for automatic penalty

**Design Vector Structure:**

| Index | Parameter | Type | Range | Units |
|-------|-----------|------|-------|-------|
| 0 | Slots | int | 12-72 | count |
| 1 | Poles | int | 4-40 | count |
| 2 | Stack Length | float | 30-200 | mm |
| 3 | Stator OD | float | 120-400 | mm |
| 4 | Stator ID | float | 60-(OD-20) | mm |
| 5 | Air Gap | float | 0.3-2.5 | mm |
| 6 | Magnet Thickness | float | 1.0-10.0 | mm |
| 7 | Wire Diameter | float | 0.3-2.5 | mm |
| 8 | Turns per Phase | int | 20-300 | count |

**Constraint Handling:**
- Slot-pole combinations auto-repaired via `nearest_valid_sp()`
- Geometric feasibility: `ID < OD - 20 mm` (minimum wall thickness)
- Hard bounds enforced; violations return penalty fitness

---

### `is_valid_sp(S, P)`

Validates slot-pole combinations for balanced three-phase windings.

**Parameters:**
- `S` (int): Number of stator slots
- `P` (int): Number of magnetic poles

**Returns:**
- `bool`: True if configuration produces balanced windings

**Validation Criteria:**
1. **Phase Balance**: `S % 3 == 0` (three phases)
2. **Pole Pairs**: `P % 2 == 0` (poles come in N-S pairs)
3. **Winding Symmetry**: `(S // gcd(S,P)) % 3 == 0` (equal coils per phase)

**Common Valid Combinations:**

| Slots | Poles | SPP* | Winding | Notes |
|-------|-------|------|---------|-------|
| 12 | 8 | 0.5 | Fractional | Compact, low torque ripple |
| 24 | 4 | 2.0 | Integer | Simple, high torque |
| 36 | 6 | 2.0 | Integer | Balanced, low noise |
| 48 | 8 | 2.0 | Integer | High power density |
| 24 | 20 | 0.4 | Fractional | High-speed applications |

*SPP = Slots per pole = S/P

**Example:**

```python
# Test common configurations
configs = [(12,8), (24,4), (36,6), (24,10), (15,6)]
for s, p in configs:
    valid = is_valid_sp(s, p)
    print(f"{s} slots, {p} poles: {'✓ Valid' if valid else '✗ Invalid'}")
```

---

### `nearest_valid_sp(S, P, s_min, s_max, p_min, p_max)`

Repairs invalid slot-pole combinations by finding the nearest valid configuration.

**Parameters:**
- `S` (int): Current slots (possibly invalid)
- `P` (int): Current poles (possibly invalid)
- `s_min`, `s_max` (int, default=12, 72): Slot bounds
- `p_min`, `p_max` (int, default=4, 40): Pole bounds

**Returns:**
- `tuple[int, int]`: `(S_repaired, P_repaired)` guaranteed valid

**Algorithm:**
1. Snap S to nearest multiple of 3, P to nearest even number
2. Check if resulting (S, P) is valid via `is_valid_sp()`
3. If invalid, search small neighborhood (±9 slots, ±4 poles)
4. If still invalid, return random valid combination from pre-computed list

**Use Cases:**
- Mutation operator: Prevents genetic drift into invalid regions
- Initialization: Ensures all starting designs are valid
- Crossover repair: Fixes offspring with incompatible parent traits

**Example:**

```python
# Repair invalid designs
print(nearest_valid_sp(25, 7))   # → (24, 8) or (27, 6)
print(nearest_valid_sp(14, 9))   # → (15, 10) or (12, 8)

# Boundary handling
print(nearest_valid_sp(5, 3))    # → (12, 4) (snaps to minimum)
print(nearest_valid_sp(80, 50))  # → (72, 40) (snaps to maximum)
```

---

### `bounded_mutation(individual, indpb, sigma)`

Custom mutation operator with adaptive step sizes and constraint enforcement.

**Parameters:**
- `individual` (list): Design vector to mutate (modified in-place)
- `indpb` (float, default=0.25): Probability each gene mutates
- `sigma` (float, default=0.3): Mutation strength (relative to parameter range)

**Returns:**
- `tuple[list]`: Mutated individual (DEAP convention)

**Mutation Strategy:**
- **Integer parameters** (slots, poles, turns): Gaussian with σ=20×sigma
- **Float parameters** (lengths, diameters): Gaussian with σ=10×sigma
- **Post-mutation**: Clip to valid bounds, repair slot-pole combination

**Adaptive Step Sizes:**

| Parameter | σ_base | Typical Δ | Rationale |
|-----------|--------|-----------|-----------|
| Slots | 20×0.3 = 6 | ±2-3 slots | Coarse discrete steps |
| Poles | 20×0.3 = 6 | ±2 poles | Preserve pole pairs |
| Stack Length | 10×0.3 = 3 | ±1-2 mm | Fine geometric tuning |
| Wire Diameter | 10×0.3 = 3 | ±0.1 mm | Sensitive to current density |
| Turns | 20×0.3 = 6 | ±2-4 turns | Balance voltage/current |

**Example:**

```python
from deap import creator, base

# Setup (normally done in setup_toolbox)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create and mutate individual
ind = creator.Individual([24, 8, 100.0, 250.0, 150.0, 1.0, 4.0, 1.2, 80])
mutated = bounded_mutation(ind, indpb=0.5, sigma=0.5)  # Aggressive mutation

print(f"Original: {ind[:3]}")
print(f"Mutated:  {mutated[0][:3]}")
```

**Tuning Guidelines:**
- `indpb=0.1`: Conservative (fine-tuning near optima)
- `indpb=0.25`: Balanced (default, good exploration/exploitation)
- `indpb=0.5`: Aggressive (escape local minima, early generations)

---

### `setup_toolbox()`

Configures DEAP toolbox with custom operators for motor design optimization.

**Returns:**
- `deap.base.Toolbox`: Configured DEAP toolbox

**Registered Operations:**
- `individual`: Generates valid random designs
- `population`: Creates population of N individuals
- `evaluate`: Maps design → (efficiency, cost)
- `mate`: Blend crossover (BLX-α with α=0.5)
- `mutate`: Bounded mutation with slot-pole repair
- `select`: NSGA-II selection (Pareto ranking + crowding distance)

**Customization Example:**

```python
from ga import setup_toolbox
from deap import tools

toolbox = setup_toolbox()

# Override mutation strength
toolbox.register("mutate", bounded_mutation, indpb=0.3, sigma=0.4)

# Use different crossover
toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                 low=[12,4,30,120,60,0.3,1,0.3,20],
                 up=[72,40,200,400,380,2.5,10,2.5,300],
                 eta=20.0)

# Run with custom toolbox
pop = toolbox.population(n=100)
# ... continue optimization ...
```

---

## Algorithm Details

### NSGA-II Overview

**Non-dominated Sorting Genetic Algorithm II** (NSGA-II) is the gold standard for multi-objective optimization. It maintains population diversity while converging to the Pareto front.

**Key Mechanisms:**
1. **Non-dominated Sorting**: Ranks individuals into Pareto fronts (best, 2nd best, ...)
2. **Crowding Distance**: Preserves diversity by favoring spread-out solutions
3. **Elitism**: Best individuals always survive to next generation

**Why NSGA-II for Motor Design?**
- ✓ Finds entire trade-off curve (not just one "optimum")
- ✓ No need to manually weight objectives
- ✓ Robust to noisy/discontinuous fitness landscapes
- ✓ Well-tested in engineering applications

### Genetic Operators

**Selection (NSGA-II):**
```
1. Sort population by Pareto rank
2. Within same rank, sort by crowding distance
3. Keep top N individuals for next generation
```

**Crossover (Blend BLX-α):**
```
Given parents P1, P2 and parameter α=0.5:
For each gene i:
    range = |P1[i] - P2[i]|
    offspring[i] = uniform(min - α×range, max + α×range)
```

**Mutation (Bounded Gaussian):**
```
For each gene with probability indpb:
    gene += N(0, σ)
    gene = clip(gene, lower_bound, upper_bound)
Repair (S, P) to nearest valid configuration
```

### Convergence Criteria

The algorithm doesn't have automatic stopping. Monitor these metrics:

1. **Hypervolume**: Volume dominated by Pareto front (should plateau)
2. **Pareto Front Size**: Stabilizes around 5-15% of population
3. **Best Fitness Improvement**: <1% change over 10 generations → converged

---

## Advanced Usage

### Tracking Convergence

```python
from deap import tools, algorithms

toolbox = setup_toolbox()
pop = toolbox.population(n=80)

# Setup statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", lambda x: sum(f[0] for f in x)/len(x))  # Avg efficiency
stats.register("max", lambda x: max(f[0] for f in x))         # Best efficiency
stats.register("min_cost", lambda x: min(f[1] for f in x))    # Cheapest

# Track Pareto front over time
hof = tools.ParetoFront()

# Run with logging
pop, logbook = algorithms.eaMuPlusLambda(
    pop, toolbox,
    mu=80, lambda_=160,  # 80 parents, 160 offspring
    cxpb=0.7, mutpb=0.3,
    ngen=50,
    stats=stats,
    halloffame=hof,
    verbose=True
)

# Plot convergence
import matplotlib.pyplot as plt
gen = logbook.select("gen")
max_eff = logbook.select("max")
min_cost = logbook.select("min_cost")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(gen, max_eff)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Best Efficiency")
ax1.grid(True)

ax2.plot(gen, min_cost)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Minimum Cost ($)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("convergence.png")
```

### Multi-Population Island Model

```python
from deap import tools

def evolve_island(pop, toolbox, n_gen):
    """Evolve isolated population"""
    for _ in range(n_gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        pop = toolbox.select(pop + offspring, len(pop))
    return pop

# Create 4 islands
toolbox = setup_toolbox()
islands = [toolbox.population(n=50) for _ in range(4)]

# Evolve in isolation
for gen in range(30):
    islands = [evolve_island(isl, toolbox, 1) for isl in islands]
    
    # Migration every 5 generations
    if gen % 5 == 0:
        # Exchange best 5 individuals between adjacent islands
        for i in range(len(islands)):
            migrants = tools.selBest(islands[i], 5)
            next_island = islands[(i + 1) % len(islands)]
            next_island[-5:] = migrants

# Combine all islands
final_pop = sum(islands, [])
pareto = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]
```

### Constraint Handling with Penalties

```python
def evaluate_with_constraints(individual):
    """Add soft constraints via penalty functions"""
    S, P, Lmm, dOD, dID, gap, pm_t, wire_d, turns = individual
    S, P = nearest_valid_sp(int(S), int(P))
    individual[0], individual[1] = S, P
    
    # Base evaluation
    mi = MotorInputs(
        slots=S, poles=P, stack_length_mm=Lmm,
        stator_od_mm=dOD, stator_id_mm=dID,
        air_gap_mm=gap, magnet_thickness_mm=pm_t,
        wire_d_mm=wire_d, turns_per_phase=turns,
        target_torque_Nm=30.0, target_speed_rpm=300.0
    )
    res = solve_operating_point(mi)
    
    efficiency = res.eff
    cost = res.cost_est_usd
    
    # Soft constraints (prefer, but don't require)
    penalty = 0.0
    
    # Prefer copper losses < 200W
    if res.P_cu_W > 200:
        penalty += 0.05 * (res.P_cu_W - 200) / 200
    
    # Prefer core losses < 100W
    if res.P_core_W > 100:
        penalty += 0.03 * (res.P_core_W - 100) / 100
    
    # Prefer compact designs (OD < 300mm)
    if dOD > 300:
        penalty += 0.02 * (dOD - 300) / 100
    
    # Apply penalties
    efficiency -= penalty
    cost *= (1 + penalty)
    
    return (max(0, efficiency), cost)
```

---

## Parameter Tuning Guide

### Population Size (`pop_size`)

| Size | Convergence | Diversity | Computation | Use Case |
|------|-------------|-----------|-------------|----------|
| 40-60 | Fast | Low | Quick | Exploration |
| 80-120 | Balanced | Good | Medium | Production |
| 150-200 | Slow | High | Expensive | High-stakes |

**Rule of thumb**: `pop_size ≥ 10 × num_objectives × sqrt(num_variables)`
- For this problem: ≥ 10 × 2 × √9 ≈ 60

### Number of Generations (`n_gen`)

| Generations | Quality | Time | Convergence |
|-------------|---------|------|-------------|
| 15-20 | Fair | ~3 min | Partial |
| 30-50 | Good | ~8 min | Near-complete |
| 80-100 | Excellent | ~20 min | Fully converged |
| 150+ | Diminishing returns | ~40 min | Over-optimization |

**Stopping criteria**: If best fitness improves <0.5% over 10 generations, stop.

### Crossover/Mutation Rates

**Standard Settings (recommended):**
- `cxpb=0.7` (70% of offspring via crossover)
- `mutpb=0.3` (30% of offspring via mutation)

**Exploration phase (early generations):**
- `cxpb=0.5`, `mutpb=0.5` (more mutation)

**Exploitation phase (late generations):**
- `cxpb=0.9`, `mutpb=0.1` (more crossover, refine solutions)

**Adaptive approach:**
```python
for gen in range(100):
    # Decay mutation rate over time
    mutpb = 0.5 * (1 - gen/100) + 0.1  # 0.5 → 0.1
    cxpb = 1 - mutpb
    
    offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
    # ... continue ...
```

---

## Troubleshooting

### Pareto front has only 1-2 designs

**Cause:** Objectives are correlated (efficiency always decreases with cost).

**Solution:** This is actually a sign that there's no real trade-off! If you want more diversity:
```python
def evaluate_diverse(individual):
    # ... standard evaluation ...
    
    # Add diversity objective (e.g., geometric uniqueness)
    diversity_score = abs(dOD - 250) + abs(Lmm - 100)  # Prefer varied sizes
    
    return (efficiency, cost, -diversity_score)  # 3 objectives

# Update fitness weights
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))
```

### Population degenerates to identical individuals

**Cause:** Mutation rate too low or selection pressure too high.

**Solution:**
```python
# Increase mutation
toolbox.register("mutate", bounded_mutation, indpb=0.4, sigma=0.5)

# Use smaller tournament size
toolbox.register("select", tools.selTournament, tournsize=2)  # Less aggressive
```

### Optimization plateaus early

**Cause:** Premature convergence to local optimum.

**Solution 1 - Restart with best individuals:**
```python
pop1, pareto1 = run_ga(n_gen=30, pop_size=80)

# Seed new population with Pareto front
toolbox = setup_toolbox()
pop2 = list(pareto1) + toolbox.population(n=60)  # 20 elite + 60 random

# Continue evolution
for gen in range(30):
    # ... evolve pop2 ...
```

**Solution 2 - Increase diversity:**
```python
# Larger population
pop, pareto = run_ga(n_gen=30, pop_size=150)

# Or more mutation
def aggressive_mutate(ind):
    return bounded_mutation(ind, indpb=0.5, sigma=0.6)

toolbox.register("mutate", aggressive_mutate)
```

---

## Comparison with Other Optimizers

| Method | Time | Pareto Front | Best Use |
|--------|------|--------------|----------|
| **GA (NSGA-II)** | 10-30 min | Yes (20-50 designs) | Trade-off analysis |
| **Bayesian (TPE)** | 3-10 min | No (single best) | Quick optimization |
| **Random Search** | 1 min | No | Baseline |
| **Grid Search** | Hours | No | Exhaustive |
| **PSO** | 10-20 min | Limited | Continuous-only |

**When to use GA:**
- You need the full Pareto front (not just one solution)
- You have 15-30 minutes for optimization
- You want robustness to discrete + continuous parameters
- Multi-objective is essential

---

## Example Workflow

```python
# complete_ga_workflow.py
from ga import run_ga, evaluate
from viz.plots import plot_pareto, plot_parallel_coords, top_designs_table
from models.physics import MotorInputs, solve_operating_point
import pickle

# ============ Phase 1: Evolution ============
print("=== Running NSGA-II Optimization ===")
print("Generations: 50, Population: 100")
print("Estimated time: ~15 minutes\n")

final_pop, pareto_front = run_ga(n_gen=50, pop_size=100)

# Save results
with open("optimization_results.pkl", "wb") as f:
    pickle.dump({"population": final_pop, "pareto": pareto_front}, f)

print(f"\nOptimization complete!")
print(f"Final population: {len(final_pop)} individuals")
print(f"Pareto front: {len(pareto_front)} optimal designs")

# ============ Phase 2: Analysis ============
print("\n=== Analyzing Results ===")

# Visualize Pareto front
plot_pareto(pareto_front, title="50-Gen NSGA-II Optimization")
plot_parallel_coords(pareto_front)

# Extract top designs
df_top = top_designs_table(
    pareto_front,
    k=20,
    eff_floor=0.95,
    save_path="pareto_designs.csv"
)

# ============ Phase 3: Decision Support ============
print("\n=== Design Recommendations ===\n")

# High-efficiency design (cost-insensitive)
high_eff = max(pareto_front, key=lambda x: x.fitness.values[0])
mi_eff = MotorInputs(
    slots=int(high_eff[0]), poles=int(high_eff[1]),
    stack_length_mm=high_eff[2], stator_od_mm=high_eff[3],
    stator_id_mm=high_eff[4], air_gap_mm=high_eff[5],
    magnet_thickness_mm=high_eff[6], wire_d_mm=high_eff[7],
    turns_per_phase=int(high_eff[8]),
    target_torque_Nm=30.0, target_speed_rpm=300.0
)
res_eff = solve_operating_point(mi_eff)

print("🏆 HIGHEST EFFICIENCY DESIGN:")
print(f"  Geometry: {int(high_eff[0])} slots, {int(high_eff[1])} poles")
print(f"  Efficiency: {res_eff.eff*100:.2f}%")
print(f"  Cost: ${res_eff.cost_est_usd:.2f}")
print(f"  Losses: Cu={res_eff.P_cu_W:.1f}W, Core={res_eff.P_core_W:.1f}W\n")

# Low-cost design (budget-constrained)
low_cost = min(pareto_front, key=lambda x: x.fitness.values[1])
mi_cost = MotorInputs(
    slots=int(low_cost[0]), poles=int(low_cost[1]),
    stack_length_mm=low_cost[2], stator_od_mm=low_cost[3],
    stator_id_mm=low_cost[4], air_gap_mm=low_cost[5],
    magnet_thickness_mm=low_cost[6], wire_d_mm=low_cost[7],
    turns_per_phase=int(low_cost[8]),
    target_torque_Nm=30.0, target_speed_rpm=300.0
)
res_cost = solve_operating_point(mi_cost)

print("💰 LOWEST COST DESIGN:")
print(f"  Geometry: {int(low_cost[0])} slots, {int(low_cost[1])} poles")
print(f"  Efficiency: {res_cost.eff*100:.2f}%")
print(f"  Cost: ${res_cost.cost_est_usd:.2f}")
print(f"  Losses: Cu={res_cost.P_cu_W:.1f}W, Core={res_cost.P_core_W:.1f}W\n")

# Balanced design (minimize distance to ideal point)
import numpy as np
effs = [ind.fitness.values[0] for ind in pareto_front]
costs = [ind.fitness.values[1] for ind in pareto_front]
eff_norm = (np.array(effs) - min(effs)) / (max(effs) - min(effs))
cost_norm = (max(costs) - np.array(costs)) / (max(costs) - min(costs))
distances = np.sqrt((1-eff_norm)**2 + (1-cost_norm)**2)
balanced = pareto_front[np.argmin(distances)]

mi_bal = MotorInputs(
    slots=int(balanced[0]), poles=int(balanced[1]),
    stack_length_mm=balanced[2], stator_od_mm=balanced[3],
    stator_id_mm=balanced[4], air_gap_mm=balanced[5],
    magnet_thickness_mm=balanced[6], wire_d_mm=balanced[7],
    turns_per_phase=int(balanced[8]),
    target_torque_Nm=30.0, target_speed_rpm=300.0
)
res_bal = solve_operating_point(mi_bal)

print("⚖️ BALANCED DESIGN (Best Trade-off):")
print(f"  Geometry: {int(balanced[0])} slots, {int(balanced[1])} poles")
print(f"  Efficiency: {res_bal.eff*100:.2f}%")
print(f"  Cost: ${res_bal.cost_est_usd:.2f}")
print(f"  Losses: Cu={res_bal.P_cu_W:.1f}W, Core={res_bal.P_core_W:.1f}W\n")

print(f"All results saved to: pareto_designs.csv")
print(f"Visualizations saved to: _figures/")
```

---

## License

GNU Lesser General Public License v3.0 (LGPL v3)

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this library; if not, see <https://www.gnu.org/licenses/>.

---

## Contact

**Power Electronics Group**

Website: [www.powerelectronicsgroup.com](https://www.powerelectronicsgroup.com)  
Email: info@powerelectronicsgroup.com  
Phone: USA +1 571 781 2453

For technical support, feature requests, or collaboration opportunities, please reach out via email or visit our website.

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Maintainer**: Power Electronics Group
