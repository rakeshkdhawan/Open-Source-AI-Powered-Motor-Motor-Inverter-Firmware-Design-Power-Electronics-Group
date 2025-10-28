# main.py
from __future__ import annotations
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Ensure local packages (optim, viz, models, ml) are importable when running this file directly
sys.path.append(os.path.dirname(__file__))

from optim.ga import run_ga
from optim.bayes import run_bayes
from ml.model_compare import compare_models
from viz.plots import (
    plot_pareto,
    plot_parallel_coords,
    bar_model_scores,
    plot_torque_eff_cost,
    top_designs_table,
)


def main():
    print("== Motor Optimization (Modular) ==")

    # 1) Bayesian Optimization (quick single-objective scan)
    try:
        study = run_bayes(n_trials=40)
        print("Best Bayesian params:", study.best_params)
        print("Best Bayesian score:", study.best_value)
    except Exception as e:
        print("[WARN] Bayesian optimization skipped due to error:", e)
        study = None

    # 2) Genetic Algorithm (multi-objective: efficiency ↑, cost ↓)
    #    Returns full population and the Pareto front (first nondominated set).
    pop, pareto = run_ga(n_gen=25, pop_size=80)
    print("Pareto solutions:", len(pareto))

    # 3) ML model comparison demo (synthetic)
    results = compare_models()
    for name, (mae, std) in results.items():
        print(f"{name}: MAE={mae:.4f} (+/- {std:.4f})")

    # 4) Visualizations
    #    If your matplotlib backend can't show windows, change to saving-only in viz/plots.py
    plot_pareto(pareto)
    plot_parallel_coords(pareto)
    bar_model_scores(results)
    plot_torque_eff_cost(pareto)

    # 5) Tabulate top ~10–15 designs that are high-efficiency and low-cost
    #    - eff_floor=0.95 keeps designs within the top 5% of the efficiency range
    #    - cheap_quantile=0.5 keeps cheapest half before ranking (optional)
    #    - cost_weight=2.0 emphasizes cost in the ranking (optional)
    top_df = top_designs_table(
        pareto,
        k=15,
        eff_floor=0.95,
        cheap_quantile=0.5,
        cost_weight=2.0,
    )

    # Optional: do something with the returned DataFrame (already saved to _figures/)
    if top_df is not None:
        print("\nTop designs table saved to _figures/top_designs.csv")

    print("\n✓ Complete.")


if __name__ == "__main__":
    main()
    # Old: print("\n✓ Complete.")
    print("COmplete")
