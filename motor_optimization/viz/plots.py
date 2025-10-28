# viz/plots.py
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import math

def _is_valid_sp(S: int, P: int) -> bool:
    if S <= 0 or P <= 0:
        return False
    if S % 3 != 0:
        return False
    if P % 2 != 0:
        return False
    g = math.gcd(S, P)
    return (S // g) % 3 == 0


# Optional GUI backend. If you don't need windows, you can comment this out.
import matplotlib
# matplotlib.use("QtAgg")  # or "TkAgg" / remove line to use default
matplotlib.use("Agg")  # changed from QtAgg to Agg
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

__all__ = [
    "plot_pareto",
    "plot_parallel_coords",
    "bar_model_scores",
    "plot_torque_eff_cost",
    "top_designs_table",
]

# --------- helpers for saving ----------
_OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_figures"))

def _ensure_out_dir() -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)

def _save_current_fig(basename: str) -> str:
    """Save current plt figure to _figures with a timestamp suffix; return file path."""
    _ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{basename}_{ts}.png"
    fpath = os.path.join(_OUT_DIR, fname)
    plt.gcf().savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"[saved] {fpath}")
    return fpath


# -----------------------------
# Pareto scatter: Efficiency vs Cost
# -----------------------------
def plot_pareto(pareto, title: str = "Pareto: Efficiency vs Cost") -> None:
    # Split valid/invalid by slot–pole rules
    val_inds, bad_inds = [], []
    for ind in pareto:
        try:
            S, P = int(ind[0]), int(ind[1])
            (val_inds if _is_valid_sp(S, P) else bad_inds).append(ind)
        except Exception:
            bad_inds.append(ind)

    if not val_inds and not bad_inds:
        print("No individuals to plot.")
        return

    plt.figure(figsize=(6, 4))

    if val_inds:
        eff_v = [i.fitness.values[0] for i in val_inds if getattr(i.fitness, "valid", False) and len(i.fitness.values) >= 2]
        cost_v = [i.fitness.values[1] for i in val_inds if getattr(i.fitness, "valid", False) and len(i.fitness.values) >= 2]
        if eff_v and cost_v:
            plt.scatter(eff_v, cost_v, alpha=0.85, label="Valid (S,P)", zorder=2)

    if bad_inds:
        eff_b = [i.fitness.values[0] for i in bad_inds if getattr(i.fitness, "valid", False) and len(i.fitness.values) >= 2]
        cost_b = [i.fitness.values[1] for i in bad_inds if getattr(i.fitness, "valid", False) and len(i.fitness.values) >= 2]
        if eff_b and cost_b:
            plt.scatter(eff_b, cost_b, marker="x", s=70, linewidths=1.5, color="red",
                        alpha=0.9, label="INVALID (S,P)", zorder=3)

    ttl = title if not bad_inds else f"{title}  —  invalid combos: {len(bad_inds)}"
    plt.xlabel("Efficiency"); plt.ylabel("Cost (USD)")
    plt.title(ttl)
    plt.grid(True, alpha=0.3)
    if val_inds or bad_inds:
        plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    _save_current_fig("pareto_eff_vs_cost")
    plt.show()


# -----------------------------
# Parallel coordinates across design variables
# -----------------------------
def plot_parallel_coords(pareto) -> None:
    valid = []
    bad_count = 0
    for ind in pareto:
        try:
            S, P = int(ind[0]), int(ind[1])
            if _is_valid_sp(S, P):
                valid.append(ind)
            else:
                bad_count += 1
        except Exception:
            bad_count += 1

    if not valid:
        print("No valid individuals for parallel coordinates.")
        return

    df = pd.DataFrame([{
        "Slots": int(ind[0]),
        "Poles": int(ind[1]),
        "L(mm)": float(ind[2]),
        "OD(mm)": float(ind[3]),
        "ID(mm)": float(ind[4]),
        "Gap(mm)": float(ind[5]),
        "PMt(mm)": float(ind[6]),
        "Wire(mm)": float(ind[7]),
        "Turns": int(ind[8]),
        "Eff": float(ind.fitness.values[0]),
    } for ind in valid if getattr(ind.fitness, "valid", False) and len(ind.fitness.values) >= 2])

    q = min(5, max(3, len(df["Eff"].unique())))
    bands = pd.qcut(df["Eff"], q=q, duplicates="drop").astype(str)
    df["EffBand"] = bands

    plt.figure(figsize=(10, 5))
    parallel_coordinates(
        df[["Slots", "Poles", "L(mm)", "OD(mm)", "ID(mm)", "Gap(mm)", "PMt(mm)", "Wire(mm)", "Turns", "EffBand"]],
        "EffBand",
        alpha=0.6,
    )
    title = "Design Space — Parallel Coordinates"
    if bad_count:
        title += f"  (ignored invalid combos: {bad_count})"
    plt.title(title)
    plt.xticks(rotation=25)
    plt.tight_layout()
    _save_current_fig("parallel_coords")
    plt.show()

# -----------------------------
# Model score bars
# -----------------------------
def bar_model_scores(results: dict) -> None:
    names = list(results.keys())
    maes = [results[n][0] for n in names]

    plt.figure(figsize=(6, 4))
    plt.bar(names, maes, alpha=0.8)
    plt.ylabel("CV MAE")
    plt.title("Model Comparison")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_current_fig("model_scores")
    plt.show()


# -----------------------------
# Torque–Efficiency–Cost scatter (2D + color)
# -----------------------------
def plot_torque_eff_cost(pareto) -> None:
    val_inds, bad_inds = [], []
    for ind in pareto:
        try:
            S, P = int(ind[0]), int(ind[1])
            (val_inds if _is_valid_sp(S, P) else bad_inds).append(ind)
        except Exception:
            bad_inds.append(ind)

    if not val_inds and not bad_inds:
        print("No valid designs to plot.")
        return

    # Valid
    if val_inds:
        df_v = pd.DataFrame([{
            "Torque_Nm": 30.0,
            "Efficiency": float(ind.fitness.values[0]),
            "Cost_USD": float(ind.fitness.values[1]),
        } for ind in val_inds if getattr(ind.fitness, "valid", False) and len(ind.fitness.values) >= 2])

        plt.figure(figsize=(6, 4))
        sc = plt.scatter(df_v["Torque_Nm"], df_v["Efficiency"], c=df_v["Cost_USD"],
                         cmap="viridis", s=60, alpha=0.85, edgecolor="k", label="Valid (S,P)", zorder=2)
        plt.colorbar(sc, label="Cost (USD)")

    # Invalid overlay
    if bad_inds:
        df_b = pd.DataFrame([{
            "Torque_Nm": 30.0,
            "Efficiency": float(ind.fitness.values[0]),
            "Cost_USD": float(ind.fitness.values[1]),
        } for ind in bad_inds if getattr(ind.fitness, "valid", False) and len(ind.fitness.values) >= 2])
        if not val_inds:
            plt.figure(figsize=(6, 4))
        plt.scatter(df_b["Torque_Nm"], df_b["Efficiency"], marker="x", s=70, linewidths=1.5,
                    color="red", alpha=0.9, label="INVALID (S,P)", zorder=3)

    title = "Torque vs. Efficiency vs. Cost"
    if bad_inds:
        title += f"  —  invalid combos: {len(bad_inds)}"
    plt.xlabel("Torque (Nm)"); plt.ylabel("Efficiency"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.legend(loc="best", frameon=True)
    _save_current_fig("torque_eff_cost")
    plt.show()


# -----------------------------
# Tabulate top designs near ideal (high eff, low cost)
# -----------------------------
def top_designs_table(
    pareto,
    k: int = 15,
    eff_floor: float = 0.95,
    save_path: str = "_figures/top_designs.csv",
    cheap_quantile: float | None = None,   # e.g., 0.5 to keep cheapest half
    cost_weight: float = 1.0,              # emphasize cost in the distance metric
):
    """
    Select k best designs that are close to max efficiency and low cost.
    - Normalizes both metrics so bigger = better for both.
    - Applies an efficiency floor (fraction of the range from min to max).
    - Optional: filter to the cheapest quantile (cheap_quantile).
    - Ranks by weighted Euclidean distance to the ideal point (1, 1).
    Saves CSV and returns a DataFrame.
    """
    _ensure_out_dir()

    # 1) Keep only valid individuals (have both objectives)
    valid = [
        ind for ind in pareto
        if getattr(ind.fitness, "valid", False) and len(getattr(ind.fitness, "values", ())) >= 2
    ]
    if not valid:
        print("No valid designs to tabulate.")
        return None

    # 2) Build raw table
    rows = []
    for ind in valid:
        eff, cost = ind.fitness.values
        rows.append({
            "Slots": int(ind[0]),
            "Poles": int(ind[1]),
            "L_mm": float(ind[2]),
            "OD_mm": float(ind[3]),
            "ID_mm": float(ind[4]),
            "Gap_mm": float(ind[5]),
            "PMt_mm": float(ind[6]),
            "Wire_mm": float(ind[7]),
            "Turns": int(ind[8]),
            "Efficiency": float(eff),
            "Cost_USD": float(cost),
        })
    df = pd.DataFrame(rows)

    # 3) Normalize: bigger = better for both
    eff_min, eff_max = df["Efficiency"].min(), df["Efficiency"].max()
    cost_min, cost_max = df["Cost_USD"].min(), df["Cost_USD"].max()
    eff_span = (eff_max - eff_min) or 1.0
    cost_span = (cost_max - cost_min) or 1.0

    df["eff_norm"] = (df["Efficiency"] - eff_min) / eff_span
    df["cost_norm_good"] = (cost_max - df["Cost_USD"]) / cost_span  # higher is cheaper

    # 4) Soft filter: keep designs near the best efficiency (e.g., ≥ 95% of max)
    eff_threshold = eff_min + eff_floor * eff_span
    df_filt = df[df["Efficiency"] >= eff_threshold].copy()
    if df_filt.empty:
        df_filt = df.copy()

    # Optional: keep only cheapest quantile
    if cheap_quantile is not None and 0.0 < cheap_quantile < 1.0:
        max_cost_allowed = df_filt["Cost_USD"].quantile(cheap_quantile)
        df_filt = df_filt[df_filt["Cost_USD"] <= max_cost_allowed].copy()
        if df_filt.empty:
            df_filt = df.copy()

    # 5) Rank by (weighted) distance to the ideal (1, 1)
    w_eff, w_cost = 1.0, float(cost_weight)
    df_filt["distance_to_ideal"] = np.sqrt(
        (w_eff * (1.0 - df_filt["eff_norm"]))**2 +
        (w_cost * (1.0 - df_filt["cost_norm_good"]))**2
    )

    df_top = df_filt.sort_values(["distance_to_ideal", "Cost_USD"]).head(k)

    # 6) Reorder columns for presentation
    cols = [
        "Efficiency", "Cost_USD",
        "Slots", "Poles", "L_mm", "OD_mm", "ID_mm", "Gap_mm", "PMt_mm", "Wire_mm", "Turns",
        "eff_norm", "cost_norm_good", "distance_to_ideal",
    ]
    df_top = df_top[cols]

    # 7) Save & print CSV
    csv_path = os.path.join(_OUT_DIR, os.path.basename(save_path))
    df_top.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")
    print(df_top.to_string(index=False))

    return df_top
