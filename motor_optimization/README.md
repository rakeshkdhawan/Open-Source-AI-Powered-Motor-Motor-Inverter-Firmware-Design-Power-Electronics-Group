
# Motor Optimization — Modular (VS / VS Code Friendly)

This project splits the notebook into modules and wires the optimization to **somewhat real PMSM equations** (simplified analytical). It’s suitable for Visual Studio or VS Code.

## Modules
- `models/physics.py` — PMSM physics: flux linkage, inductance, copper/core/mech losses, efficiency, mass & cost estimates
- `optim/ga.py` — NSGA-II multi-objective GA (maximize efficiency, minimize cost)
- `optim/bayes.py` — Optuna Bayesian optimization (maximize eff/cost)
- `ml/model_compare.py` — quick ML comparison on synthetic data
- `viz/plots.py` — Pareto & parallel-coordinates plots; model score bars
- `main.py` — script entry point

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Notes
- Physics is **surface PMSM** oriented with Id=0 control default.
- Loss model includes copper (temp-adjusted), core (Steinmetz-like), and mechanical (quadratic with speed).
- Material masses and cost are coarse; tune coefficients in `physics.py` for alloys/magnets.
- All parameters & bounds can be narrowed for any specific geometry.
