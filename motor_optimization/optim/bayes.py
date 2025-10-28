
# optim/bayes.py
from __future__ import annotations
import math
import optuna
from models.physics import MotorInputs, solve_operating_point

def is_valid_sp(S: int, P: int) -> bool:
    if S % 3 != 0: return False
    if P % 2 != 0: return False
    g = math.gcd(S, P)
    return (S // g) % 3 == 0

def objective(trial: optuna.Trial) -> float:
    # sample with steps that satisfy first two rules
    slots = trial.suggest_int('slots', 12, 72, step=3)     # multiples of 3
    poles = trial.suggest_int('poles', 4, 40, step=2)      # even

    # prune if the gcd condition fails
    if not is_valid_sp(slots, poles):
        raise optuna.TrialPruned()  # skip invalid combinations cleanly

    Lmm  = trial.suggest_float('stack_length_mm', 40.0, 180.0)
    dOD  = trial.suggest_float('stator_od_mm', 160.0, 360.0)
    dID  = trial.suggest_float('stator_id_mm', 80.0, 300.0)
    gap  = trial.suggest_float('air_gap_mm', 0.5, 2.0)
    pmt  = trial.suggest_float('magnet_thickness_mm', 2.0, 4.0)
    wire = trial.suggest_float('wire_d_mm', 0.4, 2.0)
    turns= trial.suggest_int('turns_per_phase', 30, 220)

    mi = MotorInputs(
        slots=slots, poles=poles,
        stack_length_mm=Lmm, stator_od_mm=dOD, stator_id_mm=dID,
        air_gap_mm=gap, magnet_thickness_mm=pmt, wire_d_mm=wire,
        turns_per_phase=turns, target_torque_Nm=30.0, target_speed_rpm=300.0
    )
    res = solve_operating_point(mi)
    # maximize efficiency / cost
    return res.eff / max(1e-6, res.cost_est_usd)

def run_bayes(n_trials=60, direction='maximize'):
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
