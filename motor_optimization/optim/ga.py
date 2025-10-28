# optim/ga.py
from __future__ import annotations
import random, math
from typing import Tuple
from deap import base, creator, tools, algorithms
from models.physics import MotorInputs, solve_operating_point

# ---------- slot–pole validity helpers ----------
def is_valid_sp(S: int, P: int) -> bool:
    if S <= 0 or P <= 0: 
        return False
    if S % 3 != 0: 
        return False
    if P % 2 != 0: 
        return False
    g = math.gcd(S, P)
    return (S // g) % 3 == 0

def nearest_valid_sp(S: int, P: int, s_min=12, s_max=72, p_min=4, p_max=40) -> tuple[int,int]:
    """Snap/repair to a nearby valid (S,P). Try a small search around current values."""
    # Snap basic divisibility first
    if S % 3: S += (3 - (S % 3))
    if P % 2: P += 1
    S = min(max(S, s_min), s_max)
    P = min(max(P, p_min), p_max)

    if is_valid_sp(S, P): 
        return S, P

    # Small neighborhood search for a valid pair
    for dS in range(0, 10, 3):         # try +0, +3, +6, +9
        for dP in (0, 2, -2, 4, -4):   # even adjustments
            SS = min(max(S + dS, s_min), s_max)
            PP = min(max(P + dP, p_min), p_max)
            if SS % 3 == 0 and PP % 2 == 0 and is_valid_sp(SS, PP):
                return SS, PP

    # Fallback: random valid within bounds
    candidates = [(s, p)
                  for s in range(s_min + (-s_min % 3), s_max + 1, 3)
                  for p in range(p_min + (p_min % 2), p_max + 1, 2)
                  if is_valid_sp(s, p)]
    return random.choice(candidates) if candidates else (S, P)

# ---------- DEAP setup ----------
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate(individual) -> Tuple[float, float]:
    S, P, Lmm, dOD, dID, gap, pm_t, wire_d, turns = individual
    # Repair S,P up front to avoid wasting evaluations
    S, P = nearest_valid_sp(int(S), int(P))
    individual[0], individual[1] = S, P

    try:
        turns = int(turns)
        Lmm = float(Lmm); dOD=float(dOD); dID=float(dID)
        gap=float(gap); pm_t=float(pm_t); wire_d=float(wire_d)
    except Exception:
        return (-1e9, 1e9)

    # Hard bounds (and ID < OD, etc.)
    if not (12 <= S <= 72 and 4 <= P <= 40 and 30.0 <= Lmm <= 200.0 and
            120.0 <= dOD <= 400.0 and 60.0 <= dID <= dOD - 20.0 and
            0.3 <= gap <= 2.5 and 1.0 <= pm_t <= 10.0 and 0.3 <= wire_d <= 2.5 and
            20 <= turns <= 300 and is_valid_sp(S, P)):
        return (-1e9, 1e9)

    mi = MotorInputs(
        slots=S, poles=P,
        stack_length_mm=Lmm, stator_od_mm=dOD, stator_id_mm=dID,
        air_gap_mm=gap, magnet_thickness_mm=pm_t, wire_d_mm=wire_d,
        turns_per_phase=turns, target_torque_Nm=30.0, target_speed_rpm=300.0
    )
    res = solve_operating_point(mi)
    return (res.eff, res.cost_est_usd)

def bounded_mutation(individual, indpb: float = 0.25, sigma: float = 0.3):
    # Gaussian nudges + clipping
    for i in range(len(individual)):
        if random.random() < indpb:
            if i in (0, 1, 8):  # slots, poles, turns
                individual[i] = int(round(individual[i] + random.gauss(0, sigma * 20)))
            else:
                individual[i] = float(individual[i] + random.gauss(0, sigma * 10))

    # Clip scalar bounds
    individual[2] = float(min(max(individual[2], 30.0), 200.0))   # Lmm
    individual[3] = float(min(max(individual[3], 120.0), 400.0))  # dOD
    individual[4] = float(min(max(individual[4], 60.0), max(80.0, individual[3] - 20.0)))  # dID
    individual[5] = float(min(max(individual[5], 0.3), 2.5))      # gap
    individual[6] = float(min(max(individual[6], 1.0), 10.0))     # pm_t
    individual[7] = float(min(max(individual[7], 0.3), 2.5))      # wire_d
    individual[8] = int(min(max(individual[8], 20), 300))         # turns

    # Repair slots/poles to nearest valid
    S = int(min(max(individual[0], 12), 72))
    P = int(min(max(individual[1], 4), 40))
    individual[0], individual[1] = nearest_valid_sp(S, P)
    return (individual,)

def setup_toolbox():
    toolbox = base.Toolbox()

    # Generate only valid (S,P)
    def rand_slots():
        s = random.randrange(12, 73, 3)   # multiples of 3
        return s
    def rand_poles():
        p = random.randrange(4, 41, 2)    # even
        return p

    toolbox.register("attr_slots", rand_slots)
    toolbox.register("attr_poles", rand_poles)
    toolbox.register("attr_Lmm", random.uniform, 40.0, 180.0)
    toolbox.register("attr_dOD", random.uniform, 160.0, 360.0)
    toolbox.register("attr_dID", random.uniform, 80.0, 300.0)
    toolbox.register("attr_gap", random.uniform, 0.4, 2.0)
    toolbox.register("attr_pmt", random.uniform, 2.0, 8.0)
    toolbox.register("attr_wire", random.uniform, 0.4, 2.0)
    toolbox.register("attr_turns", random.randint, 30, 220)

    def make_individual():
        s, p = rand_slots(), rand_poles()
        s, p = nearest_valid_sp(s, p)
        return creator.Individual([
            s, p,
            toolbox.attr_Lmm(),
            toolbox.attr_dOD(),
            toolbox.attr_dID(),
            toolbox.attr_gap(),
            toolbox.attr_pmt(),
            toolbox.attr_wire(),
            toolbox.attr_turns(),
        ])

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", bounded_mutation, indpb=0.25, sigma=0.3)
    toolbox.register("select", tools.selNSGA2)
    return toolbox

def run_ga(n_gen: int = 30, pop_size: int = 80):
    toolbox = setup_toolbox()

    # initial population + evaluation
    pop = toolbox.population(n=pop_size)
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    pop = tools.selNSGA2(pop, len(pop))
    for _ in range(1, n_gen + 1):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        pop = toolbox.select(pop + offspring, pop_size)

    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    return pop, pareto
