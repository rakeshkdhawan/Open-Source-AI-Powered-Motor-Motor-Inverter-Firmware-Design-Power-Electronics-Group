
# models/physics.py
# Simplified PMSM (surface-mounted) motor equations with practical approximations.
# These are not FEA-accurate but tie to real relationships for optimization demos.
from __future__ import annotations
import math
from dataclasses import dataclass

MU0 = 4 * math.pi * 1e-7           # vacuum permeability (H/m)
RHO_CU_20C = 1.724e-8              # copper resistivity at 20°C (Ohm·m)
TEMP_COEFF_CU = 0.00393            # copper temperature coeff (1/°C)
RHO_CU = 8960.0                    # kg/m3
RHO_PM = 7500.0                    # NdFeB density (kg/m3) ~7.5 g/cm3
BR_N35 = 1.2                       # Tesla, representative remanence for N35–N42
ETA_MECH_FRIC = 1e-6               # mechanical loss constant (N·m·s^2) rough
KH = 40.0                          # Steinmetz hysteresis coefficient (W/m^3/T^β·Hz) rough
KE = 0.2                           # Eddy loss coefficient (W/m^3/T^2·Hz^2) rough
BETA = 1.6                         # Steinmetz exponent

@dataclass
class MotorInputs:
    slots: int
    poles: int                  # even
    stack_length_mm: float      # axial stack length (mm)
    stator_od_mm: float         # stator outer diameter (mm)
    stator_id_mm: float         # stator inner diameter (mm)
    air_gap_mm: float
    magnet_thickness_mm: float
    wire_d_mm: float            # conductor diameter (mm) (single in-hand)
    turns_per_phase: int
    phase_count: int = 3
    target_torque_Nm: float = 30.0
    target_speed_rpm: float = 300.0
    winding_temp_C: float = 80.0
    tooth_fill_k: float = 1.1    # MLt fudge (end-windings etc.)
    tooth_w_ratio: float = 0.45  # tooth width / slot pitch
    pole_arc_ratio: float = 0.8  # magnet arc / pole pitch
    k_leak: float = 1.2          # leakage factor in L and flux
    k_lambda: float = 0.9        # flux linking fraction
    k_B: float = 0.85            # B-field utilization from Br
    core_loss_k: float = 1.0

@dataclass
class MotorResults:
    lambda_Vs_per_rad: float
    Ld_H: float
    Lq_H: float
    R_phase_ohm: float
    Iq_A: float
    Id_A: float
    torque_Nm: float
    omega_rad_s: float
    P_out_W: float
    P_cu_W: float
    P_core_W: float
    P_mech_W: float
    P_in_W: float
    eff: float
    copper_mass_kg: float
    magnet_mass_kg: float
    cost_est_usd: float

def mm_to_m(x): return x * 1e-3

def circle_length(r): return 2*math.pi*r

def wire_area_m2(d_mm: float) -> float:
    r = mm_to_m(d_mm)/2.0
    return math.pi * r * r

def copper_resistance_ohm(length_m: float, area_m2: float, temp_C: float) -> float:
    rho_T = RHO_CU_20C * (1 + TEMP_COEFF_CU * (temp_C - 20.0))
    return rho_T * length_m / area_m2

def mean_radii(stator_od_m: float, stator_id_m: float):
    r_tooth = (stator_od_m/2.0 + stator_id_m/2.0)/2.0
    r_slot_pitch = r_tooth
    return r_tooth, r_slot_pitch

def pole_pitch(r_m: float, poles: int) -> float:
    return circle_length(r_m) / poles

def slot_pitch(r_m: float, slots: int) -> float:
    return circle_length(r_m) / slots

def lamination_area_tooth(stack_len_m: float, slot_pitch_m: float, tooth_w_ratio: float) -> float:
    tooth_w = tooth_w_ratio * slot_pitch_m
    return tooth_w * stack_len_m

def eff_air_gap(air_gap_m: float, magnet_thickness_m: float, k_leak: float) -> float:
    # empirical effective magnetic length (accounts for leakage/reluctance)
    return k_leak * (air_gap_m + 0.5e-3)  # add Carter/slots ~0.5 mm equivalent

def flux_linkage_Vs_per_rad(inputs: MotorInputs) -> float:
    # λ ≈ k_lambda * B * A_pole * N / (2πr)   (very rough)
    r_tooth, _ = mean_radii(mm_to_m(inputs.stator_od_mm), mm_to_m(inputs.stator_id_mm))
    l_stack = mm_to_m(inputs.stack_length_mm)
    pp = pole_pitch(r_tooth, inputs.poles)
    pole_arc = inputs.pole_arc_ratio * pp
    B = inputs.k_B * BR_N35 * (mm_to_m(inputs.magnet_thickness_mm) / eff_air_gap(mm_to_m(inputs.air_gap_mm), mm_to_m(inputs.magnet_thickness_mm), inputs.k_leak))
    A_pole = pole_arc * l_stack
    N = inputs.turns_per_phase
    lam = inputs.k_lambda * B * A_pole * N / max(1e-6, (2*math.pi*r_tooth))
    # scale to Vs/rad (already flux linkage, compatible with torque eq using SI)
    return lam

def inductances_H(inputs: MotorInputs) -> tuple[float,float]:
    # Ld ≈ Lq ≈ μ0 * N^2 * A_tooth / g_eff (surface PMSM, salient=small)
    r_tooth, r_pitch = mean_radii(mm_to_m(inputs.stator_od_mm), mm_to_m(inputs.stator_id_mm))
    sp = slot_pitch(r_pitch, inputs.slots)
    A_tooth = lamination_area_tooth(mm_to_m(inputs.stack_length_mm), sp, inputs.tooth_w_ratio)
    g_eff = eff_air_gap(mm_to_m(inputs.air_gap_mm), mm_to_m(inputs.magnet_thickness_mm), inputs.k_leak)
    N = inputs.turns_per_phase
    L = MU0 * (N**2) * A_tooth / max(1e-6, g_eff)
    return L, L  # surface PMSM: Ld≈Lq

def copper_length_m(inputs: MotorInputs) -> float:
    # Mean length per turn ≈ 2πr*k + end-winding (2x stack length)
    r_tooth, _ = mean_radii(mm_to_m(inputs.stator_od_mm), mm_to_m(inputs.stator_id_mm))
    mlt = circle_length(r_tooth) * inputs.tooth_fill_k + 2.0 * mm_to_m(inputs.stack_length_mm)
    return inputs.phase_count * inputs.turns_per_phase * mlt

def copper_mass_kg(inputs: MotorInputs) -> float:
    A = wire_area_m2(inputs.wire_d_mm)
    L = copper_length_m(inputs)
    vol = A * L
    return vol * RHO_CU

def magnet_mass_kg(inputs: MotorInputs) -> float:
    r_tooth, _ = mean_radii(mm_to_m(inputs.stator_od_mm), mm_to_m(inputs.stator_id_mm))
    pp = pole_pitch(r_tooth, inputs.poles)
    l_stack = mm_to_m(inputs.stack_length_mm)
    arc = inputs.pole_arc_ratio * pp
    vol = arc * l_stack * mm_to_m(inputs.magnet_thickness_mm) * inputs.poles
    return vol * RHO_PM

def core_loss_W(inputs: MotorInputs, B_pk: float, f_elec_Hz: float) -> float:
    # Steinmetz-like Pc = kh * f * B^β + ke * f^2 * B^2, scaled by active core volume
    r_tooth, _ = mean_radii(mm_to_m(inputs.stator_od_mm), mm_to_m(inputs.stator_id_mm))
    vol_core = math.pi*(mm_to_m(inputs.stator_od_mm/2.0)**2 - mm_to_m(inputs.stator_id_mm/2.0)**2) * mm_to_m(inputs.stack_length_mm) * 2.0
    pc = inputs.core_loss_k * (KH * f_elec_Hz * (B_pk**BETA) + KE * (f_elec_Hz**2) * (B_pk**2))
    return max(0.0, pc * max(1e-6, vol_core))

def mech_loss_W(omega_rad_s: float) -> float:
    return ETA_MECH_FRIC * (omega_rad_s**2)

def solve_operating_point(inputs: MotorInputs) -> MotorResults:
    # Derived operating variables
    omega = inputs.target_speed_rpm * 2.0*math.pi/60.0
    pole_pairs = inputs.poles / 2.0
    f_elec = pole_pairs * omega / (2.0*math.pi)

    lam = flux_linkage_Vs_per_rad(inputs)
    Ld, Lq = inductances_H(inputs)

    # Surface PMSM torque (Id=0 control as default):
    # T = (3/2) * p * lam * Iq
    Iq = inputs.target_torque_Nm / max(1e-9, ((3.0/2.0) * pole_pairs * lam))
    Id = 0.0

    # Copper loss
    A = wire_area_m2(inputs.wire_d_mm)
    L_cu = copper_length_m(inputs)
    R_phase_20C = copper_resistance_ohm(L_cu/inputs.phase_count, A, 20.0)
    R_phase = copper_resistance_ohm(L_cu/inputs.phase_count, A, inputs.winding_temp_C)
    I_phase_rms = abs(Iq)/math.sqrt(2.0)  # Id=0, sinusoidal
    P_cu = inputs.phase_count * (I_phase_rms**2) * R_phase

    # Estimate B field for core losses
    r_tooth, _ = mean_radii(mm_to_m(inputs.stator_od_mm), mm_to_m(inputs.stator_id_mm))
    pp = pole_pitch(r_tooth, inputs.poles)
    g_eff = eff_air_gap(mm_to_m(inputs.air_gap_mm), mm_to_m(inputs.magnet_thickness_mm), inputs.k_leak)
    B_pk = min(1.6, inputs.k_B * BR_N35 * (mm_to_m(inputs.magnet_thickness_mm) / max(1e-6, g_eff)))
    P_core = core_loss_W(inputs, B_pk, f_elec)

    P_mech = mech_loss_W(omega)
    P_out = inputs.target_torque_Nm * omega
    P_in = P_out + P_cu + P_core + P_mech
    eff = P_out / max(1e-9, P_in)

    m_cu = copper_mass_kg(inputs)
    m_pm = magnet_mass_kg(inputs)
    cost = 12.0*m_cu + 80.0*m_pm  # rough material-only USD

    return MotorResults(
        lambda_Vs_per_rad = lam,
        Ld_H= Ld, Lq_H=Lq,
        R_phase_ohm=R_phase,
        Iq_A=Iq, Id_A=Id,
        torque_Nm=inputs.target_torque_Nm,
        omega_rad_s=omega,
        P_out_W=P_out,
        P_cu_W=P_cu,
        P_core_W=P_core,
        P_mech_W=P_mech,
        P_in_W=P_in,
        eff=eff,
        copper_mass_kg=m_cu,
        magnet_mass_kg=m_pm,
        cost_est_usd=cost
    )
