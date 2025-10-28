# PMSM Motor Physics Module

A simplified physics model for surface-mounted Permanent Magnet Synchronous Motors (PMSM) designed for rapid design exploration and optimization studies.

## Overview

This module provides analytical approximations of PMSM motor behavior, enabling quick performance estimates without requiring finite element analysis (FEA). While not FEA-accurate, the equations capture the fundamental electromagnetic relationships needed for preliminary design and optimization work.

## Key Features

- **Fast Performance Estimation**: Calculate torque, efficiency, and losses in milliseconds
- **Design Parameter Exploration**: Evaluate how geometric and winding changes affect performance
- **Material Cost Estimation**: Track copper and permanent magnet costs
- **Physics-Based**: Grounded in real electromagnetic relationships and material properties

## Physical Model

The module implements simplified equations for:

- **Flux Linkage**: Based on magnet strength, pole geometry, and winding turns
- **Inductances**: Calculated from tooth geometry and effective air gap
- **Copper Losses**: Temperature-dependent resistive losses (I²R)
- **Core Losses**: Steinmetz equation approximation for hysteresis and eddy currents
- **Mechanical Losses**: Friction and windage proportional to speed squared

### Key Assumptions

- Surface-mounted magnets (Ld ≈ Lq)
- Id = 0 control strategy (maximum torque per amp)
- Sinusoidal current waveforms
- Uniform flux distribution over pole arc
- Simplified end-winding geometry

## Installation

Simply place `physics.py` in your project directory or Python path.

```bash
# No external dependencies beyond Python standard library
python -m pip install --upgrade pip  # Ensure you have Python 3.7+
```

## Usage

### Basic Example

```python
from physics import MotorInputs, solve_operating_point

# Define motor geometry and operating conditions
motor = MotorInputs(
    slots=24,
    poles=8,
    stack_length_mm=100.0,
    stator_od_mm=200.0,
    stator_id_mm=120.0,
    air_gap_mm=1.0,
    magnet_thickness_mm=5.0,
    wire_d_mm=1.5,
    turns_per_phase=50,
    target_torque_Nm=30.0,
    target_speed_rpm=300.0,
    winding_temp_C=80.0
)

# Solve for operating point
results = solve_operating_point(motor)

# Access results
print(f"Efficiency: {results.eff * 100:.1f}%")
print(f"Required Current (Iq): {results.Iq_A:.1f} A")
print(f"Copper Loss: {results.P_cu_W:.1f} W")
print(f"Core Loss: {results.P_core_W:.1f} W")
print(f"Total Cost Estimate: ${results.cost_est_usd:.2f}")
```

### Design Sweep Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Sweep wire diameter to optimize efficiency
wire_diameters = np.linspace(0.8, 2.5, 20)
efficiencies = []
costs = []

for d in wire_diameters:
    motor.wire_d_mm = d
    results = solve_operating_point(motor)
    efficiencies.append(results.eff)
    costs.append(results.cost_est_usd)

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(wire_diameters, np.array(efficiencies) * 100)
plt.xlabel('Wire Diameter (mm)')
plt.ylabel('Efficiency (%)')

plt.subplot(1, 2, 2)
plt.plot(wire_diameters, costs)
plt.xlabel('Wire Diameter (mm)')
plt.ylabel('Cost (USD)')
plt.tight_layout()
plt.show()
```

## Data Structures

### MotorInputs

Defines the motor geometry, materials, and operating conditions:

| Parameter | Type | Description | Typical Range |
|-----------|------|-------------|---------------|
| `slots` | int | Number of stator slots | 12-48 |
| `poles` | int | Number of magnetic poles (must be even) | 4-16 |
| `stack_length_mm` | float | Axial length of lamination stack | 50-300 mm |
| `stator_od_mm` | float | Stator outer diameter | 100-500 mm |
| `stator_id_mm` | float | Stator inner diameter | 50-300 mm |
| `air_gap_mm` | float | Mechanical air gap | 0.5-2.0 mm |
| `magnet_thickness_mm` | float | Radial magnet thickness | 3-10 mm |
| `wire_d_mm` | float | Single conductor diameter | 0.5-3.0 mm |
| `turns_per_phase` | int | Series turns per phase winding | 20-200 |
| `target_torque_Nm` | float | Desired output torque | 1-1000 Nm |
| `target_speed_rpm` | float | Desired operating speed | 100-10000 rpm |
| `winding_temp_C` | float | Copper temperature for resistance | 20-120°C |

**Advanced Tuning Parameters:**
- `tooth_fill_k`: End-winding length multiplier (default: 1.1)
- `tooth_w_ratio`: Tooth width / slot pitch (default: 0.45)
- `pole_arc_ratio`: Magnet arc / pole pitch (default: 0.8)
- `k_leak`: Leakage factor (default: 1.2)
- `k_lambda`: Flux linking fraction (default: 0.9)
- `k_B`: B-field utilization from Br (default: 0.85)
- `core_loss_k`: Core loss scaling factor (default: 1.0)

### MotorResults

Contains all computed performance metrics:

| Field | Description | Units |
|-------|-------------|-------|
| `lambda_Vs_per_rad` | Flux linkage | V·s/rad |
| `Ld_H`, `Lq_H` | d and q axis inductances | H |
| `R_phase_ohm` | Phase resistance at operating temp | Ω |
| `Iq_A`, `Id_A` | q and d axis currents | A |
| `torque_Nm` | Electromagnetic torque | N·m |
| `omega_rad_s` | Mechanical speed | rad/s |
| `P_out_W` | Mechanical output power | W |
| `P_cu_W` | Copper resistive losses | W |
| `P_core_W` | Iron core losses | W |
| `P_mech_W` | Mechanical friction losses | W |
| `P_in_W` | Total input power | W |
| `eff` | Efficiency (P_out/P_in) | 0-1 |
| `copper_mass_kg` | Total copper mass | kg |
| `magnet_mass_kg` | Total permanent magnet mass | kg |
| `cost_est_usd` | Material cost estimate | USD |

## Material Constants

The module uses realistic material properties:

- **Copper Resistivity**: 1.724×10⁻⁸ Ω·m at 20°C
- **Copper Temperature Coefficient**: 0.00393 /°C
- **Copper Density**: 8,960 kg/m³
- **NdFeB Magnet Density**: 7,500 kg/m³
- **NdFeB Remanence (N35)**: 1.2 T
- **Vacuum Permeability**: 4π×10⁻⁷ H/m

Cost estimates use:
- Copper: $12/kg
- NdFeB magnets: $80/kg

## Applications

This module is ideal for:

- **Parametric Studies**: Quickly explore how design parameters affect performance
- **Optimization**: Use as an objective function for genetic algorithms, gradient descent, etc.
- **Education**: Teach motor design principles with interactive examples
- **Rapid Prototyping**: Generate initial designs before detailed FEA
- **Trade-off Analysis**: Balance efficiency, cost, size, and performance

## Limitations

⚠️ **Important**: This is a simplified analytical model. For final designs:

- Verify with FEA for accurate flux distribution and saturation effects
- Consider manufacturing tolerances and assembly variations
- Validate thermal assumptions with thermal models or testing
- Account for control system limitations and inverter effects
- Check mechanical stress and rotor dynamics

The model assumes:
- Ideal sinusoidal back-EMF
- No magnetic saturation
- Uniform temperature distribution
- Steady-state operation (no transient dynamics)
- Perfect control (Id = 0 achieved instantaneously)

## Technical Notes

### Effective Air Gap

The effective magnetic air gap includes:
- Physical mechanical gap
- Magnet reluctance contribution (magnet thickness / relative permeability)
- Carter coefficient approximation (+0.5mm equivalent for slotting effect)
- Leakage factor scaling

### Core Loss Approximation

Uses a simplified Steinmetz equation:
```
P_core = k_core × V_core × (k_h × f × B^β + k_e × f² × B²)
```

Where:
- β = 1.6 (Steinmetz exponent for silicon steel)
- k_h = 40 W/m³/T^β·Hz (hysteresis coefficient)
- k_e = 0.2 W/m³/T²·Hz² (eddy current coefficient)

### Winding Resistance

Temperature-dependent resistance:
```
R(T) = R₀ × [1 + α × (T - T₀)]
```

Mean length per turn includes:
- Circumferential length around stator
- End-winding extensions (2× stack length)
- Tooth fill factor for realistic routing

## Contributing

This module is designed for extensibility. Potential enhancements:

- Add field weakening calculations (Id < 0 control)
- Implement thermal modeling
- Add different magnet grades (N30-N52)
- Include segmented magnet geometries
- Model interior PM (IPM) configurations
- Add manufacturing tolerance sensitivity analysis

## License

GNU LGPL v3

## References

For deeper understanding of PMSM design:

1. Hendershot, J.R. & Miller, T.J.E. "Design of Brushless Permanent-Magnet Machines"
2. Hanselman, D.C. "Brushless Permanent Magnet Motor Design"
3. Gieras, J.F. "Permanent Magnet Motor Technology"
4. IEEE Standards for motor testing and performance characterization

## Contact

Power Electronics Group (www.powerelectronicsgroup.com) info@powerelectronicsgroup.com USA +1 571 781 2453

---

**Version**: 1.0  
**Last Updated**: October 2025
