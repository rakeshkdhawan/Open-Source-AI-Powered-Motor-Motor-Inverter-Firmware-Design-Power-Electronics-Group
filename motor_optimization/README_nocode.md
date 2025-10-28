# PMSM Motor Physics Calculator

A fast and practical tool for calculating internal rotor electric motor performance without expensive computer simulations.

## What Does This Do?

This tool calculates how well an electric motor will perform based on its design. Think of it as a calculator for motors - you input the design specifications, and it tells you the efficiency, cost, and power losses in less than a second.

## Why Is This Important?

Designing an electric motor usually requires expensive finite element analysis software that can take minutes to analyze a single design. This tool gives you answers in milliseconds, making it perfect for exploring hundreds or thousands of design options quickly.

**Real-World Impact:**
- Design exploration that would take days now takes minutes
- Test different motor configurations instantly
- Understand trade-offs between cost and performance
- Get reliable estimates before investing in detailed simulations

## What Can You Calculate?

The tool figures out all the important motor characteristics:

### Performance Metrics
- **Efficiency**: How much electrical power actually becomes mechanical power (the rest becomes heat)
- **Torque**: The turning force the motor produces
- **Speed**: How fast the motor shaft spins
- **Power Output**: The mechanical power delivered to your application

### Loss Analysis
Understanding where energy is wasted helps improve designs:

- **Copper Losses**: Heat generated in the motor windings (like resistance in a wire)
- **Core Losses**: Energy lost to magnetizing and demagnetizing the iron (split into hysteresis and eddy currents)
- **Mechanical Losses**: Friction and air resistance as the motor spins

### Cost Estimation
Material costs are automatically calculated:
- Copper wire weight and cost
- Permanent magnet weight and cost
- Total material cost estimate

## Design Parameters You Control

The tool needs you to specify your motor design:

### Basic Geometry
- **Number of Slots**: How many slots in the stator (typical: 12-48)
- **Number of Poles**: How many magnetic poles (must be even, typical: 4-16)
- **Stack Length**: How long the motor is (50-300 mm)
- **Diameters**: Outer and inner stator dimensions (100-500 mm)
- **Air Gap**: Space between rotor and stator (0.5-2.0 mm)

### Magnetic Components
- **Magnet Thickness**: How thick the permanent magnets are (3-10 mm)
- Uses standard neodymium magnets (NdFeB, grade N35-N42)

### Electrical Winding
- **Wire Diameter**: Thickness of copper wire (0.5-3.0 mm)
- **Turns per Phase**: How many times the wire wraps around (20-200)
- **Operating Temperature**: Expected copper temperature during operation (20-120°C)

### Operating Conditions
- **Target Torque**: How much torque you need (1-1000 N·m)
- **Target Speed**: Desired operating speed (100-10,000 RPM)

## How It Works

The tool uses simplified but accurate physics equations:

### Magnetic Calculations
Calculates the magnetic flux (magnetic field strength) based on magnet properties, air gap size, and geometry. The flux determines how strong the electromagnetic forces will be.

### Electrical Calculations
Figures out how much current is needed to produce your target torque. Also calculates winding resistance, which depends on wire length, diameter, and temperature.

### Loss Calculations
Estimates three types of energy losses:

1. **Copper Losses**: Uses the electrical resistance formula - more current or thinner wire means more heat
2. **Core Losses**: Uses the Steinmetz equation, which accounts for how iron responds to changing magnetic fields
3. **Mechanical Losses**: Simple friction model based on speed

### Efficiency Calculation
Efficiency = (Power Out) / (Power In)

Power In = Power Out + All Losses

Higher efficiency means less wasted energy and lower operating costs.

## Material Properties Used

The calculations use real-world material properties:

**Copper Wire:**
- Density: 8,960 kg/m³
- Resistivity: Changes with temperature (hotter = more resistance)
- Cost: $12 per kilogram

**Neodymium Magnets:**
- Density: 7,500 kg/m³
- Magnetic strength: 1.2 Tesla (grade N35)
- Cost: $80 per kilogram

## What This Tool Is Good For

### Design Exploration
Try different combinations of slots, poles, wire sizes, and geometries to see what works best. You can test hundreds of variations in minutes.

### Optimization Studies
Feed this calculator into optimization algorithms (genetic algorithms, Bayesian optimization) to automatically find the best design.

### Education and Learning
Understand how motor design choices affect performance. See immediately how increasing magnet thickness improves torque but increases cost.

### Preliminary Design
Get a good starting point before investing time in detailed finite element analysis or building physical prototypes.

### Trade-off Analysis
Visualize the relationship between efficiency, cost, size, and performance. Make informed engineering decisions.

## Important Limitations

This is a simplified model, not a substitute for professional engineering analysis:

### What It Assumes
- Magnets mounted on the rotor surface (not buried inside)
- Ideal sinusoidal currents and magnetic fields
- No magnetic saturation (iron doesn't "max out")
- Uniform temperature throughout
- Steady-state operation (not starting or stopping)
- Perfect control system

### What It Doesn't Include
- Detailed magnetic field distribution
- Magnetic saturation effects
- Torque ripple and cogging
- Acoustic noise predictions
- Thermal analysis (heat flow)
- Mechanical stress analysis
- Manufacturing tolerances

### When to Verify Results
Always validate important designs with:
- Finite element analysis (FEA) for accurate magnetic fields
- Thermal modeling for temperature distribution
- Mechanical analysis for structural integrity
- Physical prototypes before production

## Typical Accuracy

For preliminary design work, this tool is usually accurate within:
- Efficiency: ±2-5%
- Torque: ±5-10%
- Losses: ±10-20%
- Cost: ±15% (material prices vary)

Good enough for optimization and design exploration, but verify final designs with detailed analysis.

## Real-World Applications

This tool has been used for:

**Electric Vehicles:**
- Exploring motor designs for different vehicle sizes
- Balancing power density against cost

**Industrial Automation:**
- Selecting optimal motor configurations for robotics
- Minimizing size while maintaining torque

**Renewable Energy:**
- Wind turbine generator design
- Optimizing efficiency for variable speed operation

**Consumer Products:**
- Power tools and appliances
- Balancing performance against manufacturing cost

## Understanding the Results

After calculating, you get a complete performance report:

**Efficiency Number:**
Tells you what percentage of electrical power becomes useful mechanical power. Modern motors typically achieve 85-95% efficiency.
- 90%+ is excellent
- 85-90% is good
- Below 85% suggests room for improvement

**Cost Estimate:**
Material costs only (doesn't include manufacturing, assembly, or overhead). Useful for comparing designs relative to each other.

**Loss Breakdown:**
Shows where energy is wasted:
- High copper losses? Try thicker wire or fewer turns
- High core losses? Reduce speed or use better steel
- Understanding losses guides design improvements

## Advanced Features

The tool includes tuning parameters for experienced users:

**Geometric Factors:**
- Tooth width ratio (affects slot area)
- Pole arc ratio (how much of the pole has magnets)
- End-winding length factor (accounts for wire outside the core)

**Magnetic Factors:**
- Leakage factor (not all flux reaches the air gap)
- Flux linkage fraction (accounts for winding distribution)
- Field utilization (real vs. theoretical magnetic strength)

Most users can leave these at default values, but they allow calibration against measured motors.

## Getting Started

To use this tool:

1. Gather your motor design specifications
2. Choose reasonable values for slots, poles, and geometry
3. Specify your wire size and number of turns
4. Set your target torque and speed
5. Run the calculation
6. Review efficiency, losses, and cost
7. Adjust parameters and recalculate
8. Compare different design options

The instant feedback lets you explore the design space quickly and develop intuition about what makes a good motor.

## Best Practices

**Start Simple:**
Begin with standard configurations (24 slots, 8 poles) and modify from there.

**Validate Key Assumptions:**
If your design is unusual, the simplified model may be less accurate. Verify with FEA.

**Use Relative Comparisons:**
The tool is excellent for comparing Design A vs. Design B. Absolute values may have 5-10% error.

**Check Physical Feasibility:**
Make sure inner diameter is smaller than outer diameter, air gap is reasonable, etc. The tool will calculate anything you input, even physically impossible motors.

**Consider Manufacturing:**
Some combinations of slots and poles are hard to manufacture. Consult with motor manufacturers about practical constraints.

## Future Enhancements

Possible improvements for future versions:
- Add thermal calculations
- Include different magnet grades
- Model interior permanent magnet (IPM) designs
- Account for field weakening at high speeds
- Add sensitivity analysis for manufacturing tolerances

## Summary

This is a fast, practical calculator for electric motor design. It won't replace professional FEA software, but it will help you:
- Explore designs quickly
- Understand trade-offs
- Make informed decisions
- Optimize motor performance
- Reduce development time

Think of it as a sketch pad for motor design - perfect for brainstorming and exploration before committing to detailed analysis.

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
