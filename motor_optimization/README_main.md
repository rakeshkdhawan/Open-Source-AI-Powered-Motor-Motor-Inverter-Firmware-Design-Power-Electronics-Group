# Design Your Own Electric Motor!

A complete toolkit for designing, optimizing, and understanding electric motors - no engineering degree required.

## What Is This Project?

This is a collection of tools that help you design electric motors from scratch. Whether you're building a science fair project, learning about electric vehicles, or just curious about how motors work, these tools will help you create your own custom motor design.

**What makes this special:**
Instead of guessing or using trial-and-error, these tools use real physics and smart algorithms to help you design motors that actually work well.

## Who Is This For?

- **Students** learning about electricity, magnetism, and engineering
- **Makers** building electric vehicles, robots, or other projects
- **Hobbyists** curious about how motors work
- **Teachers** looking for hands-on engineering projects
- **Anyone** who wants to understand electric motor design

**You don't need:**
- An engineering degree
- Expensive software
- A supercomputer
- Years of experience

**You DO need:**
- Basic Python installed on your computer
- Curiosity and willingness to experiment
- About 30 minutes to explore
- A calculator (or your phone)

## The Big Picture: How These Tools Work Together

Think of designing a motor like planning a road trip:

### 1. The Physics Calculator (physics.py)
**What it does:** Calculates how well a motor will perform
**Like:** A GPS that tells you how long a route will take

You give it motor specifications (size, wire thickness, number of magnets), and it tells you:
- How efficient the motor will be
- How much torque (turning force) it produces
- How much the materials will cost
- Where energy is wasted as heat

### 2. The Optimization Tools (ga.py and bayes.py)
**What they do:** Automatically find the best motor design
**Like:** A GPS that finds the fastest route for you

Instead of testing designs one by one, these tools test thousands of combinations automatically and tell you which ones are best.

- **Bayesian Optimization (bayes.py)**: Fast and smart - finds one excellent design in 15 minutes
- **Genetic Algorithm (ga.py)**: More thorough - finds many good designs showing different trade-offs

### 3. The Visualization Tools (plots.py)
**What it does:** Creates charts and graphs of your results
**Like:** A map showing all possible routes color-coded by speed

After optimization, these tools create professional-looking charts that help you understand:
- Which designs are best
- What trade-offs exist (efficiency vs. cost)
- Why certain designs work better
- Which design to actually build

### 4. The Machine Learning Tool (model_compare.py)
**What it does:** Tests which AI method predicts motor performance most accurately
**Like:** Testing which GPS app gives the most accurate time estimates

This helps speed up optimization by training a computer to predict motor performance almost instantly.

## Quick Start: Your First Motor Design (30 minutes)

### Step 1: Understand What You Want

Before designing, answer these questions:
- **What will the motor do?** (spin a wheel? turn a fan? lift something?)
- **How much torque do you need?** (how heavy is what you're moving?)
- **How fast should it spin?** (100 RPM? 1000 RPM? 10,000 RPM?)
- **Do you care more about efficiency or cost?** (high-performance or budget-friendly?)

**Example:** "I want a motor for a small electric go-kart. It needs to provide 30 Nm of torque at 300 RPM. I want the best efficiency I can get for under $200."

### Step 2: Try a Design Manually

Open the physics calculator and input a simple design:
- 24 slots
- 8 poles  
- 100mm stack length
- 200mm outer diameter
- 120mm inner diameter
- 1mm air gap
- 4mm magnets
- 1.5mm wire
- 50 turns per phase

Run the calculator and see what you get. Try changing wire thickness or magnet size and see how efficiency and cost change.

**What you'll learn:** How each design choice affects performance. Thicker wire = better efficiency but more cost. More magnets = more torque but heavier.

### Step 3: Let the Computer Search

Now that you understand the basics, let the computer test thousands of designs:

**Option A - Fast Search (15 minutes):**
Use Bayesian optimization to quickly find one excellent design. Perfect if you just want "the answer" fast.

**Option B - Thorough Exploration (30 minutes):**
Use genetic algorithm to find many good designs showing the full range of possibilities. Perfect if you want to see all your options.

### Step 4: Look at the Results

The visualization tools will create charts showing:
- **Pareto Front Chart**: Shows the trade-off between efficiency and cost
- **Best Designs Table**: Lists the top 10-20 motor designs with all their specifications

Pick the design that best fits your needs.

### Step 5: Validate and Build

Before building anything:
1. Double-check the design makes physical sense (magnets aren't bigger than the motor!)
2. Make sure you can actually buy the materials (wire, magnets, steel)
3. If this is for something important, show the design to someone with engineering experience
4. Start with a simple prototype to test before building the final version

## Understanding Motor Design Basics

### What Is an Electric Motor?

An electric motor converts electrical energy into mechanical motion (spinning). It works by using:
- **Electromagnets** (wire coils that become magnetic when electricity flows through them)
- **Permanent magnets** (regular magnets that are always magnetic)
- **The magic of magnetic forces** (magnets push and pull on each other, creating rotation)

### The Main Parts

**Stator (The Part That Doesn't Move):**
- Outer shell of the motor
- Contains copper wire coils wound around iron teeth
- Electricity flows through these coils creating magnetic fields
- The tools design motors where this part is on the OUTSIDE

**Rotor (The Part That Spins):**
- Inner cylinder attached to the shaft
- Has permanent magnets glued to its surface
- Spins inside the stator
- The tools design motors where this part is on the INSIDE (called "inner rotor")

**Air Gap:**
- Tiny space between rotor and stator (usually less than 1mm!)
- Smaller gaps = stronger forces = more torque
- But smaller gaps are harder to manufacture accurately

### The Key Design Trade-offs

**Efficiency vs. Cost:**
- More copper (thicker wire, more turns) = better efficiency but more expensive
- Stronger magnets = better performance but much more expensive
- You can't have maximum efficiency AND minimum cost - you have to choose a balance

**Size vs. Performance:**
- Bigger motor = more torque and power
- But bigger = heavier, more expensive, takes up more space
- The art is making the motor just big enough for your needs

**Speed vs. Torque:**
- High-speed motors (10,000+ RPM) are usually smaller and lighter
- High-torque motors (lots of turning force) are usually bigger and heavier
- Can't maximize both - pick what your application needs most

## The Tools Explained (For Beginners)

### Physics Calculator (README_physics.md)

**In simple terms:**
Type in motor dimensions and it calculates performance in less than 1 second.

**When to use it:**
- Learning how motors work
- Testing one design idea
- Understanding what each parameter does
- Quick calculations

**What you'll learn:**
- Bigger motors aren't always better
- Wire thickness really matters for efficiency
- Magnets are expensive (usually 60-70% of material cost!)
- Small changes in design can make big differences

### Bayesian Optimization (README_bayes.md)

**In simple terms:**
An AI that learns from each motor design it tests and gets smarter about where to search next. Like a really smart trial-and-error that learns from its mistakes.

**When to use it:**
- You want ONE best design
- You want results fast (15-20 minutes)
- You trust the computer to find the answer
- You know what objective you're optimizing (best efficiency/cost ratio, minimum losses, etc.)

**What you'll get:**
- Single optimal motor design
- Confidence that it's near the best possible
- Fast results
- Simple answer

### Genetic Algorithm (README_ga.md)

**In simple terms:**
Mimics biological evolution - creates a population of motor designs, lets the good ones "reproduce" and "mutate" over many generations, eventually getting a population of excellent designs.

**When to use it:**
- You want to see ALL your options
- You want to understand efficiency-cost trade-offs
- You have 30-60 minutes
- You want multiple good designs to choose from

**What you'll get:**
- 20-50 optimal designs
- A curve showing efficiency vs. cost trade-offs
- Flexibility to choose based on YOUR priorities
- Deep understanding of what's possible

### Visualization Tools (README_plots.md)

**In simple terms:**
Turns thousands of numbers into pictures that make sense. Shows you what the computer found.

**When to use it:**
- After running optimization
- To understand results
- To compare designs
- For presentations or reports

**What you'll get:**
- Pretty charts showing your best designs
- Clear pictures of trade-offs
- Professional-looking reports
- Easy-to-understand summaries

### Machine Learning Comparison (README_model_compare.md)

**In simple terms:**
Tests which AI method is best at predicting motor performance, so optimization can run even faster.

**When to use it:**
- Advanced projects
- When you're testing thousands of designs
- If you want to speed up optimization by 10-100×
- Learning about machine learning

**What you'll get:**
- Understanding of AI surrogate models
- Faster optimization
- More design exploration
- Advanced techniques

## Example Projects for Students

### Science Fair Project: "Designing the Most Efficient Motor"

**Goal:** Use these tools to find the most efficient motor design for a given size.

**Process:**
1. Define constraints (motor must fit in a 200mm diameter circle, cost under $150)
2. Run optimization to find best efficiency
3. Create visualizations showing design choices
4. Explain trade-offs in your presentation
5. (Optional) Build a prototype to validate predictions

**What judges will love:**
- You used real engineering tools
- You understood trade-offs
- You can explain your results clearly
- You validated with math and physics

### Robotics Project: "Custom Motor for Competition Robot"

**Goal:** Design a motor perfectly matched to your robot's needs.

**Process:**
1. Calculate torque and speed requirements from robot specifications
2. Use optimization to find lightest motor that meets requirements
3. Compare with commercial motors to show your design is better
4. Document design process for engineering notebook

**Competition advantage:**
- Motor perfectly matched to your robot (not over or under-spec'd)
- Lighter than commercial alternatives
- Understanding of motor design helps with other robot systems

### Maker Project: "DIY Electric Bike Motor"

**Goal:** Design an affordable, efficient motor for an electric bicycle conversion.

**Process:**
1. Research typical e-bike requirements (250-750W, 200-400 RPM at wheel)
2. Use optimization to find designs balancing cost and efficiency
3. Select design that fits your budget
4. Purchase materials and work with local machine shop to build

**Real-world learning:**
- Understanding of power requirements
- Manufacturing considerations
- Budget constraints
- System integration

### Learning Project: "Exploring Motor Design Space"

**Goal:** Understand how all the parameters affect motor performance.

**Process:**
1. Start with a baseline design
2. Change ONE parameter at a time (wire size, magnet thickness, number of slots, etc.)
3. Plot how each parameter affects efficiency, cost, and performance
4. Write a report on what you learned

**Deep understanding:**
- How each design choice matters
- Why engineers make certain trade-offs
- Intuition for motor design
- Foundation for more advanced projects

## Common Questions

### "I don't know Python. Can I still use this?"

Yes! Python is easy to learn for basic use. You just need to:
1. Install Python (free download)
2. Install a few packages (one command)
3. Run the scripts (one command each)

There are thousands of free Python tutorials online. You can learn enough to use these tools in 1-2 hours.

### "How do I know if my design will actually work?"

The physics calculator uses real equations, so the predictions are typically within 5-10% of reality. However:
- Always double-check dimensions make sense
- Consider manufacturing - can you actually make this?
- If possible, build a prototype to test
- Show your design to someone with experience

### "What if I want to build the motor I designed?"

Building a motor requires:
- Access to machine shop (lathe, mill)
- Materials (copper wire, magnets, steel laminations)
- Skills (winding coils, assembling parts)
- Testing equipment (to verify it works)

This is a great follow-up project, but start simple! Consider:
- Partnering with a local makerspace
- Working with a teacher who has machine shop access
- Starting with modifying an existing motor
- Finding a mentor who's built motors before

### "Can I use this for my homework?"

Absolutely! These are real engineering tools. Just make sure to:
- Cite the software properly
- Explain what you did (don't just copy results)
- Understand the physics behind the calculations
- Show your work and reasoning

### "What if my results don't make sense?"

Check these common issues:
- Did you enter dimensions in the right units? (mm not inches!)
- Are all parameters reasonable? (don't make wire thinner than 0.3mm)
- Did optimization actually finish? (check for error messages)
- Did you set realistic goals? (can't have 99% efficiency for $10)

If still stuck, the physics might be complicated. That's okay - learning what doesn't work is part of engineering!

### "How accurate are these tools?"

For preliminary design: Very good (within 5-10% typically)
For final design: Good enough to know if you're on the right track
For production: You'll need professional FEA software and testing

Think of these tools as a sophisticated calculator, not a crystal ball. They'll point you in the right direction, but real-world motors will have some differences.

## Next Steps for Learning

### Beginner Path (2-3 hours)
1. Read README_physics.md (understand the calculator)
2. Run physics calculator on a few example designs
3. Try changing parameters and see what happens
4. Create your first motor design!

### Intermediate Path (1 day)
1. Complete beginner path
2. Read README_bayes.md
3. Run Bayesian optimization on a simple problem
4. Understand the results and why that design is optimal
5. Compare your intuition vs. what the computer found

### Advanced Path (1 week)
1. Complete intermediate path
2. Read README_ga.md and README_plots.md
3. Run genetic algorithm optimization
4. Create visualizations of your results
5. Explore the Pareto front and understand all trade-offs
6. Write a report explaining your findings

### Expert Path (ongoing)
1. Complete advanced path
2. Read README_model_compare.md
3. Experiment with machine learning surrogate models
4. Try modifying the code for your specific needs
5. Build a physical prototype of your design
6. Test and compare predictions vs. reality
7. Share your learnings with others!

## Safety and Responsibility

### If You Build a Physical Motor

**Electrical Safety:**
- Motors can produce dangerous voltages and currents
- Always work with adult supervision
- Use proper insulation and safety equipment
- Start with low voltages for testing

**Mechanical Safety:**
- Spinning parts can cause injury
- Secure all components before testing
- Use guards and shields
- Start at low speeds

**Magnet Safety:**
- Strong magnets can pinch fingers
- Keep magnets away from electronics and medical devices
- Store safely when not in use

**General:**
- When in doubt, ask an expert
- Safety first, always
- Learn proper techniques before attempting

## Resources for Learning More

### Understanding Motors
- Search YouTube for "how electric motors work"
- Look up "BLDC motor" or "PMSM motor" explanations
- Read Wikipedia articles on electric motors
- Check out Khan Academy physics lessons

### Learning Python
- Codecademy Python course (free)
- Python.org tutorial
- YouTube Python tutorials for beginners
- Any intro programming course

### Engineering Resources
- Local makerspace or FabLab
- High school physics teacher
- University engineering department (many offer tours)
- Online engineering forums

## Contributing and Sharing

If you:
- Find a bug or problem
- Have an idea for improvement
- Build something cool with these tools
- Want to share your results

Please reach out! Open source means we learn together.

## Inspiration: Why Motor Design Matters

Electric motors are everywhere:
- **Transportation**: Electric cars, trains, bikes, scooters
- **Industry**: Robots, machine tools, pumps, fans
- **Home**: Appliances, tools, HVAC systems
- **Future**: Flying cars, robots, space exploration

Understanding motor design means understanding how a huge portion of modern technology works. These skills apply to:
- Electrical engineering
- Mechanical engineering
- Robotics and automation
- Clean energy and sustainability
- Product design

Plus, it's just really cool to design something that spins and does useful work!

## Final Encouragement

Designing an electric motor might seem intimidating at first, but remember:
- Every expert was once a beginner
- These tools make it accessible
- Learning by doing is the best way
- Mistakes are how you learn
- The physics is real, but the math is handled for you

Start simple, experiment, learn from results, and gradually build your understanding. In a few hours, you'll know more about motor design than 99% of people!

Welcome to the world of electric motor design. Let's build something amazing!

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

**Questions from students always welcome!** We love helping the next generation of engineers.

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Maintainer**: Power Electronics Group

---

*"The best way to predict the future is to invent it." - Alan Kay*

*Now go design something awesome!*
