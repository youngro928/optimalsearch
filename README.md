# Optimal Search Path Planning

Supplementary code for the journal paper:

> **"Optimal Search via Discrete-Time Pontryagin’s Minimum Principle and Recursive Bayesian Estimation"**
> Youngro Lee, Vladimir N. Dobrokhodov, Mark Karpenko
> Naval Postgraduate School
> *IEEE Transactions on Automation Science and Engineering*, 2026
> submitted

This framework generates optimal search trajectories for a sensor-equipped
agent tasked with maximizing the probability of detecting a stationary target.
The approach integrates **Recursive Bayesian Estimation (RBE)** for spatial
belief updates with trajectory optimization via the **discrete-time
Pontryagin's Minimum Principle (PMP)**. The resulting Two-Point Boundary Value
Problem (TPBVP) is solved  using the **Forward-Backward Sweep Method (FBSM)**.
---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Selecting a Scenario](#selecting-a-scenario)
  - [Configuring Parameters](#configuring-parameters)
  - [Initial Control Guess](#initial-control-guess)
  - [Saving Results](#saving-results)
- [Output Description](#output-description)
- [Contact](#contact)

---

## Requirements

- Python **3.9** or later
- numpy >= 1.24
- matplotlib >= 3.7

---

## Installation

**Step 1 — Clone or download the repository**

```bash
git clone https://github.com/[your-repo]/optimal-search.git
cd optimal-search
```

Or simply unzip the downloaded archive and navigate into the folder.

**Step 2 — Run the setup script**

The setup script automatically creates an isolated virtual environment,
upgrades pip, and installs all dependencies:

```bash
python setup.py
```

Example output:

```
=======================================================
 Optimal Search Path Planning — Environment Setup
=======================================================

[1/4] Checking Python version...
  [OK] Python 3.11.4

[2/4] Setting up virtual environment...
  Creating virtual environment at '.venv/'...
  [OK] Virtual environment created.

[3/4] Upgrading pip...
  [OK] pip 26.0.1

[4/4] Installing dependencies...
  [OK] numpy 2.1.0
  [OK] matplotlib 3.9.0

=======================================================
 Setup complete. Run the simulation with:
   source .venv/bin/activate
   python main.py
=======================================================
```

**Step 3 — Activate the virtual environment**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

To deactivate when finished:

```bash
deactivate
```

---

## Quick Start

```bash
python setup.py          # First-time setup only
source .venv/bin/activate
python main.py
```

Select a scenario when prompted (1–8) and the solver will run and display results automatically.

---

## Project Structure

```
optimal-search/
│
├── main.py                  # Entry point — run this file
├── params.py                # All configuration dataclasses
├── sensor.py                # Detection likelihood and gradient model
├── belief.py                # Target prior distributions and Bayesian update (RBE)
├── solver.py                # Forward-Backward Sweep Method (FBSM) solver
├── plotting.py              # Visualization routines
├── initial_guess.py         # Initial control guess generators
│
├── setup.py                 # Environment setup script
├── requirements.txt         # Dependency list
└── README.md                # This file
```

### Module Responsibilities

| Module | Description |
|---|---|
| `main.py` | Scenario selection, parameter configuration, and top-level orchestration. The only file most users need to edit. |
| `params.py` | Dataclasses for all parameter groups: `VehicleParams`, `SensorParams`, `TargetParams`, `SearchDomain`, `SimulationParams`, `InitialGuessParams`. |
| `sensor.py` | Computes the detection likelihood function and its spatial gradients (∂L/∂x, ∂L/∂y). |
| `belief.py` | Computes all 8 target prior distributions and performs the Bayesian belief update at each time step. |
| `solver.py` | Implements the `OptimalSearchSolver` class: forward state integration, backward costate integration, PMP-based optimal control law, and the FBSM iteration loop. |
| `plotting.py` | Generates the initial conditions figure (1×2) and the 6-panel diagnostic solution figure. |
| `initial_guess.py` | Generates initial control vectors: spiral (constant heading-rate) for all scenarios, and Coast Guard search patterns (parallel track, expanding square) for the uniform prior. |

---

## Usage

### Selecting a Scenario

When `python main.py` is run, a menu is displayed:

```
================================================================================
 OPTIMAL SEARCH PATH PLANNING
================================================================================

Available Scenarios:
  1. Single Gaussian       — Target location known approximately
  2. Bimodal Gaussian      — Target in one of two locations
  3. Multiple Gaussians    — Mixture model of possible locations
  4. Ring-Shaped           — Target likely on a perimeter
  5. Hotspot Regions       — Circular high-probability zones
  6. Grid Pattern          — Patrol area with evenly-spaced checkpoints
  7. Diagonal Gradient     — Probability increases along a diagonal
  8. Uniform Prior         — No prior information about target location
================================================================================

Select scenario [1-8]:
```

Each scenario corresponds to a different target prior distribution,
reflecting different levels of prior knowledge about the target's location.

---

### Configuring Parameters

All physical and simulation parameters are defined at the top of `main.py`
and can be modified directly:

**Vehicle**

```python
vehicle = VehicleParams(
    x0 = -900,    # Initial x position [m]
    y0 = -900,    # Initial y position [m]
    h  = 250.0,   # Constant flight altitude [m]
    v  = 50.0     # Constant flight speed [m/s]
)
```

**Sensor**

```python
sensor = SensorParams(
    P_std     = 0.7,      # Standard detection probability [-]
    d_std     = 250,      # Standard detection distance [m]
    alpha     = 1/250,    # Attenuation coefficient [1/m]
    alpha_std = 1/250     # Standard attenuation coefficient [1/m]
)
```

**Search Domain**

```python
domain = SearchDomain(
    sd    = 30,      # Grid resolution (sd × sd points)
    x_min = -2000,   # Domain x bounds [m]
    x_max =  2000,
    y_min = -2000,   # Domain y bounds [m]
    y_max =  2000
)
```

**Simulation**

```python
simulation = SimulationParams(
    tf             = 100,    # Total flight time [s]
    dt             = 1,      # Time step [s]
    max_iter       = 10000,  # Maximum FBSM iterations
    omega_init     = 0.3,    # Initial relaxation parameter
    omega_min      = 0.001,  # Minimum relaxation parameter
    omega_max      = 1.0,    # Maximum relaxation parameter
    omega_adj_rate = 0.8,    # Omega reduction rate on divergence
    convergence_tol= 1,      # Convergence threshold
    plot_every     = 0,      # 0 = final only; N = plot every N iterations
    save_plot      = False,  # Save figures to .png
    save_results   = False,  # Save solution arrays to .npz
    results_filename = 'optimal_search_solution'
)
```

> **Note on grid resolution:** Increasing `sd` beyond 30 significantly
> increases computation time, as the belief array scales as `sd × sd`
> at every time step.

---

### Initial Control Guess

The solver requires an initial trajectory guess to start the FBSM iteration.

**Spiral guess (default — all scenarios)**

A smooth spiral trajectory generated from a constant heading-angle rate:

```python
ig_params = InitialGuessParams(
    heading_rate    = 0.002,  # Angular rate [rad/s]
    initial_heading = 0.0     # Starting heading [rad]; 0 = East
)
```

**Coast Guard search pattern (Scenario 8 — Uniform Prior only)**

For the uniform prior, a conventional search pattern can be used instead:

```python
USE_SEARCH_PATTERN = True              # Enable pattern-based initial guess
SEARCH_PATTERN     = 'parallel_track' # 'parallel_track' or 'expanding_square'
TRACK_SPACING      = 500.0            # Track spacing [m]
PATTERN_DIRECTION  = 'horizontal'     # 'horizontal' or 'vertical'
```

This option is only activated when Scenario 8 is selected.

---

### Saving Results

**Figures** — Set `save_plot = True` in `SimulationParams`. Figures are
saved as high-resolution `.png` files (300 dpi):

```
optimal_search_solution_initial_conditions.png
optimal_search_solution_plot.png
```

**Solution arrays** — Set `save_results = True`. All solution arrays are
saved to a single compressed `.npz` file:

```
optimal_search_solution_data.npz
```

To load saved results in a separate script:

```python
import numpy as np
data = np.load('optimal_search_solution_data.npz')
x, y = data['x'], data['y']       # Position trajectory [m]
Q    = data['Q']                   # Cumulative non-detection probability
PoD  = (1 - data['Q'][-1]) * 100  # Final probability of detection [%]
```

The `.npz` file contains: `x`, `y`, `b`, `q`, `Q`, `lx`, `ly`, `lb`,
`ux`, `uy`, `t`, `iterations`, `converged`.

---

## Output Description

The solver produces two figures automatically.

**Figure 1 — Initial Conditions (1×2)**

Displayed before optimization begins:

- *(a) Initial Belief Map* — the target prior distribution over the search domain
- *(b) Initial Trajectory Guess* — the trajectory corresponding to the initial control guess

**Figure 2 — Solution (2×3)**

Displayed after the FBSM converges:

| Panel | Description |
|---|---|
| Top-left | Optimal trajectory overlaid on the final belief distribution |
| Top-center | Heading angle θ(t) profile |
| Top-right | Probability of Detection (PoD) over time |
| Bottom-left | Position costates λ_x(t), λ_y(t) |
| Bottom-center | Belief costates λ_b(t) for all grid points |
| Bottom-right | Hamiltonian H(t) — should be approximately constant at convergence |

---

## Contact

**Youngro Lee** PostDoc, youngro.lee.ks@nps.edu

**Vladimir N. Dobrokhodov** Associate Chair, vldobr@nps.edu

**Mark Karpenko** Research Professor, mkarpenk@nps.edu
