"""
main.py — Optimal Search Path Planning: Main Entry Point

Selects the target prior scenario, configures all parameters, generates an
initial control guess, and runs the Forward-Backward Sweep Method (FBSM)
to compute the optimal search trajectory.

Scenario ordering:
    1. Single Gaussian
    2. Bimodal Gaussian
    3. Multiple Gaussians (mixture model)
    4. Ring-Shaped
    5. Hotspot Regions
    6. Grid Pattern
    7. Diagonal Gradient (custom PDF)
    8. Uniform Prior  ← uniform is last; also supports search-pattern initial guess

Author: Youngro Lee, PostDoc at Naval Postgraduate School
Email : youngro.lee.ks@nps.edu
"""

import numpy as np

from params import (
    VehicleParams, SensorParams, TargetParams,
    SearchDomain, SimulationParams, InitialGuessParams
)
from belief import compute_target_prior
from solver import OptimalSearchSolver
from plotting import plot_initial_conditions, plot_solution
from initial_guess import generate_spiral_guess, generate_search_pattern_guess


# ===========================================================================
# Shared configuration — edit here to change physical parameters
# ===========================================================================

vehicle = VehicleParams(
    x0=-900,
    y0=-900,
    h=250.0,
    v=50.0
)

sensor = SensorParams(
    P_std=0.7,
    d_std=250,
    alpha=1 / 250,
    alpha_std=1 / 250
)

domain = SearchDomain(
    sd=30,
    x_min=-2000,
    x_max=2000,
    y_min=-2000,
    y_max=2000
)

simulation = SimulationParams(
    tf=100,
    dt=1,
    max_iter=10000,
    omega_init=0.3,
    omega_min=0.001,
    omega_max=1.0,
    omega_adj_rate=0.8,
    convergence_tol=1,
    plot_every=0,           # 0 = plot only final result; N = plot every N iterations
    save_plot=False,        # Set True to save figures to disk
    save_results=False,     # Set True to save solution arrays to .npz
    results_filename='optimal_search_solution'
)

# ---------------------------------------------------------------------------
# Initial guess parameters for the spiral (constant-rate) generator.
# These apply to all scenarios except uniform prior with search-pattern option.
# ---------------------------------------------------------------------------
ig_params = InitialGuessParams(
    heading_rate=0.002,     # Heading angle rate [rad/s]  — adjust to taste
    initial_heading=0.0     # Initial heading angle [rad] — 0 = East
)

# ---------------------------------------------------------------------------
# Uniform prior — initial guess options
# ---------------------------------------------------------------------------
# Set USE_SEARCH_PATTERN = True to initialise with a Coast Guard search pattern
# instead of the spiral guess. Only meaningful for scenario 8 (Uniform Prior).
USE_SEARCH_PATTERN = False     # True / False
SEARCH_PATTERN     = 'parallel_track'   # 'parallel_track' or 'expanding_square'
TRACK_SPACING      = 500.0              # [m]
PATTERN_DIRECTION  = 'horizontal'       # for parallel_track: 'horizontal' or 'vertical'


# ===========================================================================
# Scenario menu
# ===========================================================================

def print_menu():
    print("=" * 80)
    print(" OPTIMAL SEARCH PATH PLANNING")
    print("=" * 80)
    print("\nAvailable Scenarios:")
    print("  1. Single Gaussian       — Target location known approximately")
    print("  2. Bimodal Gaussian      — Target in one of two locations")
    print("  3. Multiple Gaussians    — Mixture model of possible locations")
    print("  4. Ring-Shaped           — Target likely on a perimeter")
    print("  5. Hotspot Regions       — Circular high-probability zones")
    print("  6. Grid Pattern          — Patrol area with evenly-spaced checkpoints")
    print("  7. Diagonal Gradient     — Probability increases along a diagonal")
    print("  8. Uniform Prior         — No prior information about target location")
    print("=" * 80)


def build_scenario(choice: int):
    """
    Construct and return the TargetParams and prior_type for the chosen scenario.

    Args:
        choice: Integer 1–8 from the menu.

    Returns:
        Tuple (target, prior_type) ready for compute_target_prior().
    """
    if choice == 1:
        target = TargetParams(mu_x=0.0, mu_y=0.0, sigma_x=500.0, sigma_y=500.0)
        prior_type = 1

    elif choice == 2:
        target = TargetParams(mu_x=0.0, mu_y=0.0, sigma_x=400.0, sigma_y=400.0)
        prior_type = 2

    elif choice == 3:
        gaussians_list = [
            # (mu_x, mu_y, sigma_x, sigma_y, weight)
            (-800,  -800, 300, 300, 0.5),   # 50% — southwest
            ( 800,   800, 300, 300, 0.3),   # 30% — northeast
            (   0, -1000, 400, 400, 0.2),   # 20% — south
        ]
        target = TargetParams(gaussians=gaussians_list)
        prior_type = 3

    elif choice == 4:
        target = TargetParams(mu_x=0.0, mu_y=0.0, sigma_x=600.0, sigma_y=600.0)
        prior_type = 4

    elif choice == 5:
        hotspots_list = [
            # (center_x, center_y, radius, probability_weight)
            (-1000, 0, 700, 1.0),   # Left hotspot
            ( 1000, 0, 700, 1.0),   # Right hotspot
        ]
        target = TargetParams(hotspots=hotspots_list)
        prior_type = 5

    elif choice == 6:
        target = TargetParams(sigma_x=300.0, sigma_y=300.0)
        prior_type = 6

    elif choice == 7:
        # Diagonal gradient — custom PDF increasing along the main diagonal
        custom_dist = np.zeros((domain.sd, domain.sd))
        for i in range(domain.sd):
            for j in range(domain.sd):
                x_idx = i / domain.sd
                y_idx = j / domain.sd
                custom_dist[i, j] = (x_idx + y_idx) / 2.0
        target = TargetParams(custom_pdf=custom_dist)
        prior_type = 7

    elif choice == 8:
        # Uniform prior
        target = TargetParams()
        prior_type = 8

    else:
        print(f"Invalid choice '{choice}'. Defaulting to Scenario 1 (Single Gaussian).")
        target = TargetParams(mu_x=0.0, mu_y=0.0, sigma_x=500.0, sigma_y=500.0)
        prior_type = 1

    return target, prior_type


# ===========================================================================
# Main
# ===========================================================================

def main():
    print_menu()

    try:
        choice = int(input("\nSelect scenario [1-8]: "))
    except ValueError:
        print("Invalid input. Running Scenario 1 (Single Gaussian) by default.")
        choice = 1

    # ----- Build scenario -----
    target, prior_type = build_scenario(choice)

    # ----- Instantiate solver -----
    solver = OptimalSearchSolver(vehicle, sensor, domain, simulation)

    # ----- Compute initial belief -----
    b0 = compute_target_prior(prior_type, target, domain, solver.Xd, solver.Yd)

    # ----- Generate initial control guess -----
    is_uniform = (choice == 8) or (prior_type == 8)

    if is_uniform and USE_SEARCH_PATTERN:
        print(f"\n[Info] Uniform prior selected — using '{SEARCH_PATTERN}' "
              f"pattern as initial guess.")
        ig_ux, ig_uy = generate_search_pattern_guess(
            vehicle=vehicle,
            domain_bounds=(domain.x_min, domain.x_max, domain.y_min, domain.y_max),
            sim=simulation,
            pattern=SEARCH_PATTERN,
            track_spacing=TRACK_SPACING,
            direction=PATTERN_DIRECTION
        )
    else:
        print(f"\n[Info] Initial guess: spiral with "
              f"heading_rate={ig_params.heading_rate:.4f} rad/s, "
              f"initial_heading={np.degrees(ig_params.initial_heading):.1f}°.")
        ig_ux, ig_uy = generate_spiral_guess(vehicle, simulation, ig_params, solver.n)

    # ----- Plot initial conditions -----
    plot_initial_conditions(
        b0, ig_ux, ig_uy,
        solver.Xd, solver.Yd,
        vehicle, domain, simulation
    )

    # ----- Solve -----
    print("\n" + "-" * 80)
    print("Starting optimization...")
    print("-" * 80 + "\n")

    solution = solver.solve(
        b0,
        initial_control=(ig_ux, ig_uy),
        plot_callback=plot_solution
    )

    # ----- Display results -----
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"  Converged   : {solution['converged']}")
    print(f"  Iterations  : {solution['iterations'] - 1}")
    print(f"  Final PoD   : {(1 - solution['Q'][-1]) * 100:.2f}%")
    print(f"  Flight time : {simulation.tf:.1f} s")
    print(f"{'='*80}")

    # ----- Optionally save results -----
    if simulation.save_results:
        fname = f'{simulation.results_filename}_data.npz'
        np.savez(fname, **solution)
        print(f"\n✓ Results saved to {fname}")


if __name__ == '__main__':
    main()