"""
plotting.py — Visualization Routines

Provides all plotting functions for the Optimal Search Path Planning framework,
including diagnostic FBSM iteration plots and the initial condition overview figure.

Author: Youngro Lee, PostDoc at Naval Postgraduate School
Email : youngro.lee.ks@nps.edu
"""

import numpy as np
import matplotlib.pyplot as plt

from params import SimulationParams, SearchDomain, VehicleParams


# ---------------------------------------------------------------------------
# Shared matplotlib style
# ---------------------------------------------------------------------------

_RCPARAMS = {
    'font.family': 'Times New Roman',
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'figure.titlesize': 16,
}


def plot_initial_conditions(
    b0: np.ndarray,
    ig_ux: np.ndarray,
    ig_uy: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    vehicle: VehicleParams,
    domain: SearchDomain,
    sim: SimulationParams
):
    """
    Plot the initial belief map and initial trajectory guess side by side (1×2 subplot).

    Args:
        b0: Initial belief distribution
        ig_ux: Initial control guess in x-direction
        ig_uy: Initial control guess in y-direction
        Xd: Meshgrid x-coordinates
        Yd: Meshgrid y-coordinates
        vehicle: Vehicle parameters (for start position)
        domain: Search domain parameters
        sim: Simulation parameters
    """
    plt.rcParams.update(_RCPARAMS)

    # Reconstruct initial trajectory from initial control guess
    n_steps = len(ig_ux)
    x_ig = np.zeros(n_steps + 1)
    y_ig = np.zeros(n_steps + 1)
    x_ig[0] = vehicle.x0
    y_ig[0] = vehicle.y0
    for k in range(n_steps):
        x_ig[k + 1] = x_ig[k] + ig_ux[k] * sim.dt
        y_ig[k + 1] = y_ig[k] + ig_uy[k] * sim.dt

    # Convert to km
    x_ig_km = x_ig * 1e-3
    y_ig_km = y_ig * 1e-3
    Xd_km = Xd * 1e-3
    Yd_km = Yd * 1e-3
    x_min_km = domain.x_min * 1e-3
    x_max_km = domain.x_max * 1e-3
    y_min_km = domain.y_min * 1e-3
    y_max_km = domain.y_max * 1e-3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Initial Conditions', fontsize=16, fontweight='bold')

    # --- Left: Initial belief map ---
    ax1.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
    ax1.axvline(x=0, color='black', linestyle=':', linewidth=1.5)
    cf = ax1.contourf(Xd_km, Yd_km, b0, cmap='Greens', levels=20)
    ax1.scatter(vehicle.x0 * 1e-3, vehicle.y0 * 1e-3, s=120, marker='o',
                color='r', facecolors='none', label='Start', lw=2, zorder=5)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xticks(np.linspace(x_min_km, x_max_km, 5))
    ax1.set_yticks(np.linspace(y_min_km, y_max_km, 5))
    ax1.set_xlabel('x [km]')
    ax1.set_ylabel('y [km]')
    ax1.set_title('(a) Initial Belief Map')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.6)

    # --- Right: Initial trajectory guess ---
    ax2.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
    ax2.axvline(x=0, color='black', linestyle=':', linewidth=1.5)
    ax2.plot(x_ig_km, y_ig_km, lw=2.5, color='black', label='Path')
    ax2.scatter(x_ig_km[0], y_ig_km[0], s=120, marker='o', color='r',
                facecolors='none', label='Start', lw=2, zorder=5)
    ax2.scatter(x_ig_km[-1], y_ig_km[-1], s=120, marker='x', color='navy',
                label='End', lw=2, zorder=5)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xticks(np.linspace(x_min_km, x_max_km, 5))
    ax2.set_yticks(np.linspace(y_min_km, y_max_km, 5))
    ax2.set_xlabel('x [km]')
    ax2.set_ylabel('y [km]')
    ax2.set_title('(b) Initial Trajectory Guess')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.6)

    plt.tight_layout()

    if sim.save_plot:
        fname = f'{sim.results_filename}_initial_conditions.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"✓ Initial conditions plot saved to {fname}")

    plt.show()


def plot_solution(
    solver,
    x: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    Q: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    lb: np.ndarray,
    ig_ux: np.ndarray,
    ig_uy: np.ndarray,
    up_ux: np.ndarray,
    up_uy: np.ndarray,
    iteration: int
):
    """
    Generate the 6-panel diagnostic plot for a given FBSM iteration.

    This is the canonical result plot: trajectory/belief, heading angle,
    probability of detection, position costates, belief costates, and
    the Hamiltonian.

    Args:
        solver: OptimalSearchSolver instance (provides grid, parameters, time)
        x, y: Position trajectories
        b: Belief distribution over time
        q: Non-detection probability at each step
        Q: Cumulative non-detection probability
        lx, ly: Position costates
        lb: Belief costates
        ig_ux, ig_uy: Current (old) control inputs
        up_ux, up_uy: Updated (new) control inputs
        iteration: Current FBSM iteration index
    """
    plt.rcParams.update(_RCPARAMS)

    sim = solver.sim
    domain = solver.domain
    t = solver.t

    # Heading angles
    ig_theta = np.unwrap(np.arctan2(ig_uy, ig_ux)) * 180 / np.pi
    up_theta = np.unwrap(np.arctan2(up_uy, up_ux)) * 180 / np.pi

    # Control effort
    theta_dot = np.diff(ig_theta)
    control_effort = np.linalg.norm(theta_dot)

    # Probability of Detection [%]
    PoD = (1 - Q) * 100

    # Hamiltonian
    H = np.zeros(solver.n - 1)
    for k in range(solver.n - 1):
        temp = np.sum(lb[k + 1, :, :] * b[k + 1, :, :])
        H[k] = (
            np.log(q[k] + 1e-15) +
            lx[k + 1] * (x[k] + ig_ux[k] * sim.dt) +
            ly[k + 1] * (y[k] + ig_uy[k] * sim.dt) +
            temp
        )
    H_mean = np.mean(H)
    H_std = np.std(H)

    # Convert to km
    x_km = x * 1e-3
    y_km = y * 1e-3
    Xd_km = solver.Xd * 1e-3
    Yd_km = solver.Yd * 1e-3
    x_min_km = domain.x_min * 1e-3
    y_min_km = domain.y_min * 1e-3
    x_max_km = domain.x_max * 1e-3
    y_max_km = domain.y_max * 1e-3

    fig = plt.figure(figsize=(14, 8))

    # --- Plot 1: Trajectory and final belief ---
    ax1 = fig.add_subplot(231)
    ax1.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
    ax1.axvline(x=0, color='black', linestyle=':', linewidth=1.5)
    ax1.contourf(Xd_km, Yd_km, b[-1, :, :], cmap='Greens', levels=20)
    ax1.plot(x_km, y_km, lw=2.5, color='black')
    ax1.scatter(x_km[0], y_km[0], s=100, marker='o', color='r',
                facecolors='none', label='Start', lw=2)
    ax1.scatter(x_km[-1], y_km[-1], s=100, marker='x', color='b',
                label='End', lw=2)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xticks(np.linspace(x_min_km, x_max_km, 5))
    ax1.set_yticks(np.linspace(y_min_km, y_max_km, 5))
    ax1.set_xlabel('x [km]')
    ax1.set_ylabel('y [km]')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.6)

    # --- Plot 2: Control angle ---
    ax2 = fig.add_subplot(232)
    ax2.plot(t[:-1], ig_theta, lw=2)
    ax2.text(10, np.max(ig_theta) * 0.85,
             r'$||\dot{\theta}||_2$' + f' = {control_effort:.1f} deg²/s', fontsize=18,
             bbox=dict(boxstyle='round', facecolor='wheat'))
    ax2.set_xlim(t[0], t[-1])
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel(r'$\theta$ [deg]')
    ax2.grid(True, alpha=0.6)

    # --- Plot 3: Probability of Detection ---
    ax3 = fig.add_subplot(233)
    ax3.plot(t, PoD, lw=2, color='darkgreen')
    ax3.text(t[-1] * 0.1, 85,
             f'PoD(t$_f$) = {PoD[-1]:.1f}%', fontsize=18,
             bbox=dict(boxstyle='round', facecolor='wheat'))
    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(0, 100)
    ax3.set_xlabel('t [s]')
    ax3.set_ylabel('PoD [%]')
    ax3.grid(True, alpha=0.6)

    # --- Plot 4: Position costates ---
    ax4 = fig.add_subplot(234)
    ax4.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
    line_x, = ax4.plot(t, lx, label=r'$\lambda_x$', lw=2)
    ax4.scatter(t[-1], solver.lxf, s=100, marker='o',
                color=line_x.get_color(), facecolors='none')
    line_y, = ax4.plot(t, ly, linestyle='--', label=r'$\lambda_y$', lw=2)
    ax4.scatter(t[-1], solver.lyf, s=100, marker='x',
                color=line_y.get_color(), lw=1.5)
    ax4.legend(loc='best')
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax4.set_xlim(t[0], t[-1] * 1.03)
    ax4.set_xlabel('t [s]')
    ax4.grid(True, alpha=0.6)

    # --- Plot 5: Belief costates ---
    ax5 = fig.add_subplot(235)
    ax5.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
    lb_flat = lb.reshape(solver.n, -1)
    lbf_flat = solver.lbf.reshape(domain.sd * domain.sd, 1)
    for idx in range(domain.sd * domain.sd):
        line, = ax5.plot(t, lb_flat[:, idx], alpha=0.6)
        ax5.scatter(t[-1], lbf_flat[idx], s=50, marker='o',
                    facecolors='none', edgecolors=line.get_color(), alpha=0.6)
    ax5.set_xlim(t[0], t[-1] * 1.03)
    ax5.set_xlabel('t [s]')
    ax5.set_ylabel(r'$\lambda_b$')
    ax5.grid(True, alpha=0.6)

    # --- Plot 6: Hamiltonian ---
    ax6 = fig.add_subplot(236)
    ax6.plot(t[:-1], H, lw=2, color='darkred')
    ax6.text(t[-1] * 0.2, 1.2,
             f'Mean: {H_mean:.3f}\nStd: {H_std:.4f}', fontsize=18,
             bbox=dict(boxstyle='round', facecolor='wheat'))
    ax6.set_xlim(t[0], t[-1])
    ax6.set_ylim(0.6, 1.4)
    ax6.set_xlabel('t [s]')
    ax6.set_ylabel(r'$H$')
    ax6.grid(True, alpha=0.6)

    plt.tight_layout()

    if sim.save_plot:
        fname = f'{sim.results_filename}_plot.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {fname}")

    plt.show()
