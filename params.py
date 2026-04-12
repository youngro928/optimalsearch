"""
params.py — Configuration Dataclasses

Defines all parameter groups for the Optimal Search Path Planning framework.

Author: Youngro Lee, PostDoc at Naval Postgraduate School
Email : youngro.lee.ks@nps.edu
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class VehicleParams:
    """Vehicle configuration parameters."""
    x0: float          # Initial x location [m]
    y0: float          # Initial y location [m]
    h: float           # Constant altitude [m]
    v: float           # Constant speed [m/s]


@dataclass
class SensorParams:
    """Sensor detection model parameters."""
    P_std: float          # Standard detection probability
    d_std: float          # Standard distance [m]
    alpha: float          # Attenuation coefficient
    alpha_std: float      # Standard attenuation coefficient


@dataclass
class TargetParams:
    """Target prior distribution parameters."""
    mu_x: float = None           # Mean in x-direction [m] (for single Gaussian)
    mu_y: float = None           # Mean in y-direction [m] (for single Gaussian)
    sigma_x: float = None        # Standard deviation in x-direction [m]
    sigma_y: float = None        # Standard deviation in y-direction [m]

    # For multiple Gaussians (prior_type=3)
    # Each element: (mu_x, mu_y, sigma_x, sigma_y, weight)
    gaussians: Optional[List[Tuple[float, float, float, float, float]]] = None

    # For hotspot regions (prior_type=5)
    # Each element: (center_x, center_y, radius, probability_weight)
    hotspots: Optional[List[Tuple[float, float, float, float]]] = None

    # For custom PDF (prior_type=7 — Diagonal Gradient)
    custom_pdf: Optional[np.ndarray] = None


@dataclass
class SearchDomain:
    """Search domain discretization parameters."""
    sd: int              # Number of grid points per dimension
    x_min: float         # Minimum x coordinate [m]
    x_max: float         # Maximum x coordinate [m]
    y_min: float         # Minimum y coordinate [m]
    y_max: float         # Maximum y coordinate [m]


@dataclass
class SimulationParams:
    """Simulation configuration parameters."""
    tf: float              # Flight time [s]
    dt: float              # Time interval [s]
    max_iter: int          # Maximum FBSM iterations
    omega_init: float      # Initial relaxation parameter
    omega_min: float       # Minimum relaxation parameter
    omega_max: float       # Maximum relaxation parameter
    omega_adj_rate: float  # Adjustment rate
    convergence_tol: float # Convergence tolerance
    plot_every: int        # Plot every N iterations (0 = only final)
    save_plot: bool = False          # Whether to save the final plot to file
    save_results: bool = False       # Whether to save results to .npz file
    results_filename: str = 'optimal_search_solution'  # Filename for saved results


@dataclass
class InitialGuessParams:
    """Parameters for generating the initial control guess."""
    heading_rate: float = 0.002    # Constant heading angle rate [rad/s]
    initial_heading: float = 0.0   # Initial heading angle [rad]
