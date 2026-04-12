"""
initial_guess.py — Initial Control Guess Generator

Generates the initial control vector (u_x, u_y) for the FBSM solver.

Two generation strategies are provided:

  1. Spiral (constant heading-angle rate) — default for all scenarios.
     Configurable via InitialGuessParams (heading_rate, initial_heading).

  2. Coast Guard search patterns — available only for the uniform prior
     scenario, where no spatial information biases the trajectory choice.
     Supported patterns:
         - Parallel Track Search (ladder pattern)
         - Expanding Square Search

Author: Youngro Lee, PostDoc at Naval Postgraduate School
Email : youngro.lee.ks@nps.edu
"""

import numpy as np
from typing import Tuple

from params import VehicleParams, SimulationParams, InitialGuessParams


# ===========================================================================
# Strategy 1 — Spiral (constant heading-angle rate)
# ===========================================================================

def generate_spiral_guess(
    vehicle: VehicleParams,
    sim: SimulationParams,
    ig_params: InitialGuessParams,
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an initial control guess using a constant heading-angle rate.

    The heading angle evolves as:
        theta(k) = initial_heading + cumsum(heading_rate * t_k)

    Args:
        vehicle: Vehicle parameters (speed used for control magnitude)
        sim: Simulation parameters (tf, dt used for time vector)
        ig_params: Initial guess parameters
            - heading_rate    : constant angular rate [rad/s]
            - initial_heading : starting heading angle [rad]
        n: Number of time steps (solver.n)

    Returns:
        Tuple (ig_ux, ig_uy) — initial control vectors of length n-1
    """
    t_vec = np.linspace(0, sim.tf, n - 1)
    ig_theta = ig_params.initial_heading + np.cumsum(ig_params.heading_rate * t_vec)

    ig_ux = vehicle.v * np.cos(ig_theta)
    ig_uy = vehicle.v * np.sin(ig_theta)

    return ig_ux, ig_uy


# ===========================================================================
# Strategy 2 — Coast Guard search patterns (uniform prior only)
# ===========================================================================

def generate_search_pattern_guess(
    vehicle: VehicleParams,
    domain_bounds: Tuple[float, float, float, float],
    sim: SimulationParams,
    pattern: str = 'parallel_track',
    track_spacing: float = 500.0,
    direction: str = 'horizontal'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an initial control guess from a conventional Coast Guard search
    pattern. Applicable **only for the uniform prior** scenario.

    Args:
        vehicle: Vehicle parameters
        domain_bounds: (x_min, x_max, y_min, y_max) [m]
        sim: Simulation parameters
        pattern: 'parallel_track' or 'expanding_square'
        track_spacing: Distance between parallel tracks [m]
        direction: 'horizontal' or 'vertical' (for parallel_track only)

    Returns:
        Tuple (ig_ux, ig_uy) — initial control vectors
    """
    n = int(sim.tf / sim.dt)

    if pattern == 'parallel_track':
        return _parallel_track_search(
            vehicle.x0, vehicle.y0, domain_bounds,
            track_spacing, vehicle.v, n, sim.dt,
            direction=direction
        )
    elif pattern == 'expanding_square':
        return _expanding_square_search(
            vehicle.x0, vehicle.y0,
            track_spacing, vehicle.v, n, sim.dt
        )
    else:
        raise ValueError(
            f"Unknown pattern: '{pattern}'. Choose 'parallel_track' or 'expanding_square'."
        )


# ===========================================================================
# Pattern implementations
# ===========================================================================

def _parallel_track_search(x0, y0, domain_bounds, track_spacing, vehicle_speed,
                            n, dt, direction='horizontal', **kwargs):
    """
    Generate parallel track (ladder) search pattern.

    Covers the search area with parallel lines, turning at domain boundaries.
    Maintains a gap of track_spacing from domain edges.

    Parameters
    ----------
    direction : str
        'horizontal' for E-W tracks, 'vertical' for N-S tracks
    """
    x_min, x_max, y_min, y_max = domain_bounds

    x_min_search = x_min + track_spacing
    x_max_search = x_max - track_spacing
    y_min_search = y_min + track_spacing
    y_max_search = y_max - track_spacing

    waypoints = [(x0, y0)]

    if direction == 'horizontal':
        going_right = x0 < (x_min_search + x_max_search) / 2
        y_current = y0

        while y_current <= y_max_search:
            if going_right:
                waypoints.append((x_max_search, y_current))
                waypoints.append((x_max_search, y_current + track_spacing))
            else:
                waypoints.append((x_min_search, y_current))
                waypoints.append((x_min_search, y_current + track_spacing))
            y_current += track_spacing
            going_right = not going_right

    else:  # vertical
        going_up = y0 < (y_min_search + y_max_search) / 2
        x_current = x0

        while x_current <= x_max_search:
            if going_up:
                waypoints.append((x_current, y_max_search))
                waypoints.append((x_current + track_spacing, y_max_search))
            else:
                waypoints.append((x_current, y_min_search))
                waypoints.append((x_current + track_spacing, y_min_search))
            x_current += track_spacing
            going_up = not going_up

    return _waypoints_to_controls(waypoints, x0, y0, vehicle_speed, n, dt)


def _expanding_square_search(x0, y0, track_spacing, vehicle_speed, n, dt,
                              start_corner='SW', **kwargs):
    """
    Generate expanding square search pattern.

    Starts at the initial position and spirals outward in a square pattern.
    Used when the target datum is known with some uncertainty.

    Parameters
    ----------
    start_corner : str
        Starting direction: 'SW', 'SE', 'NW', 'NE'
    """
    corner_directions = {
        'SW': [(1, 0), (0, 1), (-1, 0), (0, -1)],
        'SE': [(0, 1), (-1, 0), (0, -1), (1, 0)],
        'NW': [(0, -1), (1, 0), (0, 1), (-1, 0)],
        'NE': [(-1, 0), (0, -1), (1, 0), (0, 1)],
    }

    directions = corner_directions[start_corner]

    waypoints = [(x0, y0)]
    x, y = x0, y0
    leg_length = track_spacing
    max_distance = vehicle_speed * n * dt
    total_distance = 0

    while total_distance < max_distance:
        for i in range(4):
            dx, dy = directions[i]
            x_next = x + dx * leg_length
            y_next = y + dy * leg_length
            waypoints.append((x_next, y_next))
            x, y = x_next, y_next
            total_distance += leg_length

            if total_distance >= max_distance:
                break

            if i % 2 == 1:
                leg_length += track_spacing

    return _waypoints_to_controls(waypoints, x0, y0, vehicle_speed, n, dt)


def _waypoints_to_controls(waypoints, x0, y0, vehicle_speed, n, dt):
    """
    Convert a sequence of (x, y) waypoints to velocity control inputs.

    Parameters
    ----------
    waypoints : list of tuples
        (x, y) waypoint coordinates [m]
    x0, y0 : float
        Initial position [m]
    vehicle_speed : float
        Constant speed [m/s]
    n : int
        Number of time steps
    dt : float
        Time step [s]

    Returns
    -------
    ux, uy : ndarray
        Control velocity inputs [m/s]
    """
    ux = np.zeros(n)
    uy = np.zeros(n)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0

    current_waypoint = 0

    for k in range(n):
        if current_waypoint >= len(waypoints):
            ux[k] = ux[k - 1] if k > 0 else 0
            uy[k] = uy[k - 1] if k > 0 else 0
        else:
            target_x, target_y = waypoints[current_waypoint]
            dx = target_x - x[k]
            dy = target_y - y[k]
            distance = np.sqrt(dx ** 2 + dy ** 2)

            if distance < vehicle_speed * dt:
                current_waypoint += 1
                if current_waypoint < len(waypoints):
                    target_x, target_y = waypoints[current_waypoint]
                    dx = target_x - x[k]
                    dy = target_y - y[k]
                    distance = np.sqrt(dx ** 2 + dy ** 2)

            if distance > 1e-6:
                ux[k] = vehicle_speed * dx / distance
                uy[k] = vehicle_speed * dy / distance
            else:
                ux[k] = 0
                uy[k] = 0

        x[k + 1] = x[k] + ux[k] * dt
        y[k + 1] = y[k] + uy[k] * dt

    return ux, uy