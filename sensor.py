"""
sensor.py — Sensor Detection Model

Computes detection likelihood and its spatial gradients for a
sensor-equipped searcher operating at constant altitude.

Author: Youngro Lee, PostDoc at Naval Postgraduate School
Email : youngro.lee.ks@nps.edu
"""

import numpy as np
from typing import Tuple
from params import VehicleParams, SensorParams


def detection_likelihood(
    x: float,
    y: float,
    Xd: np.ndarray,
    Yd: np.ndarray,
    vehicle: VehicleParams,
    sensor: SensorParams
) -> np.ndarray:
    """
    Compute detection likelihood function at searcher position (x, y).

    Args:
        x: Searcher x-coordinate [m]
        y: Searcher y-coordinate [m]
        Xd: Grid x-coordinates
        Yd: Grid y-coordinates
        vehicle: Vehicle parameters
        sensor: Sensor parameters

    Returns:
        Detection probability at each grid point
    """
    d = np.sqrt(
        vehicle.h ** 2 +
        (x - Xd) ** 2 +
        (y - Yd) ** 2
    )

    P = (
        sensor.P_std *
        sensor.d_std ** 4 /
        (d ** 4 + 1e-15) *
        np.exp(-2 * (sensor.alpha * d - sensor.alpha_std * sensor.d_std))
    )

    return P


def detection_gradient(
    x: float,
    y: float,
    Xd: np.ndarray,
    Yd: np.ndarray,
    vehicle: VehicleParams,
    sensor: SensorParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute partial derivatives of detection likelihood with respect to x and y.

    Args:
        x: Searcher x-coordinate [m]
        y: Searcher y-coordinate [m]
        Xd: Grid x-coordinates
        Yd: Grid y-coordinates
        vehicle: Vehicle parameters
        sensor: Sensor parameters

    Returns:
        Tuple of (dL/dx, dL/dy) arrays
    """
    d = np.sqrt(
        vehicle.h ** 2 +
        (x - Xd) ** 2 +
        (y - Yd) ** 2
    )

    # Gradient computation (mathematical formulation preserved exactly)
    term1 = -2 * sensor.P_std * sensor.d_std ** 4
    term2 = (2 + sensor.alpha * d) / (d ** 6 + 1e-15)
    term3x = (x - Xd)
    term3y = (y - Yd)
    term4 = np.exp(2 * (sensor.alpha_std * sensor.d_std - sensor.alpha * d))

    ddldx = term1 * term2 * term3x * term4
    ddldy = term1 * term2 * term3y * term4

    return ddldx, ddldy
