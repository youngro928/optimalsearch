"""
belief.py — Target Belief Distribution

Implements the Recursive Bayesian Estimation (RBE) belief update and
provides factory functions for computing various target prior distributions.

Author: Youngro Lee, PostDoc at Naval Postgraduate School
Email : youngro.lee.ks@nps.edu
"""

import numpy as np
from typing import Tuple, Optional
from params import TargetParams, SearchDomain

def compute_target_prior(
    prior_type: int,
    target: TargetParams,
    domain: SearchDomain,
    Xd: np.ndarray,
    Yd: np.ndarray
) -> np.ndarray:
    """
    Compute probability density function for target location prior.

    Args:
        prior_type: Type of prior distribution
            1 - Single Gaussian distribution
            2 - Bimodal Gaussian (two peaks)
            3 - Multiple Gaussians (mixture model)
            4 - Ring-shaped distribution
            5 - Hotspot regions (circular high-probability zones)
            6 - Grid pattern (multiple evenly-spaced Gaussians)
            7 - Diagonal gradient (custom user-provided PDF)
            8 - Uniform distribution
        target: Target distribution parameters
        domain: Search domain parameters
        Xd: Meshgrid x-coordinates
        Yd: Meshgrid y-coordinates

    Returns:
        Normalized PDF over the search domain
    """
    if prior_type == 1:
        # Single Gaussian prior
        exp_term = -(
            (Xd - target.mu_x) ** 2 / (2 * target.sigma_x ** 2) +
            (Yd - target.mu_y) ** 2 / (2 * target.sigma_y ** 2)
        )
        normal_dist = (1 / (2 * np.pi * target.sigma_x * target.sigma_y)) * np.exp(exp_term)
        pdf = normal_dist / (np.sum(normal_dist) + 1e-15)

    elif prior_type == 2:
        # Bimodal Gaussian (two peaks)
        separation = max(target.sigma_x, target.sigma_y) * 2

        exp_term1 = -(
            (Xd - (target.mu_x - separation)) ** 2 / (2 * target.sigma_x ** 2) +
            (Yd - target.mu_y) ** 2 / (2 * target.sigma_y ** 2)
        )
        gaussian1 = np.exp(exp_term1)

        exp_term2 = -(
            (Xd - (target.mu_x + separation)) ** 2 / (2 * target.sigma_x ** 2) +
            (Yd - target.mu_y) ** 2 / (2 * target.sigma_y ** 2)
        )
        gaussian2 = np.exp(exp_term2)

        pdf = (gaussian1 + gaussian2) / 2.0
        pdf = pdf / (np.sum(pdf) + 1e-15)

    elif prior_type == 3:
        # Multiple Gaussians (mixture model)
        if target.gaussians is None:
            raise ValueError("prior_type=3 requires target.gaussians to be specified")

        pdf = np.zeros_like(Xd)

        for mu_x, mu_y, sigma_x, sigma_y, weight in target.gaussians:
            exp_term = -(
                (Xd - mu_x) ** 2 / (2 * sigma_x ** 2) +
                (Yd - mu_y) ** 2 / (2 * sigma_y ** 2)
            )
            gaussian = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(exp_term)
            pdf += weight * gaussian

        pdf = pdf / (np.sum(pdf) + 1e-15)

    elif prior_type == 4:
        # Ring-shaped distribution
        ring_radius = max(target.sigma_x, target.sigma_y) * 2
        ring_width = ring_radius / 4

        dist_from_center = np.sqrt(
            (Xd - target.mu_x) ** 2 +
            (Yd - target.mu_y) ** 2
        )

        pdf = np.exp(-((dist_from_center - ring_radius) ** 2) / (2 * ring_width ** 2))
        pdf = pdf / (np.sum(pdf) + 1e-15)

    elif prior_type == 5:
        # Hotspot regions (circular high-probability zones)
        if target.hotspots is None:
            raise ValueError("prior_type=5 requires target.hotspots to be specified")

        pdf = np.ones_like(Xd) * 1e-6

        for center_x, center_y, radius, prob_weight in target.hotspots:
            dist = np.sqrt((Xd - center_x) ** 2 + (Yd - center_y) ** 2)
            hotspot = prob_weight * np.exp(-((dist - radius) ** 2) / (2 * (radius / 3) ** 2))
            pdf += hotspot

        pdf = pdf / (np.sum(pdf) + 1e-15)

    elif prior_type == 6:
        # Grid pattern (evenly-spaced Gaussians)
        n_peaks_x = 3
        n_peaks_y = 3

        x_spacing = (domain.x_max - domain.x_min) / (n_peaks_x + 1)
        y_spacing = (domain.y_max - domain.y_min) / (n_peaks_y + 1)

        pdf = np.zeros_like(Xd)

        for i in range(1, n_peaks_x + 1):
            for j in range(1, n_peaks_y + 1):
                center_x = domain.x_min + i * x_spacing
                center_y = domain.y_min + j * y_spacing

                exp_term = -(
                    (Xd - center_x) ** 2 / (2 * target.sigma_x ** 2) +
                    (Yd - center_y) ** 2 / (2 * target.sigma_y ** 2)
                )
                pdf += np.exp(exp_term)

        pdf = pdf / (np.sum(pdf) + 1e-15)

    elif prior_type == 7:
        # Diagonal gradient (custom user-provided PDF)
        if target.custom_pdf is None:
            raise ValueError("prior_type=7 requires target.custom_pdf to be specified")

        if target.custom_pdf.shape != Xd.shape:
            raise ValueError(
                f"custom_pdf shape {target.custom_pdf.shape} must match grid shape {Xd.shape}"
            )

        pdf = target.custom_pdf / (np.sum(target.custom_pdf) + 1e-15)

    elif prior_type == 8:
        # Uniform prior
        N = Xd.size
        pdf = np.full(Xd.shape, 1.0 / N)

    else:
        raise ValueError(f"Invalid prior_type: {prior_type}. Must be 1-8.")

    return pdf


def bayesian_belief_update(
    x: float,
    y: float,
    b: np.ndarray,
    dl_val: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Update target belief using Bayesian conditional probability (RBE step).

    Args:
        x: Searcher x-coordinate [m] (unused directly; dl_val is pre-computed)
        y: Searcher y-coordinate [m] (unused directly; dl_val is pre-computed)
        b: Current belief distribution
        dl_val: Pre-computed detection likelihood array

    Returns:
        Tuple of (non-detection probability q, updated belief b1)
    """
    bp = (1 - dl_val) * b
    q = np.sum(bp) + 1e-15    # Numerical stability
    b1 = bp / q

    return q, b1
