"""
solver.py — Optimal Control Solver (FBSM)

Implements the Forward-Backward Sweep Method (FBSM) to solve the Two-Point
Boundary Value Problem (TPBVP) arising from the discrete-time Pontryagin's
Minimum Principle (PMP).

Author: Youngro Lee, PostDoc at Naval Postgraduate School
Email : youngro.lee.ks@nps.edu
"""

import warnings
import numpy as np
from typing import Tuple, Dict, Any

from params import VehicleParams, SensorParams, SearchDomain, SimulationParams
from sensor import detection_likelihood, detection_gradient
from belief import bayesian_belief_update


class OptimalSearchSolver:
    """
    Solver for optimal search path planning.

    Implements the Forward-Backward Sweep Method to solve
    the two-point boundary value problem arising from
    the discrete-time Pontryagin's Minimum Principle.
    """

    def __init__(
        self,
        vehicle: VehicleParams,
        sensor: SensorParams,
        domain: SearchDomain,
        simulation: SimulationParams
    ):
        self.vehicle = vehicle
        self.sensor = sensor
        self.domain = domain
        self.sim = simulation

        # Derived time grid
        self.n = 1 + int(self.sim.tf / self.sim.dt)
        self.t = np.linspace(0, self.sim.tf, self.n)

        # Initialize search domain grid
        self._initialize_grid()

        # Pre-compute terminal costate conditions
        self.lxf, self.lyf, self.lbf = self._compute_terminal_conditions()

    # ------------------------------------------------------------------
    # Grid initialization
    # ------------------------------------------------------------------

    def _initialize_grid(self):
        """Initialize the search domain discretization grid."""
        xd = np.linspace(self.domain.x_min, self.domain.x_max, self.domain.sd)
        yd = np.linspace(self.domain.y_min, self.domain.y_max, self.domain.sd)
        self.Xd, self.Yd = np.meshgrid(xd, yd)
        self.xyd = (xd[1] - xd[0]) * (yd[1] - yd[0])   # Grid cell area

    # ------------------------------------------------------------------
    # Terminal conditions
    # ------------------------------------------------------------------

    def _compute_terminal_conditions(self) -> Tuple[float, float, np.ndarray]:
        """
        Compute terminal boundary conditions for the costates.

        Returns:
            Tuple of (lambda_x_f, lambda_y_f, lambda_b_f)
        """
        lxf = 0.0
        lyf = 0.0
        lbf = np.ones((self.domain.sd, self.domain.sd))
        return lxf, lyf, lbf

    # ------------------------------------------------------------------
    # Forward sweep
    # ------------------------------------------------------------------

    def forward_state_integration(
        self,
        b0: np.ndarray,
        u1: np.ndarray,
        u2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward integration of the state equations.

        Args:
            b0: Initial belief distribution
            u1: Control input in x-direction  [m/s]
            u2: Control input in y-direction  [m/s]

        Returns:
            Tuple (x, y, b, q, Q) where:
                x : x-position trajectory
                y : y-position trajectory
                b : belief distribution over time
                q : non-detection probability at each step
                Q : cumulative non-detection probability
        """
        x = np.zeros(self.n)
        y = np.zeros(self.n)
        b = np.zeros((self.n, self.domain.sd, self.domain.sd))
        q = np.ones(self.n)
        Q = np.ones(self.n)

        x[0] = self.vehicle.x0
        y[0] = self.vehicle.y0
        b[0, :, :] = b0

        for k in range(self.n - 1):
            dl_val = detection_likelihood(x[k], y[k], self.Xd, self.Yd, self.vehicle, self.sensor)
            q1, b1 = bayesian_belief_update(x[k], y[k], b[k, :, :], dl_val)
            b[k + 1, :, :] = b1
            q[k + 1] = q1
            Q[k + 1] = Q[k] * q1

            x[k + 1] = x[k] + u1[k] * self.sim.dt
            y[k + 1] = y[k] + u2[k] * self.sim.dt

        return x, y, b, q, Q

    # ------------------------------------------------------------------
    # Backward sweep
    # ------------------------------------------------------------------

    def backward_costate_integration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        b: np.ndarray,
        q: np.ndarray,
        Q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward integration of the costate equations.

        Args:
            x: x-position trajectory
            y: y-position trajectory
            b: belief distribution over time
            q: non-detection probability at each step
            Q: cumulative non-detection probability

        Returns:
            Tuple (lambda_x, lambda_y, lambda_b) costates
        """
        lx = np.zeros(self.n)
        ly = np.zeros(self.n)
        lb = np.zeros((self.n, self.domain.sd, self.domain.sd))

        lx[-1] = self.lxf
        ly[-1] = self.lyf
        lb[-1, :, :] = self.lbf

        for j in range(self.n - 1):
            k = self.n - 1 - j    # Reversed index

            ddldx, ddldy = detection_gradient(
                x[k - 1], y[k - 1], self.Xd, self.Yd, self.vehicle, self.sensor
            )
            qk = q[k - 1]
            dk = detection_likelihood(x[k - 1], y[k - 1], self.Xd, self.Yd, self.vehicle, self.sensor)

            # Lambda_x update
            term0x = np.sum(b[k - 1, :, :] * ddldx)
            term1x = (-qk * ddldx + (1 - dk) * term0x) / (qk ** 2 + 1e-15)
            term2x = np.sum(lb[k, :, :] * b[k - 1, :, :] * term1x)
            lx[k - 1] = lx[k] + (-term0x / (qk + 1e-15) + term2x) * self.sim.dt

            # Lambda_y update
            term0y = np.sum(b[k - 1, :, :] * ddldy)
            term1y = (-qk * ddldy + (1 - dk) * term0y) / (qk ** 2 + 1e-15)
            term2y = np.sum(lb[k, :, :] * b[k - 1, :, :] * term1y)
            ly[k - 1] = ly[k] + (-term0y / (qk + 1e-15) + term2y) * self.sim.dt

            # Lambda_b update
            term1b = (1 - dk) / (qk + 1e-15)
            term2b = 1 + lb[k, :, :] - np.sum(lb[k, :, :] * b[k, :, :])
            lb[k - 1, :, :] = term1b * term2b

        return lx, ly, lb

    # ------------------------------------------------------------------
    # Optimal control law
    # ------------------------------------------------------------------

    def compute_optimal_control(
        self,
        lx: np.ndarray,
        ly: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal control from costates using PMP.

        Args:
            lx: Costate for x-position
            ly: Costate for y-position

        Returns:
            Tuple (u_x, u_y) optimal control inputs
        """
        mag = np.sqrt(lx ** 2 + ly ** 2) + 1e-12

        ux = np.zeros(self.n - 1)
        uy = np.zeros(self.n - 1)

        for k in range(self.n - 1):
            ux[k] = -self.vehicle.v * lx[k + 1] / mag[k + 1]
            uy[k] = -self.vehicle.v * ly[k + 1] / mag[k + 1]

        # Control extrapolation for boundary
        ux[-1] = 2 * ux[-2] - ux[-3]
        uy[-1] = 2 * uy[-2] - uy[-3]

        return ux, uy

    # ------------------------------------------------------------------
    # Main FBSM solve loop
    # ------------------------------------------------------------------

    def solve(
        self,
        b0: np.ndarray,
        initial_control: Tuple[np.ndarray, np.ndarray],
        plot_callback=None
    ) -> Dict[str, Any]:
        """
        Solve the optimal control problem using the Forward-Backward Sweep Method.

        Args:
            b0: Initial belief distribution
            initial_control: Initial guess for control (u_x, u_y)
            plot_callback: Optional callable(solver, x, y, b, q, Q, lx, ly, lb,
                           ig_ux, ig_uy, ux, uy, iteration) for plotting

        Returns:
            Dictionary containing solution: x, y, b, q, Q, lx, ly, lb, ux, uy,
            t, iterations, converged
        """
        ig_ux, ig_uy = initial_control
        omega = self.sim.omega_init
        err_prev = np.inf

        print(f"Starting FBSM with {self.sim.max_iter} max iterations...")
        print(f"{'Iter':>4} {'RMS Error':>12} {'Omega':>10} {'Status':>20}")
        print("-" * 50)

        for i in range(self.sim.max_iter):
            # Forward sweep
            x, y, b, q, Q = self.forward_state_integration(b0, ig_ux, ig_uy)

            # Backward sweep
            lx, ly, lb = self.backward_costate_integration(x, y, b, q, Q)

            # Optimal control
            ux, uy = self.compute_optimal_control(lx, ly)

            # Optional per-iteration plot
            if plot_callback is not None:
                if self.sim.plot_every > 0 and (
                    i % self.sim.plot_every == 0 or i == self.sim.max_iter - 1
                ):
                    plot_callback(self, x, y, b, q, Q, lx, ly, lb, ig_ux, ig_uy, ux, uy, i)

            # Relaxation with speed constraint
            tempx = omega * ux + (1 - omega) * ig_ux
            tempy = omega * uy + (1 - omega) * ig_uy
            tempv = np.sqrt(tempx ** 2 + tempy ** 2) + 1e-12

            up_ux = self.vehicle.v * tempx / tempv
            up_uy = self.vehicle.v * tempy / tempv

            # Control error
            err = np.sum(np.sqrt((up_ux - ig_ux) ** 2 + (up_uy - ig_uy) ** 2))

            # Convergence check
            if err < self.sim.convergence_tol:
                print(f"{i:4d} {err:12.4f} {omega:10.4f} {'CONVERGED':>20}")
                print("-" * 50)
                print(f"✓ Converged at iteration {i}")
                break

            # Adaptive omega
            if i > 0:
                if err > err_prev:
                    omega = max(self.sim.omega_min, self.sim.omega_adj_rate * omega)
                    status = "Decreasing omega"
                else:
                    status = "Keeping omega"
            else:
                status = "Initializing"

            print(f"{i:4d} {err:12.4f} {omega:10.4f} {status:>20}")

            err_prev = err
            ig_ux = up_ux
            ig_uy = up_uy

        else:
            warnings.warn(f"Maximum iterations ({self.sim.max_iter}) reached without convergence")

        # Final plot (only-final mode)
        if plot_callback is not None and self.sim.plot_every == 0:
            plot_callback(self, x, y, b, q, Q, lx, ly, lb, ig_ux, ig_uy, ux, uy, i)

        return {
            'x': x, 'y': y, 'b': b, 'q': q, 'Q': Q,
            'lx': lx, 'ly': ly, 'lb': lb,
            'ux': ig_ux, 'uy': ig_uy,
            't': self.t,
            'iterations': i + 1,
            'converged': err < self.sim.convergence_tol
        }
