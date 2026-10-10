"""NumPy-only, box-constrained linear MPC in [right, forward, down] axes.

Prediction uses the same semi-implicit first-order velocity model as LandingEnv.
An ADMM QP solver optimizes position/velocity tracking and command
increments. Command bounds and horizon-wide slew limits are hard constraints;
predicted states are not hard bounded. A velocity-aided observer rejects jitter.
"""
from __future__ import annotations

import numpy as np


class LandingMPC:
    def __init__(self, dt=0.05, horizon=40, tau_xy=1.3, tau_z=0.95,
                 v_xy_max=2.0, v_down_max=1.5, v_up_max=0.8,
                 descent_speed=0.8, align_radius=0.3, iterations=80,
                 max_accel_xy=0.8, max_accel_z=0.4, position_filter_tau=0.25,
                 velocity_filter_tau=0.1, descent_filter_tau=0.4,
                 command_smooth_w=4.0):
        values = [dt, tau_xy, tau_z, v_xy_max, v_down_max, v_up_max,
                  descent_speed, align_radius, max_accel_xy, max_accel_z,
                  descent_filter_tau, command_smooth_w]
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("MPC time constants, limits and descent parameters must be positive")
        if horizon < 2 or iterations < 1 or dt > min(tau_xy, tau_z):
            raise ValueError("Invalid MPC horizon/iterations or dt exceeds time constant")
        if not np.all(np.isfinite([position_filter_tau, velocity_filter_tau])) or min(
                position_filter_tau, velocity_filter_tau) < 0:
            raise ValueError("Observation filter time constants must be finite and nonnegative")
        self.max_delta = dt * np.array([max_accel_xy, max_accel_xy, max_accel_z])
        self.position_filter_tau = position_filter_tau
        self.velocity_filter_tau = velocity_filter_tau
        self.descent_filter_tau = descent_filter_tau
        self.command_smooth_w = command_smooth_w
        self.dt, self.horizon = dt, int(horizon)
        self.iterations = int(iterations)
        self.descent_speed, self.align_radius = descent_speed, align_radius
        # Landing MPC descends or brakes to hover; it does not request climb.
        self.lower = np.array([-v_xy_max, -v_xy_max, 0.0])
        self.upper = np.array([v_xy_max, v_xy_max, min(v_down_max, descent_speed)])
        self.models = []
        self.solvers = []
        n = self.horizon
        difference = np.eye(n) - np.eye(n, k=-1)
        for tau, sign, qp, qv in [(tau_xy, -1, 8., 2.),
                                  (tau_xy, -1, 8., 2.), (tau_z, -1, 6., 3.)]:
            beta = dt / tau
            A = np.array([[1., sign * dt * (1. - beta)], [0., 1. - beta]])
            B = np.array([sign * dt * beta, beta])
            F, G = np.zeros((2*n, 2)), np.zeros((2*n, n))
            for k in range(n):
                F[2*k:2*k+2] = np.linalg.matrix_power(A, k+1)
                for j in range(k+1):
                    G[2*k:2*k+2, j] = np.linalg.matrix_power(A, k-j) @ B
            weights = np.tile([qp, qv], n)
            weights[-2:] *= 4.
            H = G.T @ (weights[:, None] * G) + 0.1*np.eye(n) + command_smooth_w*difference.T @ difference
            self.models.append((F, G, weights, H, difference))
            constraints = np.vstack([np.eye(n), difference])
            rho = 5.0
            inverse = np.linalg.inv(H + rho*constraints.T @ constraints)
            self.solvers.append((constraints, inverse, rho))
        self.reset()

    def reset(self):
        self.sequence = np.zeros((3, self.horizon))
        self.previous = np.zeros(3)
        self.filtered_position = None
        self.filtered_velocity = None
        self.last_measurement_stamp = None
        self.filtered_down = 0.0
        self.dual = np.zeros((3, 2*self.horizon))

    def command(self, p_cam, velocity, measurement_stamp=None):
        """Return physical velocity command; invalid observations fail to zero."""
        p = np.asarray(p_cam, dtype=float)
        v = np.asarray(velocity, dtype=float)
        if p.shape != (3,) or v.shape != (3,) or not np.all(np.isfinite([p, v])):
            self.reset()
            return np.zeros(3)
        # Predict the stationary marker's relative position using measured
        # velocity, then correct only on a new camera frame. This avoids treating
        # a low-rate held camera sample as a new measurement on every timer tick.
        if self.filtered_position is None:
            self.filtered_position = p.copy()
            self.filtered_velocity = v.copy()
        else:
            alpha_v = self.dt / (self.velocity_filter_tau + self.dt)
            self.filtered_velocity += alpha_v * (v - self.filtered_velocity)
            self.filtered_position -= self.filtered_velocity * self.dt
            if measurement_stamp is None or measurement_stamp != self.last_measurement_stamp:
                if measurement_stamp is not None and self.last_measurement_stamp is not None:
                    elapsed = max(self.dt, float(measurement_stamp - self.last_measurement_stamp)*1e-9)
                else:
                    elapsed = self.dt
                alpha_p = elapsed / (self.position_filter_tau + elapsed)
                self.filtered_position += alpha_p * (p - self.filtered_position)
        if measurement_stamp is not None:
            self.last_measurement_stamp = measurement_stamp
        p, v = self.filtered_position, self.filtered_velocity
        # Gradually permit descent as the target is aligned. Cap terminal descent
        # at 0.2 m/s, below LandingEnv's 0.5 m/s total touchdown speed criterion.
        # Vision noise grows with depth; a fixed low-altitude alignment radius
        # would prevent descent at high altitude even when physically centered.
        radius = self.align_radius + 0.1 * max(float(p[2]), 0.)
        gate = np.clip(1. - (np.linalg.norm(p[:2])/radius)**2, 0., 1.)
        target_down = min(self.upper[2], max(0.2, 0.5*max(float(p[2]), 0.))) * gate
        alpha_down = self.dt / (self.descent_filter_tau + self.dt)
        self.filtered_down += alpha_down * (target_down - self.filtered_down)
        h_ref = max(float(p[2]), 0.)
        reference = np.zeros((3, 2*self.horizon))
        for k in range(self.horizon):
            down = min(self.filtered_down, max(0.2, 0.5*h_ref))
            h_ref = max(0., h_ref - down*self.dt)
            reference[2, 2*k:2*k+2] = [h_ref, down]
        for axis, (F, G, weights, _H, D) in enumerate(self.models):
            offset = F @ np.array([p[axis], v[axis]]) - reference[axis]
            previous = np.zeros(self.horizon)
            previous[0] = self.previous[axis]
            gradient_offset = G.T @ (weights * offset) - self.command_smooth_w*D.T @ previous
            u = np.r_[self.sequence[axis, 1:], self.sequence[axis, -1]]
            C, inverse, rho = self.solvers[axis]
            lower = np.r_[np.full(self.horizon, self.lower[axis]),
                          np.full(self.horizon, -self.max_delta[axis])]
            upper = np.r_[np.full(self.horizon, self.upper[axis]),
                          np.full(self.horizon, self.max_delta[axis])]
            lower[self.horizon] += self.previous[axis]
            upper[self.horizon] += self.previous[axis]
            dual = self.dual[axis].copy()
            z = np.clip(C @ u, lower, upper)
            for _ in range(self.iterations):
                u = inverse @ (-gradient_offset + rho*C.T @ (z-dual))
                projected = C @ u
                relaxed = 1.6*projected - 0.6*z
                z_new = np.clip(relaxed + dual, lower, upper)
                dual += relaxed - z_new
                converged = np.max(np.abs(projected-z_new)) < 1e-5 \
                    and np.max(np.abs(z_new-z)) < 1e-5
                z = z_new
                if converged:
                    break
            self.dual[axis] = dual
            # Finite ADMM iterations can leave a small constraint residual.
            # Return a feasible sequence even if the solver did not converge.
            last = self.previous[axis]
            for k in range(self.horizon):
                u[k] = np.clip(u[k], max(self.lower[axis], last-self.max_delta[axis]),
                               min(self.upper[axis], last+self.max_delta[axis]))
                last = u[k]
            self.sequence[axis] = u
        self.previous = self.sequence[:, 0].copy()
        return self.previous.copy()
