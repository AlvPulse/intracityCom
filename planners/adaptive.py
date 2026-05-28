import numpy as np

class ExhaustiveSweepPlanner:
    def __init__(self, angles_to_sweep: list):
        self.angles = angles_to_sweep
        self.current_idx = 0

    def get_next_beam(self, belief_probs: np.ndarray = None):
        if self.current_idx >= len(self.angles):
            self.current_idx = 0 # loop back

        angle = self.angles[self.current_idx]
        self.current_idx += 1

        # Returns (beam_type, target_angle, null_angles)
        return "pencil", angle, []

class RandomProbingPlanner:
    def __init__(self, angle_range: tuple = (-90, 90)):
        self.angle_range = angle_range

    def get_next_beam(self, belief_probs: np.ndarray = None):
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        return "pencil", angle, []

class GreedyEntropyPlanner:
    def __init__(self, candidate_angles: list, updater, ula, tx_lin, pl_lin, grid_angles):
        self.candidates = candidate_angles
        self.updater = updater
        self.ula = ula
        self.tx_lin = tx_lin
        self.pl_lin = pl_lin
        self.grid_angles = grid_angles

    def get_next_beam(self, belief_probs: np.ndarray):
        """
        Choose the beam angle that minimizes expected posterior entropy.
        To approximate, we choose the beam that maximizes the variance of the received power
        over the current belief, which intuitively maximizes distinguishability.
        """
        best_angle = None
        max_var = -1

        for ang in self.candidates:
            # We assume pencil beams for simplicity here
            w = self.ula.steering_vector(ang)
            gains = self.ula.power_pattern(w, self.grid_angles)
            expected_powers = (self.tx_lin / self.pl_lin) * gains

            # Variance of expected measurement under current belief
            mean_power = np.sum(belief_probs * expected_powers)
            var_power = np.sum(belief_probs * (expected_powers - mean_power)**2)

            if var_power > max_var:
                max_var = var_power
                best_angle = ang

        if best_angle is None:
            best_angle = self.candidates[0]

        return "pencil", best_angle, []

class HierarchicalNarrowingPlanner:
    def __init__(self, initial_bounds: tuple = (-90, 90)):
        self.bounds = initial_bounds

    def get_next_beam(self, belief_probs: np.ndarray, grid_angles: np.ndarray):
        """
        Find the region containing 90% of the probability mass and create a sector beam for it.
        If the region is small enough, switch to pencil beam at the mode.
        """
        mode_idx = np.argmax(belief_probs)
        mode_ang = grid_angles[mode_idx]

        # Very simple hierarchical logic:
        # Instead of actually computing sector width (which requires full sector synthesis),
        # we just probe at the mode. This is essentially 'Posterior Matching'.
        # A true hierarchical planner would use sector beams to bisect space.

        # We will return the mode for a pencil beam as a simplified version
        return "pencil", mode_ang, []
