import numpy as np

class ExhaustiveSweepPlanner:
    def __init__(self, angles_to_sweep: list):
        self.angles = angles_to_sweep
        self.current_idx = 0

    def get_next_beam(self, belief_probs: np.ndarray, grid_angles: np.ndarray):
        if self.current_idx >= len(self.angles):
            self.current_idx = 0 # loop back

        angle = self.angles[self.current_idx]
        self.current_idx += 1

        # Returns (beam_type, target_angle, width_or_nulls)
        return "pencil", angle, 0.0

class RandomProbingPlanner:
    def __init__(self, angle_range: tuple = (-90, 90)):
        self.angle_range = angle_range

    def get_next_beam(self, belief_probs: np.ndarray, grid_angles: np.ndarray):
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        return "pencil", angle, 0.0

class GreedyEntropyPlanner:
    def __init__(self, candidate_angles: list, updater, ula, tx_lin, pl_lin, grid_angles):
        self.candidates = candidate_angles
        self.updater = updater
        self.ula = ula
        self.tx_lin = tx_lin
        self.pl_lin = pl_lin
        self.grid_angles = grid_angles

    def get_next_beam(self, belief_probs: np.ndarray, grid_angles: np.ndarray):
        best_angle = None
        max_var = -1

        for ang in self.candidates:
            w = self.ula.steering_vector(ang)
            gains = self.ula.power_pattern(w, self.grid_angles)
            expected_powers = (self.tx_lin / self.pl_lin) * gains

            mean_power = np.sum(belief_probs * expected_powers)
            var_power = np.sum(belief_probs * (expected_powers - mean_power)**2)

            if var_power > max_var:
                max_var = var_power
                best_angle = ang

        if best_angle is None:
            best_angle = self.candidates[0]

        return "pencil", best_angle, 0.0

class HierarchicalNarrowingPlanner:
    def __init__(self, initial_bounds: tuple = (-90, 90)):
        self.bounds = initial_bounds

    def get_next_beam(self, belief_probs: np.ndarray, grid_angles: np.ndarray):
        """
        Probabilistic binary partitioning:
        Find the region containing highest uncertainty and bisect it.
        We adapt the beamwidth based on the entropy/mass of the belief.
        """
        # Calculate cumulative distribution function of belief
        cdf = np.cumsum(belief_probs)

        # Find the interquartile range (25th to 75th percentile)
        # This gives us a good bound on where the mass is located.
        idx_25 = np.searchsorted(cdf, 0.25)
        idx_75 = np.searchsorted(cdf, 0.75)

        # Ensure we don't go out of bounds
        idx_25 = np.clip(idx_25, 0, len(grid_angles)-1)
        idx_75 = np.clip(idx_75, 0, len(grid_angles)-1)

        ang_25 = grid_angles[idx_25]
        ang_75 = grid_angles[idx_75]

        width = ang_75 - ang_25
        center = (ang_75 + ang_25) / 2.0

        # If the probability mass is very concentrated, we switch to a pencil beam at the mode
        entropy = -np.sum(np.clip(belief_probs, 1e-12, 1) * np.log2(np.clip(belief_probs, 1e-12, 1)))

        # Simple heuristic:
        # High entropy -> wide sector beam
        # Low entropy -> pencil beam
        if entropy < 3.0 or width < 5.0:
            mode_idx = np.argmax(belief_probs)
            return "pencil", grid_angles[mode_idx], 0.0

        # Ensure the sector width is at least something reasonable to synthesize
        width = max(width, 10.0)

        # In a true binary partition, we would probe one half of the high-mass region.
        # Let's probe the left half of the 25-75 region.
        # This will either confirm it's in the left half, or implicitly update that it's in the right half.
        probe_center = ang_25 + width/4.0
        probe_width = width/2.0

        return "sector", probe_center, probe_width
