import numpy as np

class GridBelief:
    def __init__(self, angles_deg: np.ndarray):
        """
        Initialize a discrete grid-based belief over angles.
        angles_deg: array of possible angles (e.g. -90 to 90)
        """
        self.angles = angles_deg
        self.num_states = len(self.angles)
        # Uniform prior
        self.probs = np.ones(self.num_states) / self.num_states

    def get_entropy(self) -> float:
        """Compute Shannon entropy of the belief."""
        # Avoid log(0)
        p = np.clip(self.probs, 1e-12, 1.0)
        return -np.sum(p * np.log2(p))

    def reset(self):
        """Reset to uniform prior."""
        self.probs = np.ones(self.num_states) / self.num_states

    def update(self, likelihoods: np.ndarray):
        """
        Perform a Bayesian update step.
        likelihoods: array of P(measurement | angle) of shape (num_states,)
        """
        unnormalized_posterior = self.probs * likelihoods
        sum_p = np.sum(unnormalized_posterior)

        if sum_p < 1e-15:
            # Belief collapse failure, just add uniform noise
            self.probs = np.ones(self.num_states) / self.num_states
        else:
            self.probs = unnormalized_posterior / sum_p

class BayesianUpdater:
    def __init__(self, array, noise_power_linear: float):
        self.array = array
        self.noise_power = noise_power_linear

    def compute_likelihood(self, measured_power: float, beam_weights: np.ndarray, angles: np.ndarray, tx_power_linear: float, path_loss_linear: float) -> np.ndarray:
        """
        Compute likelihood P(measurement | angle) for all angles.
        Assuming Gaussian noise on the non-coherent energy measurement for simplicity in this stage.
        """
        # Expected signal power at each angle
        # power_pattern returns power gain linear
        gains = self.array.power_pattern(beam_weights, angles)

        # Expected received power (assuming LOS for the likelihood model)
        expected_rx_power = (tx_power_linear / path_loss_linear) * gains

        # We model the measurement roughly as Gaussian around the expected power
        # Note: True non-coherent energy detection with fading is more complex (Chi-square / Exponential).
        # We use a Gaussian approximation for the sake of the update engine.
        variance = 2 * self.noise_power * expected_rx_power + self.noise_power**2

        # Prevent division by zero
        variance = np.clip(variance, 1e-12, None)

        diff = measured_power - expected_rx_power
        # log-likelihood to prevent underflow
        log_lh = -0.5 * (diff**2) / variance - 0.5 * np.log(2 * np.pi * variance)

        # shift log_lh to max = 0 before exp for stability
        log_lh -= np.max(log_lh)
        likelihoods = np.exp(log_lh)

        # Ensure it's not exactly zero anywhere
        return np.clip(likelihoods, 1e-300, None)
