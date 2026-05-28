import numpy as np
from utils.propagation import friis_path_loss_linear

class GridBelief:
    def __init__(self, angles_deg: np.ndarray):
        self.angles = angles_deg
        self.num_states = len(self.angles)
        self.probs = np.ones(self.num_states) / self.num_states

    def get_entropy(self) -> float:
        p = np.clip(self.probs, 1e-12, 1.0)
        return -np.sum(p * np.log2(p))

    def reset(self):
        self.probs = np.ones(self.num_states) / self.num_states

    def update(self, likelihoods: np.ndarray):
        unnormalized_posterior = self.probs * likelihoods
        sum_p = np.sum(unnormalized_posterior)

        if sum_p < 1e-15:
            # Belief collapse, reset to uniform
            self.probs = np.ones(self.num_states) / self.num_states
        else:
            self.probs = unnormalized_posterior / sum_p

class BayesianUpdater:
    def __init__(self, array, noise_power_watts: float):
        self.array = array
        self.noise_power = noise_power_watts

    def compute_likelihood(self, measured_power_watts: float, beam_weights: np.ndarray, angles: np.ndarray,
                           tx_power_watts: float, distance_m: float) -> np.ndarray:
        """
        Compute likelihood P(measurement | angle).
        Using a normalized log-likelihood proxy to prevent numerical collapse.
        """
        # Expected gains for all angles
        gains = self.array.power_pattern(beam_weights, angles)

        # Expected received power (Watts) assuming LOS
        attenuation = friis_path_loss_linear(distance_m, self.array.wavelength)
        expected_rx_power = tx_power_watts * attenuation * gains

        # To avoid the variance of measured power completely wiping out likelihoods when signal << noise
        # We work in normalized SNR domain.
        # Variance of the non-coherent energy detection measurement is roughly 2 * sigma^2 * P_sig + sigma^4
        # We use a Gaussian approximation.
        variance = 2 * self.noise_power * expected_rx_power + self.noise_power**2
        variance = np.clip(variance, 1e-30, None)

        diff = measured_power_watts - expected_rx_power

        # log-likelihood
        log_lh = -0.5 * (diff**2) / variance - 0.5 * np.log(2 * np.pi * variance)

        # Sharpness sanity check logic (can be logged externally)
        # Shift to max=0 for stability
        log_lh -= np.max(log_lh)

        likelihoods = np.exp(log_lh)
        return np.clip(likelihoods, 1e-300, None)
