import numpy as np

class UniformLinearArray:
    def __init__(self, num_elements: int, frequency: float = 5.4e9, spacing_factor: float = 0.5):
        self.num_elements = num_elements
        self.frequency = frequency
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / self.frequency
        self.spacing = spacing_factor * self.wavelength
        self.positions = np.arange(self.num_elements) * self.spacing
        self.positions -= np.mean(self.positions)

    def steering_vector(self, angle_deg: float) -> np.ndarray:
        angle_rad = np.radians(angle_deg)
        k = 2 * np.pi / self.wavelength
        phases = k * self.positions * np.sin(angle_rad)
        return np.exp(-1j * phases) / np.sqrt(self.num_elements)

    def apply_phase_quantization(self, weights: np.ndarray, bits: int) -> np.ndarray:
        if bits is None or bits <= 0:
            return weights
        num_states = 2**bits
        phases = np.angle(weights)
        phases = np.mod(phases, 2*np.pi)
        step = 2*np.pi / num_states
        quantized_phases = np.round(phases / step) * step
        magnitudes = np.abs(weights)
        return magnitudes * np.exp(1j * quantized_phases)

    def array_factor(self, weights: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
        angles_rad = np.radians(angles_deg)
        k = 2 * np.pi / self.wavelength
        phases = k * np.outer(self.positions, np.sin(angles_rad))
        af = weights.conj().T @ np.exp(-1j * phases)
        return af

    def power_pattern(self, weights: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
        af = self.array_factor(weights, angles_deg)
        power = np.abs(af)**2
        # Normalize by num_elements^2 because maximum array factor is num_elements, and its power is num_elements^2
        # The weights are usually normalized such that norm(w) = 1, in which case max power is roughly 1.0 depending on the weights.
        # However, to ensure single-angle requests aren't normalized individually to 1:
        # For a steering vector, its max power is 1.0 (if w is 1/sqrt(N) * exp(...))
        # But we want to normalize globally for the given weights.
        # The maximum possible gain for weights w with norm(w)=1 is N when w is perfectly matched.
        # But let's keep it simple: normalize by the maximum possible for a matched filter.
        # if max power over the array is known, we can divide by it. For single angle, this breaks.
        # So we should not divide by max(power) here. We can divide by maximum possible power, which is N for w of norm 1.
        # Let's just return power, and let the user normalize if they want a relative pattern.
        # Or better, for a general weights vector, max power occurs when w matches steering vector.
        # If w is normalized to 1, max power is N * (1/sqrt(N))^2 = 1? No, max AF is sqrt(N). Max power is N.
        # For simplicity, just return the raw power if it's a single point, or just return raw power overall.
        return power

class UniformPlanarArray:
    def __init__(self, num_elements_x: int, num_elements_y: int, frequency: float = 5.4e9, spacing_factor: float = 0.5):
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y
        self.num_elements = num_elements_x * num_elements_y
        self.frequency = frequency
        self.c = 3e8
        self.wavelength = self.c / self.frequency
        self.spacing = spacing_factor * self.wavelength

        x = np.arange(num_elements_x) * self.spacing
        y = np.arange(num_elements_y) * self.spacing
        x -= np.mean(x)
        y -= np.mean(y)

        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.positions_x = xx.flatten()
        self.positions_y = yy.flatten()

    def steering_vector(self, theta_deg: float, phi_deg: float) -> np.ndarray:
        theta_rad = np.radians(theta_deg)
        phi_rad = np.radians(phi_deg)
        k = 2 * np.pi / self.wavelength
        u = np.sin(theta_rad) * np.cos(phi_rad)
        v = np.sin(theta_rad) * np.sin(phi_rad)
        phases = k * (self.positions_x * u + self.positions_y * v)
        return np.exp(-1j * phases) / np.sqrt(self.num_elements)

    def apply_phase_quantization(self, weights: np.ndarray, bits: int) -> np.ndarray:
        if bits is None or bits <= 0:
            return weights
        num_states = 2**bits
        phases = np.angle(weights)
        phases = np.mod(phases, 2*np.pi)
        step = 2*np.pi / num_states
        quantized_phases = np.round(phases / step) * step
        magnitudes = np.abs(weights)
        return magnitudes * np.exp(1j * quantized_phases)

    def array_factor(self, weights: np.ndarray, theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
        theta_rad = np.radians(theta_deg)
        phi_rad = np.radians(phi_deg)
        k = 2 * np.pi / self.wavelength
        u = np.sin(theta_rad) * np.cos(phi_rad)
        v = np.sin(theta_rad) * np.sin(phi_rad)
        phases = k * (np.outer(self.positions_x, u) + np.outer(self.positions_y, v))
        af = weights.conj().T @ np.exp(-1j * phases)
        return af

    def power_pattern(self, weights: np.ndarray, theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
        af = self.array_factor(weights, theta_deg, phi_deg)
        return np.abs(af)**2
