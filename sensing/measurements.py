import numpy as np
from utils.propagation import dbm_to_linear, linear_to_dbm, friis_path_loss_linear

class MeasurementModel:
    def __init__(self, array, noise_power_dbm: float = -90.0):
        self.array = array
        self.noise_power = dbm_to_linear(noise_power_dbm) # Watts

    def _generate_fading(self, size, fading_type='los', k_factor_db=10.0):
        fading_type = fading_type.lower()
        if fading_type == 'los':
            random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size))
            return random_phase
        elif fading_type == 'rayleigh':
            return (np.random.randn(*size) + 1j * np.random.randn(*size)) / np.sqrt(2)
        elif fading_type == 'rician':
            K = 10 ** (k_factor_db / 10.0)
            mu = np.sqrt(K / (K + 1))
            sigma = np.sqrt(1 / (2 * (K + 1)))
            los_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size))
            los_component = mu * los_phase
            scattered = sigma * (np.random.randn(*size) + 1j * np.random.randn(*size))
            return los_component + scattered
        else:
            raise ValueError(f"Unknown fading type: {fading_type}")

    def measure(self, target_angle: float, beam_weights: np.ndarray,
                tx_power_dbm: float = 0.0, distance_m: float = 100.0,
                fading_type: str = 'los', k_factor_db: float = 10.0) -> dict:

        # 1. Base received power (Isotropic element)
        tx_power_watts = dbm_to_linear(tx_power_dbm)
        attenuation = friis_path_loss_linear(distance_m, self.array.wavelength)
        rx_power_element = tx_power_watts * attenuation

        # 2. Spatial array factor / beam gain
        beam_gain_linear = self.array.power_pattern(beam_weights, np.array([target_angle]))[0]

        # 3. Fading
        h = self._generate_fading((1,), fading_type, k_factor_db)[0]

        # 4. Clean signal power after beamforming and fading
        rx_power_faded = rx_power_element * beam_gain_linear * (np.abs(h) ** 2)

        # 5. CWGN
        noise_real = np.random.randn() * np.sqrt(self.noise_power / 2)
        noise_imag = np.random.randn() * np.sqrt(self.noise_power / 2)
        noise = noise_real + 1j * noise_imag

        # 6. Signal voltage combination
        # Signal voltage amplitude is sqrt(power). Apply channel phase.
        signal = np.sqrt(rx_power_element * beam_gain_linear) * h
        rx_signal = signal + noise

        # Measured Power (Watts)
        measured_power = np.abs(rx_signal) ** 2

        # 7. Metrics calculation
        rssi_dbm = linear_to_dbm(measured_power)
        snr_db = 10 * np.log10((rx_power_faded + 1e-15) / self.noise_power)

        return {
            "rssi_dbm": rssi_dbm,
            "snr_db": float(snr_db),
            "measured_power_watts": measured_power,
            "expected_faded_power_watts": rx_power_faded,
            "beam_gain_linear": beam_gain_linear,
            "fading_coeff": h,
            "attenuation": attenuation
        }
