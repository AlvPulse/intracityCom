import numpy as np

class MeasurementModel:
    def __init__(self, array, noise_power_dbm: float = -90.0):
        self.array = array
        self.noise_power = 10 ** (noise_power_dbm / 10.0) # linear scale

    def _generate_fading(self, size, fading_type='los', k_factor_db=10.0):
        if fading_type.lower() == 'los':
            return np.ones(size, dtype=complex)
        elif fading_type.lower() == 'rayleigh':
            return (np.random.randn(*size) + 1j * np.random.randn(*size)) / np.sqrt(2)
        elif fading_type.lower() == 'rician':
            K = 10 ** (k_factor_db / 10.0)
            mu = np.sqrt(K / (K + 1))
            sigma = np.sqrt(1 / (2 * (K + 1)))
            x = mu + sigma * np.random.randn(*size)
            y = sigma * np.random.randn(*size)
            return x + 1j * y
        else:
            raise ValueError(f"Unknown fading type: {fading_type}")

    def measure(self, target_angle: float, beam_weights: np.ndarray,
                tx_power_dbm: float = 0.0, path_loss_db: float = 80.0,
                fading_type: str = 'los', k_factor_db: float = 10.0) -> dict:
        beam_gain_linear = self.array.power_pattern(beam_weights, np.array([target_angle]))[0]
        tx_power = 10 ** (tx_power_dbm / 10.0)
        path_loss = 10 ** (path_loss_db / 10.0)

        rx_power_base = (tx_power / path_loss) * beam_gain_linear
        h = self._generate_fading((1,), fading_type, k_factor_db)[0]
        rx_power_faded = rx_power_base * np.abs(h)**2

        noise_real = np.random.randn() * np.sqrt(self.noise_power / 2)
        noise_imag = np.random.randn() * np.sqrt(self.noise_power / 2)

        rx_signal = np.sqrt(rx_power_faded) + (noise_real + 1j * noise_imag)
        measured_power = np.abs(rx_signal)**2

        rssi_dbm = 10 * np.log10(measured_power + 1e-12)
        snr_db = 10 * np.log10((rx_power_faded + 1e-12) / self.noise_power)

        return {
            "rssi_dbm": rssi_dbm,
            "snr_db": snr_db,
            "measured_power": measured_power,
            "expected_faded_power": rx_power_faded,
            "beam_gain_linear": beam_gain_linear,
            "fading_coeff": h
        }
