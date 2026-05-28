import numpy as np

class MeasurementModel:
    def __init__(self, array, noise_power_dbm: float = -90.0):
        self.array = array
        self.noise_power = 10 ** (noise_power_dbm / 10.0) # linear scale (Watts)

    def _generate_fading(self, size, fading_type='los', k_factor_db=10.0):
        fading_type = fading_type.lower()
        if fading_type == 'los':
            # Randomized baseline phase to represent absolute distance variation
            random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size))
            return random_phase

        elif fading_type == 'rayleigh':
            return (np.random.randn(*size) + 1j * np.random.randn(*size)) / np.sqrt(2)

        elif fading_type == 'rician':
            K = 10 ** (k_factor_db / 10.0)
            mu = np.sqrt(K / (K + 1))
            sigma = np.sqrt(1 / (2 * (K + 1)))

            # LoS component gets a random phase shift
            los_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size))
            los_component = mu * los_phase

            # Scattered (Rayleigh) component
            scattered = sigma * (np.random.randn(*size) + 1j * np.random.randn(*size))
            return los_component + scattered
        else:
            raise ValueError(f"Unknown fading type: {fading_type}")

    def measure(self, target_angle: float, beam_weights: np.ndarray,
                tx_power_dbm: float = 0.0, path_loss_db: float = 80.0,
                fading_type: str = 'los', k_factor_db: float = 10.0) -> dict:

        # 1. Calculate base received power before array gain (at the element level)
        tx_power = 10 ** (tx_power_dbm / 10.0)
        path_loss = 10 ** (path_loss_db / 10.0)
        rx_power_element = tx_power / path_loss

        # 2. Get spatial array factor / beam gain
        # Using the standard power_pattern response
        beam_gain_linear = self.array.power_pattern(beam_weights, np.array([target_angle]))[0]

        # 3. Generate fading channel coefficient
        h = self._generate_fading((1,), fading_type, k_factor_db)[0]

        # 4. Calculate exact clean signal power after beamforming and fading
        rx_power_faded = rx_power_element * beam_gain_linear * (np.abs(h) ** 2)

        # 5. Generate Complex White Gaussian Noise (CWGN)
        noise_real = np.random.randn() * np.sqrt(self.noise_power / 2)
        noise_imag = np.random.randn() * np.sqrt(self.noise_power / 2)
        noise = noise_real + 1j * noise_imag

        # 6. Signal voltage + Noise voltage combination
        # The received signal voltage amplitude is sqrt(rx_power_faded)
        # Apply the channel phase (already in h, so we can use it or just assume arbitrary phase)
        # Since h includes the phase shift, we can apply it to the signal
        # Clean signal voltage (complex):
        signal = np.sqrt(rx_power_element * beam_gain_linear) * h
        rx_signal = signal + noise

        # Measured Power is the true total power (Signal + Noise)
        measured_power = np.abs(rx_signal) ** 2

        # 7. Metrics calculation
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
