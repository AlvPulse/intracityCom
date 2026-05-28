import numpy as np
import matplotlib.pyplot as plt
import os
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from inference.belief import BayesianUpdater
from sensing.measurements import MeasurementModel
from utils.propagation import dbm_to_linear, friis_path_loss_linear

def plot_observability():
    ula = UniformLinearArray(num_elements=16)
    synth = BeamSynthesizer(ula)
    angles = np.linspace(-90, 90, 181)

    noise_dbm = -100
    noise_watts = dbm_to_linear(noise_dbm)
    tx_power_dbm = 10
    tx_watts = dbm_to_linear(tx_power_dbm)

    model = MeasurementModel(ula, noise_power_dbm=noise_dbm)
    updater = BayesianUpdater(ula, noise_watts)

    w = synth.synthesize_pencil_beam(0.0)

    distances = [10.0, 5000.0, 50000.0] # Easy, Medium, Hard/Impossible

    plt.figure(figsize=(15, 5))

    for i, dist in enumerate(distances):
        meas = model.measure(0.0, w, tx_power_dbm, dist, 'los')

        # Expected powers for all angles
        gains = ula.power_pattern(w, angles)
        attenuation = friis_path_loss_linear(dist, ula.wavelength)
        expected_rx_power = tx_watts * attenuation * gains

        likelihoods = updater.compute_likelihood(meas["measured_power_watts"], w, angles, tx_watts, dist)

        mean_lh = np.mean(likelihoods)
        sharpness = np.max(likelihoods) / mean_lh if mean_lh > 0 else 1.0

        snr_peak_db = 10 * np.log10(np.max(expected_rx_power) / noise_watts + 1e-15)

        plt.subplot(1, 3, i+1)
        plt.plot(angles, likelihoods)
        plt.title(f"Dist: {dist}m | Peak SNR: {snr_peak_db:.1f} dB\nSharpness: {sharpness:.2f}")
        plt.xlabel("Angle (degrees)")
        if i == 0:
            plt.ylabel("Normalized Likelihood")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("reports/likelihood_sharpness.png")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    plot_observability()
    print("Plots saved to reports/likelihood_sharpness.png")
