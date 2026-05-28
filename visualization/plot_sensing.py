import numpy as np
import matplotlib.pyplot as plt
import os
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from sensing.measurements import MeasurementModel

def plot_snr_distribution():
    ula = UniformLinearArray(num_elements=16)
    synth = BeamSynthesizer(ula)
    w = synth.synthesize_pencil_beam(0.0)

    model = MeasurementModel(ula, noise_power_dbm=-100)

    angles = np.linspace(-90, 90, 181)
    snrs_los = []

    for ang in angles:
        meas = model.measure(ang, w, tx_power_dbm=10, path_loss_db=80, fading_type='los')
        snrs_los.append(meas['snr_db'])

    plt.figure(figsize=(8,5))
    plt.plot(angles, snrs_los)
    plt.title("Expected SNR vs Angle (LOS Fading)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("SNR (dB)")
    plt.grid(True)
    plt.savefig("reports/sensing_snr_vs_angle.png")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    plot_snr_distribution()
    print("Plots saved to reports/")
