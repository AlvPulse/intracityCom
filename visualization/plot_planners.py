import numpy as np
import matplotlib.pyplot as plt
import os
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from inference.belief import GridBelief, BayesianUpdater
from sensing.measurements import MeasurementModel
from planners.adaptive import GreedyEntropyPlanner
from utils.propagation import dbm_to_linear, friis_path_loss_linear

def simulate_adaptive_probing():
    ula = UniformLinearArray(num_elements=16)
    synth = BeamSynthesizer(ula)

    angles = np.linspace(-90, 90, 181)
    belief = GridBelief(angles)

    noise_dbm = -100
    noise_watts = dbm_to_linear(noise_dbm)

    updater = BayesianUpdater(ula, noise_watts)
    model = MeasurementModel(ula, noise_power_dbm=noise_dbm)

    tx_power_dbm = 10
    tx_watts = dbm_to_linear(tx_power_dbm)
    distance_m = 10.0
    attenuation = friis_path_loss_linear(distance_m, ula.wavelength)

    true_angle = 45.0

    candidates = np.linspace(-90, 90, 31)
    planner = GreedyEntropyPlanner(candidates, updater, ula, tx_watts, 1.0/attenuation, angles)

    entropies = [belief.get_entropy()]
    probe_angles = []

    for i in range(15):
        b_type, ang, _ = planner.get_next_beam(belief.probs, angles)
        probe_angles.append(ang)

        w = synth.synthesize_pencil_beam(ang)
        meas = model.measure(true_angle, w, tx_power_dbm, distance_m, 'los')

        likelihoods = updater.compute_likelihood(meas["measured_power_watts"], w, angles, tx_watts, distance_m)
        belief.update(likelihoods)

        entropies.append(belief.get_entropy())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(entropies, marker='o')
    plt.title("Entropy Reduction vs Probes")
    plt.xlabel("Probe Step")
    plt.ylabel("Entropy (bits)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(probe_angles, marker='x', linestyle='--')
    plt.axhline(true_angle, color='red', linestyle=':', label="True Angle")
    plt.title("Adaptive Probe Trajectory")
    plt.xlabel("Probe Step")
    plt.ylabel("Probe Angle (degrees)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reports/adaptive_planner_trajectory.png")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    simulate_adaptive_probing()
    print("Plots saved to reports/")
