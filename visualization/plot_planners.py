import numpy as np
import matplotlib.pyplot as plt
import os
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from inference.belief import GridBelief, BayesianUpdater
from sensing.measurements import MeasurementModel
from planners.adaptive import GreedyEntropyPlanner

def simulate_adaptive_probing():
    ula = UniformLinearArray(num_elements=16)
    synth = BeamSynthesizer(ula)

    angles = np.linspace(-90, 90, 181)
    belief = GridBelief(angles)

    noise_dbm = -100
    noise_lin = 10**(noise_dbm/10)

    updater = BayesianUpdater(ula, noise_lin)
    model = MeasurementModel(ula, noise_power_dbm=noise_dbm)

    tx_power = 10
    path_loss = 70
    tx_lin = 10**(tx_power/10)
    pl_lin = 10**(path_loss/10)

    true_angle = 45.0

    candidates = np.linspace(-90, 90, 31)
    planner = GreedyEntropyPlanner(candidates, updater, ula, tx_lin, pl_lin, angles)

    entropies = [belief.get_entropy()]
    probe_angles = []

    for i in range(15):
        b_type, ang, _ = planner.get_next_beam(belief.probs)
        probe_angles.append(ang)

        w = synth.synthesize_pencil_beam(ang)
        meas = model.measure(true_angle, w, tx_power, path_loss, 'los')

        likelihoods = updater.compute_likelihood(meas["measured_power"], w, angles, tx_lin, pl_lin)
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
