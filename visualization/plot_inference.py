import numpy as np
import matplotlib.pyplot as plt
import os
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from inference.belief import GridBelief, BayesianUpdater
from sensing.measurements import MeasurementModel

def plot_belief_evolution():
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

    true_angle = 20.0

    plt.figure(figsize=(10, 6))
    plt.plot(angles, belief.probs, label="Prior", linestyle='--')

    # Probes: sweep 3 different angles
    probes = [0.0, 15.0, 20.0]

    for i, probe_ang in enumerate(probes):
        w = synth.synthesize_pencil_beam(probe_ang)
        meas = model.measure(true_angle, w, tx_power, path_loss, 'los')

        likelihoods = updater.compute_likelihood(meas["measured_power"], w, angles, tx_lin, pl_lin)
        belief.update(likelihoods)

        plt.plot(angles, belief.probs, label=f"Post Probe {i+1} (Steer {probe_ang}°)")

    plt.axvline(true_angle, color='red', linestyle=':', label='True Angle')

    plt.title("Bayesian Belief Evolution")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/belief_evolution.png")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    plot_belief_evolution()
    print("Plots saved to reports/")
