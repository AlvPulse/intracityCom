import numpy as np
import matplotlib.pyplot as plt
import os
from inference.fusion import CooperativeFusionEngine

def plot_fusion():
    angles = np.linspace(-90, 90, 181)
    fusion = CooperativeFusionEngine(num_panels=2, grid_angles=angles)

    # Simulate Ambiguous observation 1
    lh1 = np.exp(-0.5 * ((angles - 30)**2) / 100) + np.exp(-0.5 * ((angles + 50)**2) / 100)
    lh1 /= np.max(lh1)

    # Simulate Ambiguous observation 2
    lh2 = np.exp(-0.5 * ((angles - 30)**2) / 100) + np.exp(-0.5 * ((angles - 10)**2) / 100)
    lh2 /= np.max(lh2)

    joint = fusion.centralized_fusion([lh1, lh2])

    plt.figure(figsize=(10, 6))
    plt.plot(angles, lh1, label="Panel 1 Likelihood (Ambiguous)", linestyle='--')
    plt.plot(angles, lh2, label="Panel 2 Likelihood (Ambiguous)", linestyle='-.')
    plt.plot(angles, joint, label="Centralized Joint Likelihood", linewidth=2)

    plt.title("Cooperative Fusion Resolving Ambiguity")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Normalized Likelihood")
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/cooperative_fusion.png")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    plot_fusion()
    print("Plots saved to reports/")
