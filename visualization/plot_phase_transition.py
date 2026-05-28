import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_phase_transition():
    if not os.path.exists("reports/benchmark_results.csv"):
        print("Run benchmarks first to generate reports/benchmark_results.csv")
        return

    df = pd.read_csv("reports/benchmark_results.csv")

    # Filter for the Sweep experiments
    df_sweeps = df[df['Benchmark'].str.startswith('Sweep_')].copy()

    if df_sweeps.empty:
        print("No Sweep benchmarks found.")
        return

    # Extract distance from name (e.g., 'Sweep_1km' -> 1)
    df_sweeps['Distance_km'] = df_sweeps['Benchmark'].apply(lambda x: int(x.split('_')[1].replace('km', '')))
    df_sweeps = df_sweeps.sort_values('Distance_km')

    plt.figure(figsize=(10, 6))

    # Plot 1: Final Entropy vs Distance
    plt.subplot(1, 2, 1)
    plt.plot(df_sweeps['Distance_km'], df_sweeps['final_entropy'], marker='o', color='blue')
    plt.axhline(7.5, color='red', linestyle='--', label='Uninformative (Max Entropy)')
    plt.title("Observability Collapse (Entropy vs Distance)")
    plt.xlabel("Distance (km)")
    plt.ylabel("Final Entropy (bits)")
    plt.grid(True)
    plt.legend()

    # Plot 2: Likelihood Sharpness vs Distance
    plt.subplot(1, 2, 2)
    plt.plot(df_sweeps['Distance_km'], df_sweeps['avg_likelihood_sharpness'], marker='s', color='green')
    plt.axhline(1.0, color='red', linestyle='--', label='Flat Likelihood')
    plt.title("Measurement Informativeness")
    plt.xlabel("Distance (km)")
    plt.ylabel("Avg Likelihood Sharpness")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("reports/observability_phase_transition.png")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    plot_phase_transition()
    print("Phase transition plot saved to reports/")
