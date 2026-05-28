import os
import json
import pandas as pd
from experiments.runner import ExperimentRunner

def run_all_benchmarks():
    configs = [
        # A. Sanity Observability Baseline (Should be 100% perfectly locking)
        {"name": "Sanity_HighSNR_LOS", "num_elements": 64, "noise_dbm": -100, "distance_m": 10, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "los"},

        # B. Distance Sweeps (Phase Transition Study)
        {"name": "Sweep_1km", "num_elements": 32, "noise_dbm": -100, "distance_m": 1000, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "rician"},
        {"name": "Sweep_5km", "num_elements": 32, "noise_dbm": -100, "distance_m": 5000, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "rician"},
        {"name": "Sweep_10km", "num_elements": 32, "noise_dbm": -100, "distance_m": 10000, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "rician"},
        {"name": "Sweep_20km", "num_elements": 32, "noise_dbm": -100, "distance_m": 20000, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "rician"},
        {"name": "Sweep_50km", "num_elements": 32, "noise_dbm": -100, "distance_m": 50000, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "rician"},

        # C. Planner comparisons at medium difficulty
        {"name": "Planner_Exhaustive_5km", "num_elements": 32, "noise_dbm": -100, "distance_m": 5000, "tx_power_dbm": 10, "planner": "exhaustive", "fading_type": "rician"},
        {"name": "Planner_Greedy_5km", "num_elements": 32, "noise_dbm": -100, "distance_m": 5000, "tx_power_dbm": 10, "planner": "greedy", "fading_type": "rician"},
        {"name": "Planner_Random_5km", "num_elements": 32, "noise_dbm": -100, "distance_m": 5000, "tx_power_dbm": 10, "planner": "random", "fading_type": "rician"}
    ]

    all_results = []

    for cfg in configs:
        print(f"Running benchmark: {cfg['name']}")
        runner = ExperimentRunner(cfg)
        res = runner.run_single_discovery()

        row = {"Benchmark": cfg["name"]}
        row.update(res)
        all_results.append(row)

    df = pd.DataFrame(all_results)

    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/benchmark_results.csv", index=False)
    print("\nBenchmark Results (Observability Phase Transition):")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_all_benchmarks()
