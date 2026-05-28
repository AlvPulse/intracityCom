import os
import json
import pandas as pd
from experiments.runner import ExperimentRunner

def run_all_benchmarks():
    configs = [
        {"name": "Stage_A_Exhaustive_1km", "num_elements": 16, "noise_dbm": -100, "distance_m": 1000, "tx_power_dbm": 10, "planner": "exhaustive", "fading_type": "los"},
        {"name": "Stage_A_Greedy_1km", "num_elements": 16, "noise_dbm": -100, "distance_m": 1000, "tx_power_dbm": 10, "planner": "greedy", "fading_type": "los"},
        {"name": "Stage_A_Hierarchical_1km", "num_elements": 16, "noise_dbm": -100, "distance_m": 1000, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "los"},

        {"name": "Stage_B_Hierarchical_5km_Rician", "num_elements": 32, "noise_dbm": -100, "distance_m": 5000, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "rician"},
        {"name": "Stage_B_Greedy_5km_Rician", "num_elements": 32, "noise_dbm": -100, "distance_m": 5000, "tx_power_dbm": 10, "planner": "greedy", "fading_type": "rician"},

        {"name": "Stage_C_Hierarchical_20km_LowSNR_Quantized", "num_elements": 64, "noise_dbm": -95, "distance_m": 20000, "tx_power_dbm": 10, "planner": "hierarchical", "fading_type": "rayleigh", "quantization_bits": 4},
        {"name": "Stage_C_Random_20km_LowSNR_Quantized", "num_elements": 64, "noise_dbm": -95, "distance_m": 20000, "tx_power_dbm": 10, "planner": "random", "fading_type": "rayleigh", "quantization_bits": 4}
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
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_all_benchmarks()
