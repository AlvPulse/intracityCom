import os
import json
import pandas as pd
from experiments.runner import ExperimentRunner

def run_all_benchmarks():
    configs = [
        {"name": "Stage_A_Greedy", "num_elements": 16, "noise_dbm": -100, "planner": "greedy"},
        {"name": "Stage_A_Exhaustive", "num_elements": 16, "noise_dbm": -100, "planner": "exhaustive"},
        {"name": "Stage_B_Greedy_LowSNR", "num_elements": 32, "noise_dbm": -90, "path_loss_db": 85, "planner": "greedy"}
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
