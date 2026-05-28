# Cooperative Observability for Aerial Phased-Array Discovery

This is a rigorous research-grade simulation framework focused on cooperative observability, Bayesian updates, and adaptive directional sensing.

## Project Structure
- `arrays/`: ULA and UPA hardware modeling, beam pattern and quantization.
- `beams/`: Beam synthesis engine supporting pencil, sector, and null-forming.
- `sensing/`: Measurement models (LOS, Rayleigh, Rician fading) and non-coherent energy detection simulation.
- `inference/`: Grid-based belief representation and Bayesian update engines, including centralized/distributed multi-panel fusion.
- `planners/`: Adaptive beam selection strategies (Exhaustive, Greedy Entropy, Random).
- `visualization/`: Utility scripts to plot physical arrays, beams, beliefs, and trajectory evolutions.
- `experiments/` & `benchmarks/`: Automated metric gathering frameworks.

## Quickstart
Run benchmarks and verify results:
```bash
PYTHONPATH=. python benchmarks/run_benchmarks.py
```

Run tests:
```bash
python -m pytest tests/
```
