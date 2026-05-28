import numpy as np
import pytest
from planners.adaptive import ExhaustiveSweepPlanner, GreedyEntropyPlanner
from arrays.models import UniformLinearArray
from inference.belief import BayesianUpdater

def test_exhaustive_planner():
    angles = [-30, 0, 30]
    planner = ExhaustiveSweepPlanner(angles)

    b_type, ang1, _ = planner.get_next_beam()
    b_type, ang2, _ = planner.get_next_beam()
    b_type, ang3, _ = planner.get_next_beam()
    b_type, ang4, _ = planner.get_next_beam()

    assert ang1 == -30
    assert ang2 == 0
    assert ang3 == 30
    assert ang4 == -30  # looped back

def test_greedy_entropy_planner():
    ula = UniformLinearArray(16)
    noise_lin = 10**(-100/10)
    updater = BayesianUpdater(ula, noise_lin)
    grid_angles = np.linspace(-90, 90, 181)

    candidates = [-30, 0, 30]
    planner = GreedyEntropyPlanner(candidates, updater, ula, 10, 1e6, grid_angles)

    # Belief strongly peaked at 30
    belief = np.zeros_like(grid_angles)
    # 30 degrees is index 120
    belief[120] = 1.0

    b_type, ang, _ = planner.get_next_beam(belief)

    # If the target is exactly at 30, probing at 30 gives max measurement, but 0 variance.
    # Actually, variance of expected_power is sum p * (E - m)^2.
    # If belief is a delta function, variance over belief is 0 for all probes!
    # Let's make belief a mixture so variance > 0.
    belief[120] = 0.5
    belief[60] = 0.5 # -30 degrees

    b_type, ang, _ = planner.get_next_beam(belief)
    assert ang in [-30, 30] # should pick one of the peaks to distinguish them
