import numpy as np
import pytest
from planners.adaptive import ExhaustiveSweepPlanner, GreedyEntropyPlanner, HierarchicalNarrowingPlanner
from arrays.models import UniformLinearArray
from inference.belief import BayesianUpdater

def test_exhaustive_planner():
    angles = [-30, 0, 30]
    planner = ExhaustiveSweepPlanner(angles)
    grid_angles = np.linspace(-90, 90, 181)
    belief = np.ones_like(grid_angles) / len(grid_angles)

    b_type, ang1, _ = planner.get_next_beam(belief, grid_angles)
    b_type, ang2, _ = planner.get_next_beam(belief, grid_angles)
    b_type, ang3, _ = planner.get_next_beam(belief, grid_angles)
    b_type, ang4, _ = planner.get_next_beam(belief, grid_angles)

    assert ang1 == -30
    assert ang2 == 0
    assert ang3 == 30
    assert ang4 == -30

def test_greedy_entropy_planner():
    ula = UniformLinearArray(16)
    noise_lin = 10**(-100/10)
    updater = BayesianUpdater(ula, noise_lin)
    grid_angles = np.linspace(-90, 90, 181)

    candidates = [-30, 0, 30]
    planner = GreedyEntropyPlanner(candidates, updater, ula, 10, 1e6, grid_angles)

    belief = np.zeros_like(grid_angles)
    belief[120] = 0.5
    belief[60] = 0.5

    b_type, ang, _ = planner.get_next_beam(belief, grid_angles)
    assert ang in [-30, 30]

def test_hierarchical_planner():
    grid_angles = np.linspace(-90, 90, 181)
    planner = HierarchicalNarrowingPlanner()

    # Flat belief
    belief = np.ones_like(grid_angles) / len(grid_angles)
    b_type, center, width = planner.get_next_beam(belief, grid_angles)

    # Should be sector beam with significant width
    assert b_type == "sector"
    assert width > 10.0

    # Peaked belief
    belief = np.zeros_like(grid_angles)
    belief[90] = 1.0 # 0 degrees
    b_type, center, width = planner.get_next_beam(belief, grid_angles)

    # Should be pencil beam focused exactly at mode
    assert b_type == "pencil"
    assert center == 0.0
