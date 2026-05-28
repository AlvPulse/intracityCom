import numpy as np
import pytest
from inference.fusion import CooperativeFusionEngine

def test_centralized_fusion():
    angles = np.linspace(-90, 90, 181)
    fusion = CooperativeFusionEngine(num_panels=2, grid_angles=angles)

    # Panel 1 sees peak at 30
    lh1 = np.ones_like(angles) * 0.1
    lh1[120] = 1.0 # 30 deg

    # Panel 2 sees peaks at 30 and -30
    lh2 = np.ones_like(angles) * 0.1
    lh2[120] = 1.0 # 30 deg
    lh2[60] = 1.0  # -30 deg

    joint = fusion.centralized_fusion([lh1, lh2])

    # The joint likelihood should strongly favor 30 degrees (intersection)
    assert joint[120] > joint[60]
    assert np.argmax(joint) == 120

def test_distributed_consensus():
    angles = np.linspace(-90, 90, 181)
    fusion = CooperativeFusionEngine(num_panels=3, grid_angles=angles)

    b1 = np.zeros_like(angles)
    b1[90] = 1.0

    b2 = np.zeros_like(angles)
    b2[120] = 1.0

    b3 = np.zeros_like(angles)
    b3[60] = 1.0

    new_beliefs = fusion.distributed_consensus([b1, b2, b3], consensus_steps=1)

    # After 1 step of all-to-all, they should all be equal to the average
    assert np.allclose(new_beliefs[0], new_beliefs[1])
    assert np.allclose(new_beliefs[1], new_beliefs[2])
    assert np.isclose(new_beliefs[0][90], 1/3)
    assert np.isclose(new_beliefs[0][120], 1/3)
    assert np.isclose(new_beliefs[0][60], 1/3)
