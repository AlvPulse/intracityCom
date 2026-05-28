import numpy as np
import pytest
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from inference.belief import GridBelief, BayesianUpdater
from sensing.measurements import MeasurementModel

def test_grid_belief_entropy():
    angles = np.linspace(-90, 90, 181)
    belief = GridBelief(angles)

    # Uniform prior entropy should be log2(181)
    assert np.isclose(belief.get_entropy(), np.log2(181))

    # Delta function entropy should be 0
    belief.probs = np.zeros_like(angles)
    belief.probs[90] = 1.0
    assert np.isclose(belief.get_entropy(), 0.0)

def test_bayesian_update_convergence():
    ula = UniformLinearArray(num_elements=16)
    synth = BeamSynthesizer(ula)
    w = synth.synthesize_pencil_beam(0.0)

    angles = np.linspace(-90, 90, 181)
    belief = GridBelief(angles)

    noise_dbm = -100
    noise_lin = 10**(noise_dbm/10)

    updater = BayesianUpdater(ula, noise_lin)
    model = MeasurementModel(ula, noise_power_dbm=noise_dbm)

    tx_power = 10
    path_loss = 60
    tx_lin = 10**(tx_power/10)
    pl_lin = 10**(path_loss/10)

    # True target is at 0 degrees
    true_angle = 0.0
    initial_entropy = belief.get_entropy()

    # Apply multiple updates
    for _ in range(5):
        meas = model.measure(true_angle, w, tx_power, path_loss, 'los')
        likelihoods = updater.compute_likelihood(meas["measured_power"], w, angles, tx_lin, pl_lin)
        belief.update(likelihoods)

    final_entropy = belief.get_entropy()

    # Entropy should decrease significantly
    assert final_entropy < initial_entropy

    # Peak probability should be at 0 degrees (index 90)
    assert np.argmax(belief.probs) == 90
