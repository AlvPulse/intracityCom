import numpy as np
import pytest
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer
from inference.belief import GridBelief, BayesianUpdater
from sensing.measurements import MeasurementModel
from utils.propagation import dbm_to_linear

def test_grid_belief_entropy():
    angles = np.linspace(-90, 90, 181)
    belief = GridBelief(angles)
    assert np.isclose(belief.get_entropy(), np.log2(181))
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
    noise_watts = dbm_to_linear(noise_dbm)

    updater = BayesianUpdater(ula, noise_watts)
    model = MeasurementModel(ula, noise_power_dbm=noise_dbm)

    tx_power_dbm = 10
    tx_watts = dbm_to_linear(tx_power_dbm)
    distance_m = 10.0 # very close to guarantee lock

    true_angle = 0.0
    initial_entropy = belief.get_entropy()

    for _ in range(5):
        meas = model.measure(true_angle, w, tx_power_dbm, distance_m, 'los')
        likelihoods = updater.compute_likelihood(meas["measured_power_watts"], w, angles, tx_watts, distance_m)
        belief.update(likelihoods)

    final_entropy = belief.get_entropy()
    assert final_entropy < initial_entropy
    assert np.argmax(belief.probs) == 90
