import numpy as np
import pytest
from arrays.models import UniformLinearArray
from sensing.measurements import MeasurementModel
from beams.synthesis import BeamSynthesizer

def test_measurement_los():
    ula = UniformLinearArray(num_elements=16)
    synth = BeamSynthesizer(ula)
    w = synth.synthesize_pencil_beam(0.0)

    model = MeasurementModel(ula, noise_power_dbm=-100)

    meas_main = model.measure(target_angle=0.0, beam_weights=w, path_loss_db=50, fading_type='los')
    meas_null = model.measure(target_angle=15.0, beam_weights=w, path_loss_db=50, fading_type='los')

    assert meas_main["snr_db"] > meas_null["snr_db"] + 10
    # Beam gain in power should be exactly N=16 at peak
    assert meas_main["beam_gain_linear"] > 15.0

def test_measurement_rayleigh():
    ula = UniformLinearArray(num_elements=4)
    w = ula.steering_vector(0.0)
    model = MeasurementModel(ula, noise_power_dbm=-100)

    powers = []
    for _ in range(1000):
        meas = model.measure(target_angle=0.0, beam_weights=w, tx_power_dbm=10, path_loss_db=60, fading_type='rayleigh')
        powers.append(meas["measured_power"])

    mean_power = np.mean(powers)
    tx_lin = 10**(10/10)
    pl_lin = 10**(60/10)
    # Expected power includes beam gain (N=4 at broadside)
    expected_mean = (tx_lin / pl_lin) * 4.0

    assert np.isclose(mean_power, expected_mean, rtol=0.20)

def test_measurement_snr_scaling():
    ula = UniformLinearArray(num_elements=8)
    w = ula.steering_vector(0.0)
    model = MeasurementModel(ula, noise_power_dbm=-100)

    meas_high = model.measure(target_angle=0.0, beam_weights=w, tx_power_dbm=20, path_loss_db=50)
    meas_low = model.measure(target_angle=0.0, beam_weights=w, tx_power_dbm=0, path_loss_db=50)

    diff = meas_high["snr_db"] - meas_low["snr_db"]
    assert np.isclose(diff, 20.0, atol=1.0)
