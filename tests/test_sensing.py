import numpy as np
import pytest
from arrays.models import UniformLinearArray
from sensing.measurements import MeasurementModel
from beams.synthesis import BeamSynthesizer
from utils.propagation import dbm_to_linear, friis_path_loss_linear

def test_measurement_los():
    ula = UniformLinearArray(num_elements=16)
    synth = BeamSynthesizer(ula)
    w = synth.synthesize_pencil_beam(0.0)

    model = MeasurementModel(ula, noise_power_dbm=-100)

    meas_main = model.measure(target_angle=0.0, beam_weights=w, distance_m=10.0, fading_type='los')
    meas_null = model.measure(target_angle=15.0, beam_weights=w, distance_m=10.0, fading_type='los')

    assert meas_main["snr_db"] > meas_null["snr_db"] + 10
    assert meas_main["beam_gain_linear"] > 15.0

def test_measurement_rayleigh():
    ula = UniformLinearArray(num_elements=4)
    w = ula.steering_vector(0.0)
    model = MeasurementModel(ula, noise_power_dbm=-100)

    powers = []
    distance_m = 100.0
    tx_power_dbm = 10
    for _ in range(1000):
        meas = model.measure(target_angle=0.0, beam_weights=w, tx_power_dbm=tx_power_dbm, distance_m=distance_m, fading_type='rayleigh')
        powers.append(meas["measured_power_watts"])

    mean_power = np.mean(powers)
    tx_watts = dbm_to_linear(tx_power_dbm)
    attenuation = friis_path_loss_linear(distance_m, ula.wavelength)

    expected_mean = tx_watts * attenuation * 4.0 # 4 is broadside gain

    assert np.isclose(mean_power, expected_mean, rtol=0.20)

def test_measurement_snr_scaling():
    ula = UniformLinearArray(num_elements=8)
    w = ula.steering_vector(0.0)
    model = MeasurementModel(ula, noise_power_dbm=-100)

    meas_high = model.measure(target_angle=0.0, beam_weights=w, tx_power_dbm=20, distance_m=50.0)
    meas_low = model.measure(target_angle=0.0, beam_weights=w, tx_power_dbm=0, distance_m=50.0)

    diff = meas_high["snr_db"] - meas_low["snr_db"]
    assert np.isclose(diff, 20.0, atol=1.0)
