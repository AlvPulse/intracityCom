import numpy as np
import pytest
from arrays.models import UniformLinearArray
from beams.synthesis import BeamSynthesizer

def test_dft_codebook():
    ula = UniformLinearArray(num_elements=8)
    synth = BeamSynthesizer(ula)
    cb = synth.generate_dft_codebook()

    assert cb.shape == (8, 8)

    # Check orthogonality of standard DFT
    inner_prod = cb.conj().T @ cb
    assert np.allclose(inner_prod, np.eye(8), atol=1e-7)

def test_pencil_beam():
    ula = UniformLinearArray(num_elements=4)
    synth = BeamSynthesizer(ula)
    w = synth.synthesize_pencil_beam(0.0)

    expected = np.ones(4) / np.sqrt(4)
    assert np.allclose(w, expected)

def test_sector_beam_ls():
    ula = UniformLinearArray(num_elements=16)
    synth = BeamSynthesizer(ula)
    w = synth.synthesize_sector_beam_ls(start_angle=-20.0, end_angle=20.0)

    # Check normalization
    assert np.isclose(np.linalg.norm(w), 1.0)

    # Evaluate power inside sector vs outside
    angles = np.linspace(-90, 90, 181)
    power = ula.power_pattern(w, angles)

    # Simple check: max power should be somewhere inside or very near the sector
    max_idx = np.argmax(power)
    max_angle = angles[max_idx]
    assert -30 <= max_angle <= 30

def test_null_forming_beam():
    ula = UniformLinearArray(num_elements=8)
    synth = BeamSynthesizer(ula)

    target = 0.0
    nulls = [30.0, -45.0]

    w = synth.synthesize_null_forming_beam(target, nulls)

    # Check normalization
    assert np.isclose(np.linalg.norm(w), 1.0)

    # Check if nulls are formed
    af_target = np.abs(ula.array_factor(w, np.array([target])))
    af_nulls = np.abs(ula.array_factor(w, np.array(nulls)))

    # Target should be significant, nulls should be close to 0
    assert af_target > 0.5
    for af_n in af_nulls:
        assert af_n < 1e-5
