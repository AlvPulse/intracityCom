import numpy as np
from arrays.models import UniformLinearArray, UniformPlanarArray

class BeamSynthesizer:
    def __init__(self, array):
        self.array = array

    def generate_dft_codebook(self, oversampling: int = 1) -> np.ndarray:
        if not isinstance(self.array, UniformLinearArray):
            raise NotImplementedError("DFT codebook currently only supported for ULA.")
        N = self.array.num_elements
        num_beams = N * oversampling
        codebook = np.zeros((N, num_beams), dtype=complex)
        for m in range(num_beams):
            phases = 2 * np.pi * m * np.arange(N) / num_beams
            codebook[:, m] = np.exp(-1j * phases) / np.sqrt(N)
        return codebook

    def synthesize_pencil_beam(self, target_angle: float) -> np.ndarray:
        if isinstance(self.array, UniformLinearArray):
            return self.array.steering_vector(target_angle)
        else:
            raise NotImplementedError("For UPA use synthesize_pencil_beam_2d")

    def synthesize_sector_beam_ls(self, start_angle: float, end_angle: float, num_points: int = 181) -> np.ndarray:
        if not isinstance(self.array, UniformLinearArray):
            raise NotImplementedError("Sector beam currently only supported for ULA.")
        angles = np.linspace(-90, 90, num_points)
        target_pattern = np.zeros_like(angles, dtype=float)
        mask = (angles >= start_angle) & (angles <= end_angle)
        target_pattern[mask] = 1.0

        k = 2 * np.pi / self.array.wavelength
        angles_rad = np.radians(angles)
        A = np.exp(1j * k * np.outer(self.array.positions, np.sin(angles_rad)))
        A_T = A.T
        w = np.linalg.pinv(A_T) @ target_pattern
        w = w / np.linalg.norm(w)
        return w

    def synthesize_null_forming_beam(self, target_angle: float, null_angles: list) -> np.ndarray:
        if not isinstance(self.array, UniformLinearArray):
            raise NotImplementedError("Null forming currently only supported for ULA.")
        w_t = self.array.steering_vector(target_angle)
        if not null_angles:
            return w_t
        C = np.zeros((self.array.num_elements, len(null_angles)), dtype=complex)
        for i, angle in enumerate(null_angles):
            C[:, i] = self.array.steering_vector(angle)
        C_H_C = C.conj().T @ C
        C_H_C_inv = np.linalg.pinv(C_H_C)
        P = np.eye(self.array.num_elements) - C @ C_H_C_inv @ C.conj().T
        w_null = P @ w_t
        w_null = w_null / np.linalg.norm(w_null)
        return w_null
