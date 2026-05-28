import numpy as np
from arrays.models import UniformLinearArray

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
            # Standardized DFT spatial phase progression
            phases = 2 * np.pi * m * np.arange(N) / num_beams
            codebook[:, m] = np.exp(-1j * phases) / np.sqrt(N)
        return codebook

    def synthesize_pencil_beam(self, target_angle: float) -> np.ndarray:
        # Assuming your array model has a proper steering_vector method
        return self.array.steering_vector(target_angle)

    def synthesize_sector_beam_ls(self, start_angle: float, end_angle: float, num_points: int = 361) -> np.ndarray:
        if not isinstance(self.array, UniformLinearArray):
            raise NotImplementedError("Sector beam currently only supported for ULA.")

        angles = np.linspace(-90, 90, num_points)
        target_pattern = np.zeros_like(angles, dtype=complex) # Complex allows better optimization

        # Define a mask with a slightly smoother transition or just a flat top
        mask = (angles >= start_angle) & (angles <= end_angle)
        target_pattern[mask] = 1.0

        # Flatten positions explicitly to avoid shape issues with np.outer
        pos = np.asarray(self.array.positions).flatten()
        k = 2 * np.pi / self.array.wavelength
        angles_rad = np.radians(angles)

        # A matrix: Rows = Elements (N), Columns = Angles (num_points)
        A = np.exp(1j * k * np.outer(pos, np.sin(angles_rad)))

        # We want to solve: A^H * w = target_pattern -> w = pinv(A^H) * target_pattern
        A_H = A.conj().T
        w = np.linalg.pinv(A_H) @ target_pattern

        # Normalize to prevent exploding gain
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        return w

    def synthesize_null_forming_beam(self, target_angle: float, null_angles: list) -> np.ndarray:
        if not isinstance(self.array, UniformLinearArray):
            raise NotImplementedError("Null forming currently only supported for ULA.")

        w_t = self.array.steering_vector(target_angle)
        if not null_angles:
            return w_t

        N = self.array.num_elements
        if len(null_angles) >= N:
            raise ValueError(f"Too many nulls ({len(null_angles)}). Maximum allowed for {N} elements is {N-1}.")

        # Construct the Constraint Matrix C (N x Num_Nulls)
        C = np.zeros((N, len(null_angles)), dtype=complex)
        for i, angle in enumerate(null_angles):
            C[:, i] = self.array.steering_vector(angle)

        # Projection Matrix: P = I - C * inv(C^H * C) * C^H
        C_H = C.conj().T
        P = np.eye(N) - C @ np.linalg.pinv(C_H @ C) @ C_H

        w_null = P @ w_t

        # Avoid division by zero if target is completely orthogonalized out
        norm = np.linalg.norm(w_null)
        if norm < 1e-6:
            print("Warning: Target angle is severely attenuated by null constraints.")
            return w_t

        return w_null / norm
