import numpy as np

def friis_path_loss_linear(distance_m: float, wavelength: float, gt: float = 1.0, gr: float = 1.0) -> float:
    """
    Calculate Friis path loss in linear scale.
    P_r / P_t = G_t * G_r * (lambda / (4 * pi * R))^2
    Path loss is defined as P_t / P_r.
    """
    if distance_m <= 0:
        return 1.0
    loss_factor = (wavelength / (4 * np.pi * distance_m))**2
    return 1.0 / (gt * gr * loss_factor)

def friis_path_loss_db(distance_m: float, wavelength: float, gt: float = 1.0, gr: float = 1.0) -> float:
    """Calculate Friis path loss in decibels."""
    pl_linear = friis_path_loss_linear(distance_m, wavelength, gt, gr)
    return 10 * np.log10(pl_linear)
