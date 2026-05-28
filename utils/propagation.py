import numpy as np

def friis_path_loss_linear(distance_m: float, wavelength: float, gt: float = 1.0, gr: float = 1.0) -> float:
    """
    Calculate Friis path loss factor in linear scale.
    Returns P_r / P_t (which is <= 1.0).
    Note: Previously this might have returned P_t / P_r. We standardize to attenuation.
    """
    if distance_m <= 0:
        return 1.0

    # Friis equation: P_r / P_t = G_t * G_r * (lambda / (4 * pi * R))^2
    loss_factor = (wavelength / (4 * np.pi * distance_m))**2
    attenuation = gt * gr * loss_factor

    # Cap at 1.0 to prevent gain at extremely close distances
    return min(attenuation, 1.0)

def friis_path_loss_db(distance_m: float, wavelength: float, gt: float = 1.0, gr: float = 1.0) -> float:
    """Calculate Friis attenuation in decibels (will be negative)."""
    pl_linear = friis_path_loss_linear(distance_m, wavelength, gt, gr)
    return 10 * np.log10(pl_linear + 1e-12)

def dbm_to_linear(dbm: float) -> float:
    """Convert dBm to Watts."""
    # 0 dBm = 1 mW = 1e-3 W
    return (10 ** (dbm / 10.0)) * 1e-3

def linear_to_dbm(watts: float) -> float:
    """Convert Watts to dBm."""
    return 10 * np.log10(watts / 1e-3 + 1e-15)
