import numpy as np


def energy_weighted_rmse(spec1, spec2, k_min=192, eps=1e-12):
    """
    Energy-normalized RMSE between two 1D spectra for wavenumbers >= k_min.

    Each squared difference at wavenumber k is divided by a representative
    spectral energy at k (here: the average of the two spectra), giving
    a relative / normalized error.

    spec1, spec2: 1D arrays of spectral energy as a function of wavenumber.
                  Index i is assumed to correspond to wavenumber i.
    k_min:        Minimum wavenumber to include (inclusive).
    eps:          Small constant to avoid division by zero.
    """
    spec1 = np.asarray(spec1)
    spec2 = np.asarray(spec2)
    assert spec1.shape == spec2.shape, "Spectra must have same shape"

    # Select k >= k_min
    k = np.arange(spec1.size)
    mask = k >= k_min

    s1 = spec1[mask]
    s2 = spec2[mask]

    # Representative energy for normalization
    E = 0.5 * (s1 + s2)

    mean_squared_diff = (s1 - s2) ** 2

    # Each item divided by its (representative) spectrum value
    norm_sq = mean_squared_diff / (E + eps)

    return np.sqrt(np.mean(norm_sq))
