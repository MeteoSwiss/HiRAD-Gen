import numpy as np

def percentiles_from_histogram(hist_counts, bin_edges, percentiles_dict):
    """
    Estimate percentiles from a pre-computed histogram using linear interpolation
    on the cumulative distribution.
    
    Parameters
    ----------
    hist_counts : np.ndarray
        Raw (unnormalized) histogram counts per bin.
    bin_edges : np.ndarray
        Bin edges (length = len(hist_counts) + 1).
    percentiles_dict : dict
        Mapping of label -> fractional percentile, e.g. {99: 0.99, 99.9: 0.999}.
    
    Returns
    -------
    dict
        Mapping of label -> estimated percentile value.
    """
    cumulative = np.cumsum(hist_counts)
    total = cumulative[-1]
    if total == 0:
        return {key: np.nan for key in percentiles_dict}
    
    cdf = cumulative / total  # CDF at upper bin edges
    
    results = {}
    for key, p in percentiles_dict.items():
        # Find the bin where CDF crosses p
        idx = np.searchsorted(cdf, p)
        if idx >= len(cdf):
            # Beyond last bin — return upper edge
            results[key] = bin_edges[-1]
        elif idx == 0:
            # Within first bin — linearly interpolate from 0
            frac = p / cdf[0] if cdf[0] > 0 else 0.0
            results[key] = bin_edges[0] + frac * (bin_edges[1] - bin_edges[0])
        else:
            # Linearly interpolate within the bin
            cdf_low = cdf[idx - 1]
            cdf_high = cdf[idx]
            frac = (p - cdf_low) / (cdf_high - cdf_low) if (cdf_high - cdf_low) > 0 else 0.0
            results[key] = bin_edges[idx] + frac * (bin_edges[idx + 1] - bin_edges[idx])
    
    return results