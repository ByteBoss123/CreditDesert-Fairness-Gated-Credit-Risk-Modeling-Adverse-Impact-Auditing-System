"""PSI with overflow coverage and explicit invalid-input handling."""
import numpy as np

def psi(expected, actual, bins=10):
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    expected, actual = expected[np.isfinite(expected)], actual[np.isfinite(actual)]
    if not len(expected) or not len(actual):
        raise ValueError('PSI requires nonempty finite reference and current samples')
    internal = np.unique(np.quantile(expected, np.linspace(0,1,bins+1)[1:-1]))
    if expected.min() == expected.max():
        value = expected[0]
        internal = np.array([value, np.nextafter(value, np.inf)])
    edges = np.r_[-np.inf, internal, np.inf]
    e = np.histogram(expected, edges)[0].astype(float) + 1e-6
    a = np.histogram(actual, edges)[0].astype(float) + 1e-6
    e /= e.sum(); a /= a.sum()
    return float(np.sum((a-e)*np.log(a/e)))
