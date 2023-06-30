import numpy as np
from scipy.integrate import quad

rng = np.random.default_rng(seed=12345)

def inv_transf_rng(f, min_val, max_val, size):
    """Generates random numbers using the inverese transformation method according to any (not neccersaily normalised function of one variable)

    Args:
        f (ufunc): differential cross-section (fuction according to whom random numbers are distributed)
        min_val (float): minimum value of the agument of f
        max_val (float): maximum value of the agument of f
        size (int): number of random variables generated

    Returns:
        ndarray: random numbers distributed according to f
    """
    x = np.linspace(min_val, max_val, 1000)  # Dummy array of the independent varible q_i
    pdf = f(x) / quad(f, min_val, max_val)[0]   # Probability density function 

    cdf = np.cumsum(pdf)  # Cumulative density function
    cdf_norm = cdf / cdf[-1]  # Normalzing the cummulative density function to span a range of [0, 1]

    random_uniform = rng.uniform(0, 1, size)  # Generating random numbers from a uniform distribution

    inverse_cdf = np.interp(random_uniform, cdf_norm, x)  # Inverse of the cumulative density function 

    return inverse_cdf






  
