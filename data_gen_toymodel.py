import numpy as np
import matplotlib.pyplot as plt

import inv_transf_rng

rng = np.random.default_rng(seed=1234)

"""
The toy model is captured at the level of 2D distribution of E_nu, the neutrion energy and Q^2, momentum 
"""

c = 1.0

E_mu_peak = 0.6  # [GeV]
E_mu_width = 0.1  # [GeV]

m_p = 0.938 # [GeV]

def toy_model(mean_E_nu, sigma_E_nu, kappa, N_events ):
    """
    The present model is Gaussian(mu = mean_E_nu , sigma = sigma_E_nu)*exp(-E_nu/ )
    Args:
        mean_E_nu: mena neutrino energy
        sigma_E_nu: width of the neutrino energy
        kappa: Fudge factor that controls how quickly the Q distribution falls off

    Returns: 2D function that characterise the model

    """

    E_nu = rng.normal(loc=mean_E_nu, scale=sigma_E_nu)

    Q_dropoff = kappa * (E_nu / c)
    Q = rng.exponential(scale=Q_dropoff, size=N_events)

    return np.vstack((np.ones_like(Q) * E_nu, Q))



def Detection_efficiency(mom, threshold = 1.5e-4):
    """
    Args:
        threshold: detection threshold

    Returns:

    """
    def response_func(x, Amp,  mid, width):
        return Amp/2 * (np.tanh((x - mid)/width)+1)

    mid_detecor = 1.5e-4
    width_detecor = 1e-6

    def Stochaistic_acceptance(trh=threshold ):

        p_detect = response_func(mom, Amp=1,
                                 mid=mid_detecor,
                                 width=width_detecor)

        click_array = rng.choice(a=[0.0, 1.0], p=[1-p_detect, p_detect])



























