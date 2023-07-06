import numpy as np
import matplotlib.pyplot as plt

import inv_transf_rng

rng = np.random.default_rng(seed=1234)
np.set_printoptions(precision=3, linewidth=150)

"""
The toy model is captured at the level of 2D distribution of E_nu, the neutrion energy and Q^2, momentum 
"""

c = 1.0

E_mu_peak = 0.6  # [GeV]
E_mu_width = 0.1  # [GeV]

M_p = 0.938  # [GeV]
M_n = 0.939  # [GeV]

M_l = 0.106  # [GeV]  Mass of a muon


# def Detection_efficiency(mom, threshold = 1.5e-4):
#     """
#     Args:
#         threshold: detection threshold
#
#     Returns:
#
#     """
#     def response_func(x, Amp,  mid, width):
#         return Amp/2 * (np.tanh((x - mid)/width)+1)
#
#     mid_detecor = 1.5e-4
#     width_detecor = 1e-6
#
#     def Stochaistic_acceptance(trh=threshold ):
#
#         p_detect = response_func(mom, Amp=1,
#                                  mid=mid_detecor,
#                                  width=width_detecor)
#
#         click_array = rng.choice(a=[0.0, 1.0], p=[1-p_detect, p_detect])
#
#

def safe_sqrt(x):
    with np.errstate(invalid='ignore'):  # Ignore invalid value warnings
        return np.sqrt(np.maximum(x, 0))


def get_3Mom_magn(M, E):
    return safe_sqrt(E ** 2 - M ** 2 * c ** 4) / c


def ToyModel(mean_E_nu, std_E_nu, kappa, N_events):
    """

    Args: Model parameters
        mean_E_nu: Mean neutrino energy
        std_E_nu: standard deviation in the neutrino energy
        kappa: fudge factor to make the Q2 distribution have the desired drop-off
        N_events: Number of events generated

    Returns: Array of simulated data
    1st col : E_nu
    2nd col : Q
    3rd col : proton energy
    4th col : proton 3-momentum magnitude
    5th col : charged lepton energy
    6th col : charged lepton 3-momentum magnitude

    each row corresponds to a single event
    """

    # generate a random neutrino energy according to a normal distribution
    E_nu = rng.normal(loc=mean_E_nu, scale=std_E_nu, size=N_events)

    # the generated neutrino energy determines the momentum transfer distribution
    Q_dropoff = kappa * (E_nu / c)
    Q = rng.exponential(scale=Q_dropoff)

    # Using the kinematics constraints the proton and lepton energies and 3-momenta magnitudes are determined
    E_p = (-Q ** 2 + c ** 2 * (M_n ** 2 + M_p ** 2))  # proton energy form kinematic constraints
    E_l = (- Q ** 2 + c ** 2 * (M_n ** 2 + M_p ** 2)) / (2 * M_n)  # charged lepton energy form energy conservation

    # obtaining the 3 momenta magnitudes from the energies
    p_p = get_3Mom_magn(M_p, E_p)
    p_l = get_3Mom_magn(M_l, E_l)

    # constructing the data array shape = (6, 10)

    data_arr = np.column_stack((E_nu, Q, E_p, p_p, E_l, p_l))

    return data_arr


d_arr = ToyModel(E_mu_peak, E_mu_width, 3, 10)



def Efficiency_func(p, eff_max, p_th, nu, xi, delt, phi):
    """
    See modelling the detector section
    Args:
        eff_max:
        p_th: threshold momnetum value
        nu: fudge factor
        xi: detection probability in the middle
        delt: 'edge' displacement
        phi: detection probability at the left edge

    Returns: the probability of detecting a daughter particle in a particular event
    """
    sigma = (1 / xi) ** nu - 1

    alpha = 1 / (delt * nu * p_th) * np.log(((1 / phi) ** nu - 1) / sigma)

    eff = eff_max / (1 + sigma * np.exp(-alpha * nu * (p - p_th)))  # efficiency = P("click")

    return eff


def DetectorAcceptance(data_array):
    """
    Args:
        data_array: the data array generated using the Toy Model
        threshold: detection threshold (momentum value)

    Returns:
    """

    # the detection efficiency is a function of the daughter particle 3-momenta magnitudes
    # extracting the 3-momenta form the data array see encoding above

    p_p = data_array[:, 2]
    p_l = data_array[:, 5]





    p_th = 2.5  # [GeV] i. e. 250 MeV
    eff_max = 1.0
    nu = 1.0
    xi = 0.1
    delt = 0.1 * p_th
    phi = 0.2

    proton_det_prob = Efficiency_func(p_p, eff_max, p_th, nu, xi, delt, phi)
    proton_reject_status = [rng.choice(a=[True, False], p=[1-p, p]) for p in proton_det_prob]
    # # TODO: replace list comprehension here with something faster
    lepton_det_prob = Efficiency_func(p_l, eff_max, p_th, nu, xi, delt, phi)
    lepton_reject_status = [rng.choice(a=[True, False], p=[1 - p, p]) for p in lepton_det_prob]

    detected_data = data_array
    # selecting for the proton detections
    detected_data[:, 2] = np.where(proton_reject_status, np.nan, detected_data[:, 2])
    detected_data[:, 3] = np.where(proton_reject_status, np.nan, detected_data[:, 3])

    # selecting for the charged lepton detections
    detected_data[:, 4] = np.where(lepton_reject_status, np.nan, detected_data[:, 4])
    detected_data[:, 5] = np.where(lepton_reject_status, np.nan, detected_data[:, 5])






DetectorAcceptance(d_arr)
