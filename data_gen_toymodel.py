import numpy as np
import matplotlib.pyplot as plt

import inv_transf_rng

rng = np.random.default_rng(seed=1234)
np.set_printoptions(precision=3, linewidth=150)

"""
The toy model is captured at the level of 2D distribution of E_nu, the neutrion energy and Q^2, momentum 
"""


def safe_sqrt(x):
    with np.errstate(invalid='ignore'):  # Ignore invalid value warnings
        return np.sqrt(np.maximum(x, 0))

def safe_divide(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
    return result

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
    E_p = (-Q ** 2 + c ** 2 * (M_n ** 2 + M_p ** 2)) / (2*M_n) # proton energy form kinematic constraints
    E_p = Q ** 2 / (2 * M_n) + M_n * c ** 2
    E_l = E_nu + M_n * c ** 2 - E_p   # charged lepton energy form energy conservation

    # obtaining the 3 momenta magnitudes from the energies
    p_p = get_3Mom_magn(M_p, E_p)
    p_l = get_3Mom_magn(M_l, E_l)

    # Angular reconstruction
    # right now the code is sufficiently fast for its intended purpose reserving the angular reconstruction calculation
    # is possible but not worth the effort at this point
    def lepton_angle(E_l, E_nu, p_l):
        numerator = M_p ** 2 - M_n ** 2 - M_l ** 2 + 2 * E_l * M_n - 2 * M_n * E_nu + 2 * E_l * E_nu
        denominator = 2 * p_l * E_nu
        cos_theta = safe_divide(numerator, denominator)
        return cos_theta

    def proton_angle(E_p, E_nu, p_p):
        numerator = M_l ** 2 - M_n ** 2 - M_p ** 2 + 2 * E_p * M_n - 2 * M_n * E_nu + 2 * E_p * E_nu
        denominator = 2 * p_p * E_nu
        cos_theta = safe_divide(numerator, denominator)
        return cos_theta

    cos_theta_p = proton_angle(E_p, E_nu, p_p)
    cos_theta_l = lepton_angle(E_l, E_nu, p_p)

    # constructing the data array shape = (6, 10)

    data_arr = np.column_stack((E_nu, Q, E_p, p_p, E_l, p_l, cos_theta_p, cos_theta_l))
    return data_arr


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

    return detected_data


if __name__ == '__main__':

    """Physical parameters"""
    c = 1.0
    E_mu_peak = 0.6  # [GeV]
    E_mu_width = 0.1  # [GeV]
    kappa = 3.0
    N_events = 5000

    M_p = 0.938  # [GeV]
    M_n = 0.939  # [GeV]
    M_l = 0.106  # [GeV]  Mass of a muon

    """Plotting parameters"""
    all_data_c = "#21A124"
    full_reconst_c = "#D0231F"
    proton_reconst_c = "#FF7E00"
    lepton_reconst_c = "#1975B6"

    axis_font_size = 13

    data = ToyModel(E_mu_peak, E_mu_width, kappa, N_events)

    detected_data = DetectorAcceptance(data)

    fully_reconstructed_condition = np.logical_not(np.isnan(detected_data).any(axis=1))
    fully_reconstructed_data = detected_data[fully_reconstructed_condition]

    ########################################################################

    fig1 = plt.figure(figsize=(12,8))
    ax11 = fig1.add_subplot(121)
    ax12 = fig1.add_subplot(122)

    ax11.hist(data[:, 0], color=all_data_c)
    ax11.hist(fully_reconstructed_data[:, 0], color=full_reconst_c)
    ax11.grid()
    ax11.set_xlabel(r"$E_{\nu}$", fontsize=axis_font_size)
    ax11.set_ylabel("count", fontsize=axis_font_size)

    ax12.hist(data[:, 1] ** 2, color=all_data_c)
    ax12.hist(fully_reconstructed_data[:, 1] ** 2, color=full_reconst_c)
    ax12.grid()
    ax12.set_xlabel(r"$Q^2$", fontsize=axis_font_size)
    ax12.set_ylabel("count", fontsize=axis_font_size)

    ########################################################################

    fig2 = plt.figure(figsize=(12, 8))
    ax21 = fig2.add_subplot(121)
    ax22 = fig2.add_subplot(122)

    ax21.hist(data[:, 2], color=all_data_c)
    ax21.hist(fully_reconstructed_data[:, 2], color=full_reconst_c)
    ax21.grid()
    ax21.set_xlabel(r"$E_{p}$", fontsize=axis_font_size)
    ax21.set_ylabel("count", fontsize=axis_font_size)

    ax22.hist(data[:, 3], color=all_data_c)
    ax22.hist(fully_reconstructed_data[:, 3], color=full_reconst_c)
    ax22.grid()
    ax22.set_xlabel(r"$|\vec{p}_p|$", fontsize=axis_font_size)
    ax22.set_ylabel("count", fontsize=axis_font_size)

    ########################################################################

    fig3 = plt.figure(figsize=(10, 4))
    ax31 = fig3.add_subplot(121)
    ax32 = fig3.add_subplot(122)

    ax31.hist(data[:, 4], color=all_data_c)
    ax31.hist(fully_reconstructed_data[:, 4], color=full_reconst_c)
    ax31.grid()
    ax31.set_xlabel(r"$E_{l}$", fontsize=axis_font_size)
    ax31.set_ylabel("count", fontsize=axis_font_size)

    ax32.hist(data[:, 5], color=all_data_c)
    ax32.hist(fully_reconstructed_data[:, 5], color=full_reconst_c)
    ax32.grid()
    ax32.set_xlabel(r"$|\vec{p}_l|$", fontsize=axis_font_size)
    ax32.set_ylabel("count", fontsize=axis_font_size)

    ########################################################################

    plt.show()





