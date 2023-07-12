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


def CreationThreshold(M_prod, m_beam, m_target):
    return (M_prod ** 2 - m_beam ** 2 - m_target ** 2) * c ** 2 / m_target


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
    # imposing the creation threshold
    threshold = CreationThreshold(M_prod=M_p + M_l, m_beam=M_nu, m_target=M_n)

    E_nu = E_nu[E_nu >= threshold]

    # the generated neutrino energy determines the momentum transfer distribution
    Q_dropoff = np.abs(kappa * (E_nu / c) ** 2)
    Q = rng.exponential(scale=Q_dropoff)

    # Using the kinematics constraints the proton and lepton energies and 3-momenta magnitudes are determined
    E_p = (Q ** 2 + c ** 2 * (M_n ** 2 + M_p ** 2)) / (2 * M_n)  # proton energy form kinematic constraints
    E_l = E_nu + M_n * c ** 2 - E_p  # charged lepton energy form energy conservation

    conservation_constraints = np.logical_and(np.logical_and(E_p > 0, E_p < E_nu + M_n * c ** 2),
                                              np.logical_and(E_l > 0, E_l < E_nu + M_n * c ** 2))
    E_nu = E_nu[conservation_constraints]
    Q = Q[conservation_constraints]
    E_p = E_p[conservation_constraints]
    E_l = E_l[conservation_constraints]

    # obtaining the 3 momenta magnitudes from the energies
    p_p = get_3Mom_magn(M_p, E_p)
    p_l = get_3Mom_magn(M_l, E_l)


    # Angular reconstruction
    # right now the code is sufficiently fast for its intended purpose reserving the angular reconstruction calculation
    # is possible but not worth the effort at this point
    # def lepton_angle(E_l, E_nu, p_l):
    #     numerator = M_p ** 2 - M_n ** 2 - M_l ** 2 + 2 * E_l * M_n - 2 * M_n * E_nu + 2 * E_l * E_nu
    #     denominator = 2 * p_l * E_nu
    #     cos_theta = safe_divide(numerator, denominator)
    #     return cos_theta
    #
    # def proton_angle(E_p, E_nu, p_p):
    #     numerator = M_l ** 2 - M_n ** 2 - M_p ** 2 + 2 * E_p * M_n - 2 * M_n * E_nu + 2 * E_p * E_nu
    #     denominator = 2 * p_p * E_nu
    #     cos_theta = numerator/denominator
    #     return cos_theta
    #
    # cos_theta_p = proton_angle(E_p, E_nu, p_p)
    # cos_theta_l = lepton_angle(E_l, E_nu, p_p)
    #
    # # constructing the data array shape = (6, 10)
    # data_arr = np.column_stack((E_nu, Q, E_p, p_p, E_l, p_l, cos_theta_p, cos_theta_l))

    data_arr = np.column_stack((E_nu, Q, E_p, p_p, E_l, p_l))
    return data_arr


def Efficiency_func(p, eff_max, p_th, nu, xi, delt, phi):
    """
    See modelling the detector section
    Args:
        eff_max:
        p_th: threshold momentum value
        nu: fudge factor
        xi: detection probability in the middle
        delt: 'edge' displacement
        phi: detection probability at the left edge

    Returns: the probability of detecting a daughter particle in a particular event
    """
    sigma = (1 / xi) ** nu - 1

    print(sigma)

    alpha = 1 / (delt * nu * p_th) * np.log(((1 / phi) ** nu - 1) / sigma)
    print(alpha)

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

    p_th_p = 300  # [MeV/c]
    eff_max_p = 1.0  # [probability]
    nu_p = 1.0
    xi_p = 0.5  # [probability]
    delt_p = 0.01 * p_th_p  # [MeV/c]
    phi_p = 0.2  # [probability]

    p_th_l = 100  # [MeV/c]
    eff_max_l = 1.0  # [probability]
    nu_l = 1.0
    xi_l = 0.7  # [probability]
    delt_l = 0.01 * p_th_l  # [MeV/c]
    phi_l = 0.3  # [probability]

    proton_det_prob = Efficiency_func(p_p, eff_max_p, p_th_p, nu_p, xi_p, delt_p, phi_p)
    proton_reject_status = [rng.choice(a=[True, False], p=[1 - p, p]) for p in proton_det_prob]
    # # TODO: replace list comprehension here with something faster
    lepton_det_prob = Efficiency_func(p_l, eff_max_l, p_th_l, nu_l, xi_l, delt_l, phi_l)
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
    E_mu_peak = 600  # [MeV]
    E_mu_width = 100  # [MeV]
    kappa = 1
    N_events = int(1e7)

    num_bins = 15

    M_p = 938  # [MeV]
    M_n = 939  # [MeV]
    M_l = 105.7  # [MeV]  Mass of a muon
    M_nu = 1.2e-7  # [MeV] effective mass of the neutrino (can set zero)

    """Plotting parameters"""
    all_data_c = "green"
    full_reconst_c = "red"
    proton_reconst_c = "blue"
    lepton_reconst_c = "orange"

    axis_font_size = 13

    """Identification of the number and kind of particle tracks"""
    data = ToyModel(E_mu_peak, E_mu_width, kappa, N_events)

    print(np.any(np.isnan(data)))
    detected_data = DetectorAcceptance(data)

    print(detected_data.shape)
    """Two tracks"""
    fully_reconst_cond = np.logical_not(np.isnan(detected_data).any(axis=1))
    fully_reconstructed_data = detected_data[fully_reconst_cond]


    """Only pronton track"""
    proton_reconst_cond = np.logical_and(np.logical_not(np.isnan(detected_data[:, 2])),
                                         np.isnan(detected_data[:, 4]))
    proton_reconstructed_data = data[proton_reconst_cond]

    """Only charged lepton track"""

    lepton_reconst_cond = np.logical_and((np.isnan(detected_data[:, 2])),
                                         np.logical_not(np.isnan(detected_data[:, 4])))
    lepton_reconstructed_data = data[lepton_reconst_cond]

    ########################################################################

    # Beam energy and Q
    fig1 = plt.figure(figsize=(12, 8))
    ax11 = fig1.add_subplot(121)
    ax12 = fig1.add_subplot(122)

    ax11.hist(data[:, 0], bins=num_bins, color=all_data_c, histtype="step",
              label="all generated data")
    ax11.hist(fully_reconstructed_data[:, 0], bins=num_bins, color=full_reconst_c, histtype="step",
              label="full reconstructed")
    ax11.hist(proton_reconstructed_data[:, 0], bins=num_bins, color=proton_reconst_c,  histtype="step",
              label="Just proton reconst.")
    ax11.hist(lepton_reconstructed_data[:, 0], bins=num_bins, color=lepton_reconst_c,  histtype="step",
              label="Just lepton reconst.")

    ax11.grid()
    ax11.set_xlabel(r"$E_{\nu}$", fontsize=axis_font_size)
    ax11.set_ylabel("count", fontsize=axis_font_size)
    ax11.legend()

    ax12.hist(data[:, 1]**2, bins=num_bins, color=all_data_c, histtype="step",
              label="all generated data")
    ax12.hist(fully_reconstructed_data[:, 1]**2, bins=num_bins, color=full_reconst_c, histtype="step",
              label="full reconstructed")
    ax12.hist(proton_reconstructed_data[:, 1]**2 ,bins=num_bins, color=proton_reconst_c, histtype="step",
              label="Just proton reconst.")
    ax12.hist(lepton_reconstructed_data[:, 1]**2, bins=num_bins, color=lepton_reconst_c, histtype="step",
              label="Just lepton reconst.")

    ax12.grid()
    ax12.set_xlabel(r"$Q^2$", fontsize=axis_font_size)
    ax12.set_ylabel("count", fontsize=axis_font_size)
    ax12.legend()

    ########################################################################

    # proton energy momentum
    fig2 = plt.figure(figsize=(12, 12))
    ax21 = fig2.add_subplot(221)
    ax22 = fig2.add_subplot(223)
    ax31 = fig2.add_subplot(222)
    ax32 = fig2.add_subplot(224)

    ax21.hist(data[:, 2], bins=num_bins, color=all_data_c, histtype="step",
              label="all generated data")
    ax21.hist(fully_reconstructed_data[:, 2], bins=num_bins, color=full_reconst_c, histtype="step",
              label="full reconstructed")
    ax21.hist(proton_reconstructed_data[:, 2], bins=num_bins,  color=proton_reconst_c, histtype="step",
              label="Just proton reconst.")

    ax21.set_title("Proton")
    ax21.grid()
    ax21.set_xlabel(r"$E_{p}}$", fontsize=axis_font_size)
    ax21.set_ylabel("count", fontsize=axis_font_size)
    ax21.legend()

    ax22.hist(data[:, 3], color=all_data_c, histtype="step",
              label="all generated data")
    ax22.hist(fully_reconstructed_data[:, 3],  color=full_reconst_c, histtype="step",
              label="full reconstructed")
    ax22.hist(proton_reconstructed_data[:, 3], color=proton_reconst_c, histtype="step",
              label="Just proton reconst.")

    ax22.grid()
    ax22.set_xlabel(r"$|\vec{p}_p|$", fontsize=axis_font_size)
    ax22.set_ylabel("count", fontsize=axis_font_size)
    ax22.legend()

    ########################################################################

    # charged lepton energy momentum

    ax31.set_title("Charged Lepton")
    ax31.hist(data[:, 4], bins=num_bins, color=all_data_c, histtype="step",
              label="all generated data")
    ax31.hist(fully_reconstructed_data[:, 4], bins=num_bins, color=full_reconst_c, histtype="step",
              label="full reconstructed")
    ax31.hist(lepton_reconstructed_data[:, 4], bins=num_bins, color=lepton_reconst_c, histtype="step",
              label="Just proton reconst.")

    ax31.grid()
    ax31.set_xlabel(r"$E_l}$", fontsize=axis_font_size)
    ax31.legend()

    ax32.hist(data[:, 5], bins=num_bins, color=all_data_c, histtype="step",
              label="all generated data")
    ax32.hist(fully_reconstructed_data[:, 5], bins=num_bins, color=full_reconst_c, histtype="step",
              label="full reconstructed")
    ax32.hist(lepton_reconstructed_data[:, 5], bins=num_bins, color=lepton_reconst_c, histtype="step",
              label="Just proton reconst.")

    ax32.grid()
    ax32.set_xlabel(r"$|\vec{p}_l|$", fontsize=axis_font_size)
    ax32.legend()


    # empicical efficiency
    fig4 = plt.figure(figsize=(12,9))
    ax41 = fig4.add_subplot(121)
    ax42 = fig4.add_subplot(122)
    p_p = data[:, 2][np.logical_not(np.isnan(data[:, 2]))]
    p_l = data[:, 5][np.logical_not(np.isnan(data[:, 5]))]

    x_p = np.linspace(0, 1.5 * np.max(p_p), 1000)
    x_l = np.linspace(0, 1.5 * np.max(p_l), 1000)

    p_th_p = 300  # [MeV/c]
    eff_max_p = 1.0  # [probability]
    nu_p = 1.0
    xi_p = 0.5  # [probability]
    delt_p = 0.1 * p_th_p  # [MeV/c]
    phi_p = 0.2  # [probability]

    p_th_l = 100  # [MeV/c]
    eff_max_l = 1.0  # [probability]
    nu_l = 1.0
    xi_l = 0.7  # [probability]
    delt_l = 0.1 * p_th_l  # [MeV/c]
    phi_l = 0.3  # [probability]

    ax41.plot(x_p, Efficiency_func(x_p, eff_max_p, p_th_p,nu_p, xi_p, delt_p, phi_p))
    ax42.plot(x_l, Efficiency_func(x_l, eff_max_l, p_th_l, nu_l, xi_l, delt_p, phi_l))








    plt.tight_layout()
    plt.show()
