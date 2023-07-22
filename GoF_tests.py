"""
This .py file contains the sub-routines to perform the various goodness of fit tests

Design notes:
OPP principles
All tests should take in the same kind of measured and predicted data arrays
All tests should out their test statistics only, P-value calculation from the test statistics is a separate function
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import chi2
from scipy.spatial.distance import cdist

rng = np.random.default_rng(seed=124)
np.set_printoptions(precision=5, linewidth=150)


class BinData:
    def __init__(self, data, N_init=None, min_occ=None):

        self.data = data

        # input error handling

        # default value assignment
        default_N_init = 100

        if N_init is None:
            self.N_init = default_N_init
        else:
            self.N_init = N_init

        default_min_occ = 5.0
        if min_occ is None:
            self.min_occ = default_min_occ
        else:
            self.min_occ = min_occ

    def merge_bin(self):
        """
        Returns: The bin edges and bin counts for the data array
        """
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        bin_width = (data_max - data_min) / self.N_init

        # Initialize the bins
        bin_edges = np.arange(data_min, data_max + bin_width, bin_width)
        bin_counts, _ = np.histogram(self.data, bins=bin_edges)

        # Merge neighboring bins until each bin contains at least m entries
        while np.min(bin_counts) < self.min_occ:
            min_count_idx = np.argmin(bin_counts)

            # Merge the bin with its neighboring bin that has the fewest entries
            if min_count_idx == 0:  # left edge case
                merge_idx = 1
            elif min_count_idx == len(bin_counts) - 1:  # right edge case
                merge_idx = len(bin_counts) - 2
            else:
                left_count = bin_counts[min_count_idx - 1]
                right_count = bin_counts[min_count_idx + 1]
                merge_idx = min_count_idx - 1 if left_count < right_count else min_count_idx + 1

            bin_counts[min_count_idx] += bin_counts[merge_idx]
            bin_edges = np.delete(bin_edges, merge_idx)
            bin_counts = np.delete(bin_counts, merge_idx)

        return bin_counts, bin_edges


class GoF:
    def __init__(self, predicted, measured):

        self.predicted = predicted
        self.measured = measured

    def PearsonChi2(self):
        """
        This is a binned Goodness of fit test and as such calls on the functionality of the bin_data class


        Returns: Pearson's Chi-squared test statistics and the number of degrees of freedom
        """

        # Binning the data
        # Always bin data based on theory i.e. bin based on predicted.
        init_data_to_bin = BinData(self.predicted)  # call the data binning class
        pred_bin_counts, pred_bin_edges = init_data_to_bin.merge_bin()  # calls the specific binning algorith

        # bin measured data according to the same bin edges as the predicted data

        meas_bin_counts, meas_bin_edges = np.histogram(self.measured, bins=pred_bin_edges)  # bin the measured data

        # print(np.all(pred_bin_edges == meas_bin_edges))
        #
        # # Testing
        # plt.bar(x=pred_bin_edges[:-1], height=pred_bin_counts, width=np.diff(pred_bin_edges), align='edge',
        #         fc='salmon', ec='black')
        # plt.bar(x=meas_bin_edges[:-1], height=meas_bin_counts, width=np.diff(meas_bin_edges), align='edge',
        #         fc='skyblue', ec='black')
        #
        # plt.show()

        chi2_stat = np.sum((meas_bin_counts - pred_bin_counts) ** 2 / pred_bin_counts)
        # TODO: add axis argument to sum if data storge format has been decided
        DoF = len(meas_bin_counts) - 1
        return chi2_stat, DoF

    def Poisson_Likelihood_Ratio(self):
        """
        This is a binned Goodness of fit test and as such calls on the functionality of the bin_data class

        Returns: The Poisson likelihood ratio test statistic and the number of degrees of freedom
        """

        # Binning the data
        # Always bin data based on theory i.e. bin based on predicted.
        init_data_to_bin = BinData(self.predicted)  # call the data binning class
        pred_bin_counts, pred_bin_edges = init_data_to_bin.merge_bin()  # calls the specific binning algorith

        meas_bin_counts, meas_bin_edges = np.histogram(self.measured, bins=pred_bin_edges)  # bin the measured data

        Fp_stat = 2 * np.sum(
            pred_bin_counts - meas_bin_counts + pred_bin_counts * np.log(meas_bin_counts / pred_bin_counts
                                                                         ))
        DoF = len(meas_bin_counts)

        return Fp_stat, DoF

    def Point_to_Point_DissimExp(self):
        """
        This is a un-binned multidimensional GoF test
        EXPERIMENTAL
        Returns: Returns the point-to-point dissimilarity test statistic T
        """

        X_meas = self.measured
        X_pred = self.predicted

        n_meas = len(X_meas)
        n_pred = len(X_pred)

        def Psi(x):
            # Define the weight function
            sigma = 1
            return np.exp(-x ** 2 / (2 * sigma ** 2))

        # Calculate the distance matrices
        if np.ndim(X_meas) == 1:
            dist_meas = cdist(X_meas[:, np.newaxis], X_meas[:, np.newaxis])
            dist_pred = cdist(X_pred[:, np.newaxis], X_pred[:, np.newaxis])
            dist_mixed = cdist(X_meas[:, np.newaxis], X_pred[:, np.newaxis])
        else:
            dist_meas = cdist(X_meas, X_meas)
            dist_pred = cdist(X_pred, X_pred)
            dist_mixed = cdist(X_meas, X_pred)

        # Term 1: Calculate the sum of Psi(|x_i_meas - x_j_meas|)
        sum_meas = np.sum(np.triu(Psi(dist_meas), k=1))

        # Term 2: Calculate the sum of Psi(|x_i_pred - x_j_pred|)
        sum_pred = np.sum(np.triu(Psi(dist_pred), k=1))

        # Term 3: Calculate the sum of Psi(|x_i_meas - x_j_pred|)
        sum_mixed = np.sum(Psi(dist_mixed))

        # print("Vectorised")
        # print(sum_meas)
        # print(sum_pred)
        # print(sum_mixed)

        # Calculate T statistic
        T = 1 / (n_meas * (n_meas - 1)) * sum_meas \
            + 1 / (n_pred * (n_pred - 1)) * sum_pred \
            - 1 / (n_meas * n_pred) * sum_mixed

        return T

    def Point_to_Point_Dissim(self):
        """
        This is a un-binned multidimensional GoF test
        DUMB implementation (double for loop)
        Returns: Returns the point-to-point dissimilarity test statistic T
        """

        class P2PdWeightFunctions:
            def __init__(self, dist):
                self.dist = dist

            def NonAdaptive(self):
                pass

        X_meas = self.measured
        X_pred = self.predicted

        n_meas = len(X_meas)
        n_pred = len(X_pred)

        def Psi(x):
            # Define the weight function
            sigma = 1
            return np.exp(-x ** 2 / (2 * sigma ** 2))

        # Term 1: Calculate the sum of Psi(|x_i_meas - x_j_meas|)
        sum_meas = 0
        for i in tqdm(range(n_meas)):
            for j in range(i + 1, n_meas):
                distance = np.linalg.norm(X_meas[i] - X_meas[j])
                sum_meas += Psi(distance)

        # Term 2 : Calculate the sum of Psi(|x_i_pred - x_j_pred|)
        sum_pred = 0
        for i in tqdm(range(n_pred)):
            for j in range(i + 1, n_pred):
                distance = np.linalg.norm(X_pred[i] - X_pred[j])
                sum_pred += Psi(distance)

        # Term 3: Calculate the sum of Psi(|x_i_meas - x_j_pred|)
        sum_mixed = 0
        for i in tqdm(range(n_meas)):
            for j in range(n_pred):
                distance = np.linalg.norm(X_meas[i] - X_pred[j])
                sum_mixed += Psi(distance)

        # print("Loops")
        # print(sum_meas)
        # print(sum_pred)
        # print(sum_mixed)

        # Calculate T statistic
        T = 1 / (n_meas * (n_meas - 1)) * sum_meas \
            + 1 / (n_pred * (n_pred - 1)) * sum_pred \
            - 1 / (n_meas * n_pred) * sum_mixed

        return T


def Pval_Chi2Distribution(test_stat, N_DoF):
    p_value = 1 - chi2.cdf(test_stat, N_DoF)
    return p_value


def Permuation_Method(pred, meas, n_perms):
    n_meas = len(meas)

    # perform un-shuffled p2pd test
    T = GoF(pred, meas).Point_to_Point_DissimExp()

    T_perm_arr = []

    for i in range(n_perms):
        pooled_data = np.append(pred, meas)
        rng.shuffle(pooled_data)

        # selecting the temporary measurement set by randomly drawing n_meas elements from the pooled data
        meas_perm = rng.choice(pooled_data, n_meas,
                               replace=False)  # replace argument has to be false to avoid duplicates

        # the remaining events are designated predictions temporarily
        pred_perm = np.setdiff1d(pooled_data, meas_perm)

        T_perm = GoF(pred_perm, meas_perm).Point_to_Point_DissimExp()

        T_perm_arr.append(T_perm)

    T_perm_arr = np.array(T_perm_arr)

    p_val = np.sum(T < T_perm_arr) / len(T_perm_arr)

    return p_val


nA = 1000
nB = 100

muAx = 0.0
muAy = 0.0
sigAx = 0.2
sigAy = 0.2

A = np.hstack((
    rng.normal(loc=muAx, scale=sigAx, size=nA)[:, np.newaxis],
    rng.normal(loc=muAy, scale=sigAy, size=nA)[:, np.newaxis]
))

thetas = np.linspace(0, 2 * np.pi, 1000)
R = 2

sigBx = 0.2
sigBy = 0.2

T_arr = []
T_arr2 = []
T_arr5 = []

for t in thetas:
    muBx = R * np.cos(t)
    muBy = R * np.sin(t)

    B = np.hstack((
        rng.normal(loc=muBx, scale=sigBx, size=nB)[:, np.newaxis],
        rng.normal(loc=muBy, scale=sigBy, size=nB)[:, np.newaxis]
    ))

    B2 = np.hstack((
        rng.normal(loc=muBx, scale=sigBx*2, size=nB)[:, np.newaxis],
        rng.normal(loc=muBy, scale=sigBy*2, size=nB)[:, np.newaxis]
    ))

    B5 = np.hstack((
        rng.normal(loc=muBx, scale=sigBx*5, size=nB)[:, np.newaxis],
        rng.normal(loc=muBy, scale=sigBy*5, size=nB)[:, np.newaxis]
    ))


    T_arr.append(GoF(A, B).Point_to_Point_DissimExp())
    T_arr2.append(GoF(A, B2).Point_to_Point_DissimExp())
    T_arr5.append(GoF(A, B5).Point_to_Point_DissimExp())


plt.plot(thetas, (T_arr5-np.mean(T_arr5))/(np.mean(T_arr5)), label=r"$\sigma = %.2f$" % (5 * sigBx))
plt.plot(thetas, (T_arr2-np.mean(T_arr2))/(np.mean(T_arr2)), label=r"$\sigma = %.2f$" % (3 * sigBx))
plt.plot(thetas, (T_arr-np.mean(T_arr))/(np.mean(T_arr)), label=r"$\sigma = %.2f$" % (sigBx))


plt.xlabel(r"$\theta$", fontsize=14)
plt.ylabel(r"$\frac{T- \langle T\rangle}{\langle T \rangle}$", fontsize=14)
plt.grid()


plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


def plot_distrib():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(A[:, 0:1], A[:, 1:2], s=10, c='k', marker="x", label="A")
    ax1.scatter(B[:, 0:1], B[:, 1:2], s=10, c='r', marker="v", label="B")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax1.legend()
    plt.tight_layout()
    plt.show()
