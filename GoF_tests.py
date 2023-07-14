"""
This .py file contains the sub-routines to perform the various goodness of fit tests

Design notes:
OPP principles
All tests should take in the same kind of measured and predicted data arrays
All tests should out their test statistics only, P-value calculation from the test statistics is a separate function
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2


class bin_data:
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
        Args:
            data: data array to be binned
            N_init: initial guess for the number of bins
            min_occ: minimum occupancy for

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
            if min_count_idx == 0:
                merge_idx = 1
            elif min_count_idx == len(bin_counts) - 1:
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

        if predicted.shape != measured.shape:
            raise ValueError("The shapes of A and B are not the same.")
        else:
            self.predicted = predicted
            self.measured = measured

    def Chi2(self):
        """
        Returns: The Pearson chi-square statistic for the predicted and measured data arrays
        """

        # Binning the data
        # Always bin data based on theory i.e. bin based on predicted.
        init_data_to_bin = bin_data(self.predicted)  # call the data binning class
        pred_bin_counts, pred_bin_edges = init_data_to_bin.merge_bin()  # calls the specific binning algorith

        # bin measured data according to the same bin edges as the predicted data

        meas_bin_counts, meas_bin_edges = np.histogram(self.measured, bins=pred_bin_edges)

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
        DoF = len(meas_bin_counts)
        return chi2_stat, DoF

    def Pval_from_Chi2(self):

        chi2_stat, DoF = self.Chi2()
        Pvalue = 1 - chi2.cdf(chi2_stat, DoF)

        return Pvalue, chi2_stat, DoF




rng = np.random.default_rng(seed=1)
pred = rng.exponential(scale=1, size=1000)
epsilon = 0.02
meas = pred + epsilon * rng.normal(loc=1, scale=1, size=1000)

test = GoF(pred, meas)

print(test.Pval_from_Chi2())
