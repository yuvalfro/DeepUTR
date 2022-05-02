import os
import sys
from sklearn.utils import check_array
#import fcntl
import logomaker
from itertools import product
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import stats

from sklearn.linear_model import LinearRegression

import statistics

from statistics import mean, stdev

import pathlib

###############paths######################
home_dir = str(pathlib.Path(__file__).parent.parent.parent.absolute())+"/"
files_dir = home_dir+"files/"
#home_dir = "/data/yaishof/deg_project_revision/deg_project_revision/"


seq_PATH = files_dir+"dataset/mRNA_sequences.csv"
A_minus_normalized_levels_PATH = files_dir + \
    "dataset/A_minus_normalized_levels.csv"
A_plus_normalized_levels_PATH = files_dir + \
    "dataset/A_plus_normalized_levels.csv"
validation_seq_PATH = files_dir+"dataset/validation_seq.csv"
validation_A_minus_normalized_levels_PATH = files_dir + \
    "dataset/validation_A_minus_normalized_levels.csv"
validation_A_plus_normalized_levels_PATH = files_dir + \
    "dataset/validation_A_plus_normalized_levels.csv"
features_1_7_kmers_PATH = files_dir+"dataset/features_1-7kmers.sav"
features_1_8_kmers_PATH = files_dir+"dataset/RF_features_1-8kmers.sav"

A_minus_secondary_PATH = files_dir+"dataset/combined_profile_110_A_minus.txt"
A_plus_secondary_PATH = files_dir+"dataset/combined_profile_110_A_plus.txt"
validation_A_minus_secondary_PATH = files_dir + \
    "dataset/combined_profile_validation_110_A_minus.txt"
validation_A_plus_secondary_PATH = files_dir + \
    "dataset/combined_profile_validation_110_A_plus.txt"

RESA_data_PATH = files_dir+"dataset/RESA_WindowSequences.csv"

split_to_train_validation_test_disjoint_sets_ids_PATH = files_dir + \
    'dataset/split_to_train_validation_test_disjoint_sets_ids.csv'

split_to_train_validation_test_disjoint_sets_minimal_cov_ids_PATH = files_dir + \
    'dataset/split_to_train_validation_test_disjoint_sets_ids_minimal_cov_and_minimal_mRNA_set.csv'


###############one hot######################
# one hot encoding function
def one_hot_encoding(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    # return seq2  # -use if using embbiding TODO support this from run if need
    return np.eye(4)[seq2].astype('uint8')


###############kmer count######################
# kmer utilities functions


def CountKmer(seq, k):
    kFreq = {}
    for i in range(0, len(seq)-k+1):
        kmer = seq[i:i+k]
        if kmer in kFreq:
            kFreq[kmer] += 1
        else:
            kFreq[kmer] = 1
    return kFreq


def retutnAllKmers(min_kmer_length, max_kmer_length):
    kmer_list = []
    for i in range(min_kmer_length, max_kmer_length+1):
        kmer_list = kmer_list + [''.join(c) for c in product('ACGT', repeat=i)]
    return kmer_list


def createFeturesVector(allKmers, seqkMerCounter):
    AllKmersSize = len(allKmers)
    KmerCounterArray = np.zeros((AllKmersSize, 1))
    for i in range(0, AllKmersSize):
        if allKmers[i] in seqkMerCounter:
            KmerCounterArray[i] = seqkMerCounter[allKmers[i]]
    return KmerCounterArray


def createFeturesVectorsForAllSeq(allKmers, min_kmer_length, max_kmer_length, sequences):
    num_of_kmers = len(allKmers)
    num_of_sequences = len(sequences)
    FeturesVectorsOfAllSeq = np.zeros(
        (num_of_sequences, num_of_kmers, 1), dtype='int8')
    for i in range(num_of_sequences):
        seq = sequences[i]
        seqkMerCounter = {}
        for j in range(min_kmer_length, max_kmer_length+1):
            seqkMerCounter = {**seqkMerCounter, **CountKmer(seq, j)}
        FeturesVectorsOfAllSeq[i] = createFeturesVector(
            allKmers, seqkMerCounter)
    return FeturesVectorsOfAllSeq


def mean_square_percentage_error(y_true, y_pred):
    y_true = check_array(y_true.reshape(-1, 1))
    y_pred = check_array(y_pred.reshape(-1, 1))

    # Note: does not handle mix 1d representation
    # if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true))

###############evaluate model######################
# evaluate the model using different metrics


def evaluate_model(y_predicted, y_observed, compute_mean_and_std=True):
    # if y_predicted contains NaN values then don't proceed with the evaluation and return NaNs as a result.
    if (np.isnan(np.sum(y_predicted))):
        return [(np.nan, np.nan) for i in range(4)]

    y_predicted = np.squeeze(np.asarray(y_predicted))
    y_observed = np.squeeze(np.asarray(y_observed))

    print("evaluate overall:  ", "pearson:", pearsonr(
        y_predicted.flatten(), y_observed.flatten()))
    print("evaluate overall:  ", "RMSE:", mean_squared_error(
        y_predicted.flatten(), y_observed.flatten(), squared=False))

    # evaluate overall by time points and average
    time_points_pearson = [pearsonr(y_predicted[:, i], y_observed[:, i])[
        0] for i in range(y_predicted.shape[1])]
    print("overall by time points pearson:", time_points_pearson)
    time_points_RMSE = [mean_squared_error(
        y_predicted[:, i], y_observed[:, i], squared=False) for i in range(y_predicted.shape[1])]
    print("overall by time points RMSE:", time_points_RMSE)
    print("evaluate overall by time points average:  ", "pearson mean:",
          mean(time_points_pearson), "std:", stdev(time_points_pearson))

    print("evaluate overall by time points average:  ", "RMSE mean:",
          mean(time_points_RMSE), "std:", stdev(time_points_RMSE))

    num_of_samples = len(y_predicted)
    #metrics_results_MAPE = np.zeros((num_of_samples,))
    metrics_results = {'R2': np.zeros((num_of_samples,)), 'pearson': np.zeros((num_of_samples,)), 'MSE': np.zeros(
        (num_of_samples,)), 'MAE': np.zeros((num_of_samples,)), 'RMSE': np.zeros((num_of_samples,))}
    for i in range(num_of_samples):
        y_predicted_i = y_predicted[i, :]
        metrics_results['R2'][i] = r2_score(y_observed[i, :], y_predicted_i)
        metrics_results['pearson'][i] = pearsonr(
            y_predicted_i, y_observed[i, :])[0]
        metrics_results['MSE'][i] = mean_squared_error(
            y_observed[i, :], y_predicted_i)
        metrics_results['RMSE'][i] = mean_squared_error(
            y_observed[i, :], y_predicted_i, squared=False)
        metrics_results['MAE'][i] = mean_absolute_error(
            y_observed[i, :], y_predicted_i)
        #metrics_results_MAPE[i] = mean_square_percentage_error(y_observed[i,:], y_predicted_i)

    if (compute_mean_and_std == False):
        return {
            **metrics_results,
            "pearson_overall_by_time_points": time_points_pearson,
            "RMSE_overall_by_time_points": time_points_RMSE
        }
    # else - compute mean and std for each metric and return the results
    means_and_stds = {}
    for key in metrics_results:
        if (key == 'pearson' or key == 'MSE' or key == 'RMSE'):
            means_and_stds[key+'_mean'] = np.mean(metrics_results[key])
            means_and_stds[key+'_std'] = np.std(metrics_results[key])
    print(pd.DataFrame([means_and_stds]).to_string(index=False))
    for key in ['R2', 'MAE']:
        means_and_stds[key+'_mean'] = np.mean(metrics_results[key])
        means_and_stds[key+'_std'] = np.std(metrics_results[key])

    print("pearson Cumulative:")
    histogram = pd.Series(metrics_results['pearson']).value_counts(
        bins=[-1, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], sort=False).cumsum()
    print(pd.DataFrame([histogram.values],
          columns=histogram.index).to_string(index=False))

    # print("MAPE Cumulative:")
    # histogram = pd.Series(metrics_results_MAPE).value_counts(bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 5, 7, 10, 20, 100, 200, 1000, 2000], sort=False).cumsum()
    # print(pd.DataFrame([histogram.values], columns=histogram.index).to_string(index=False))

    return {
        **{metric+"_mean": means_and_stds[metric+'_mean'] for metric in ['pearson', 'MSE', 'RMSE', 'R2', 'MAE']},
        **{metric+"_std": means_and_stds[metric+'_std'] for metric in ['pearson', 'MSE', 'RMSE', 'R2', 'MAE']},
        "pearson_overall_by_time_points_mean": mean(time_points_pearson),
        "pearson_overall_by_time_points_std": stdev(time_points_pearson),
        "RMSE_overall_by_time_points_mean": mean(time_points_RMSE),
        "RMSE_overall_by_time_points_std": stdev(time_points_RMSE)
    }

###############################evaluate slop test#######################################


def evaluate_slope_test(y_predicted, y_observed, compute_mean_and_std=True):
    # if y_predicted contains NaN values then don't proceed with the evaluation and return NaNs as a result.
    if (np.isnan(np.sum(y_predicted))):
        return [(np.nan, np.nan) for i in range(4)]

    y_predicted = np.squeeze(np.asarray(y_predicted))
    y_observed = np.squeeze(np.asarray(y_observed))

    num_of_samples = len(y_predicted)
    # compute the slopes and the absolute error (AE) bewteen observed slope a predicted slope
    if (y_predicted.shape[1] == 9):
        t = np.matrix([1, 2, 3, 4, 5, 6, 7, 8, 10]).T
    else:
        t = np.matrix([2, 3, 4, 5, 6, 7, 8, 10]).T
    observed_slops, predicted_slops, AE_values = np.zeros(
        (num_of_samples,)), np.zeros((num_of_samples,)), np.zeros((num_of_samples,))
    for i in range(num_of_samples):
        mdl = LinearRegression().fit(t, y_observed[i, :])
        observed_slops[i] = mdl.coef_[0]  # beta-slop
        mdl = LinearRegression().fit(t, y_predicted[i, :])
        predicted_slops[i] = mdl.coef_[0]  # beta-slop
        AE_values[i] = np.abs(observed_slops[i]-predicted_slops[i])

    return slope_test(predicted_slops, observed_slops, compute_mean_and_std, AE_values)


def slope_test(predicted_slops, observed_slops, compute_mean_and_std, AE_values=None):
    predicted_slops = np.squeeze(np.asarray(predicted_slops))
    observed_slops = np.squeeze(np.asarray(observed_slops))

    if (AE_values is None):
        # if AE is None, then it wasn't computed, and therfore compute it.
        num_of_samples = len(predicted_slops)
        AE_values = np.zeros((num_of_samples,))
        for i in range(num_of_samples):
            AE_values[i] = np.abs(observed_slops[i]-predicted_slops[i])

    pearson_corr = stats.pearsonr(predicted_slops, observed_slops)
    print("slope test (pearson, p-value):", pearson_corr)

    if (compute_mean_and_std == False):
        return {
            "pearson": pearson_corr,
            "AE": AE_values
        }
    # else - compute mean and std for the absolute error and return the results for this and the pearson test.
    MAE = np.mean(AE_values)
    MAE_std = np.std(AE_values)
    print("slope test (MAE, std): (", MAE, MAE_std, ")")

    return {
        "pearson": pearson_corr,
        "MAE": MAE,
        "MAE_std": MAE_std,
    }

###############################evaluate nonlinear fit subset#######################################


def evaluate_linear_or_nonlinear_subset(y_predicted, y_observed, subset='nonlinear', compute_mean_and_std=True, R2_threshold=0.7):
    print('evaluate for ', subset, ' fit subset')
    y_observed, y_predicted = drop_linear_or_nonlinear_subset(
        y_predicted=y_predicted, y_observed=y_observed, subset_to_retain=subset, R2_threshold=R2_threshold)

    # perform the evaluation without the initial point.
    return evaluate_model(y_predicted[:, 1:], y_observed[:, 1:], compute_mean_and_std)

###############################evaluate slope test for linear fit subset#######################################


def evaluate_slope_test_for_linear_or_nonlinear_subset(y_predicted, y_observed, subset='linear', compute_mean_and_std=True, R2_threshold=0.7):
    print('evaluate slope test for ', subset, ' fit subset')
    y_observed, y_predicted = drop_linear_or_nonlinear_subset(
        y_predicted=y_predicted, y_observed=y_observed, subset_to_retain=subset, R2_threshold=R2_threshold)

    return evaluate_slope_test(y_predicted, y_observed, compute_mean_and_std)

###############################drop linear or nonlinear subset#######################################


def drop_linear_or_nonlinear_subset(y_predicted, y_observed, subset_to_retain, R2_threshold):
    y_observed = np.squeeze(np.asarray(y_observed))

    num_of_samples = len(y_observed)
    # compute the slopes and the absolute error (AE) bewteen observed slope a predicted slope
    if (y_predicted.shape[1] == 9):
        t = np.matrix([1, 2, 3, 4, 5, 6, 7, 8, 10]).T
    else:
        t = np.matrix([2, 3, 4, 5, 6, 7, 8, 10]).T
    samples_indexs_to_delete_list = []
    for i in range(num_of_samples):
        mdl = LinearRegression().fit(t, y_observed[i, :])
        R2 = mdl.score(t, y_observed[i, :])
        if (subset_to_retain == 'linear'):
            if (R2 < R2_threshold):
                samples_indexs_to_delete_list.append(i)
        else:
            if (R2 >= R2_threshold):
                samples_indexs_to_delete_list.append(i)

    y_observed = np.delete(y_observed, samples_indexs_to_delete_list, axis=0)
    y_predicted = np.delete(y_predicted, samples_indexs_to_delete_list, axis=0)

    return y_observed, y_predicted

###############################compute Linear regression slopes#######################################


def compute_LR_slopes(values_array):
    num_of_samples = len(values_array)
    # compute the slopes
    if (values_array.shape[1] == 9):
        t = np.matrix([1, 2, 3, 4, 5, 6, 7, 8, 10]).T
    else:
        t = np.matrix([2, 3, 4, 5, 6, 7, 8, 10]).T
    slops = np.zeros((num_of_samples,))
    for i in range(num_of_samples):
        mdl = LinearRegression().fit(t, values_array[i, :])
        slops[i] = mdl.coef_[0]  # beta-slope

    return slops


###############################create Logo object#######################################
# create Logo object


def create_DNA_logo(PWM_df, secondary_color=False, figsize=(10, 2.5), labelpad=-1, ax=None, y_label="IG"):
    if(secondary_color):
        color_scheme = 'NajafabadiEtAl2017'
    else:
        color_scheme = 'classic'

    IG_logo = logomaker.Logo(PWM_df,
                             shade_below=.5,
                             fade_below=.5,
                             color_scheme=color_scheme,
                             font_name='Arial Rounded MT Bold',
                             ax=ax,
                             figsize=figsize)

    IG_logo.style_spines(visible=False)
    IG_logo.style_spines(spines=['left', 'bottom'], visible=True)
    IG_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    IG_logo.ax.set_ylabel(y_label, labelpad=labelpad)
    # IG_logo.ax.set_xlabel(string)
    IG_logo.ax.xaxis.set_ticks_position('none')
    IG_logo.ax.xaxis.set_tick_params('both')
    IG_logo.ax.set_xticklabels([])

    return IG_logo


###############################lock using lock file######################################


# def acquireLock():
#     ''' acquire exclusive lock file access '''
#     locked_file_descriptor = open('lockfile.LOCK', 'w+')
#     fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
#     return locked_file_descriptor
#
#
# def releaseLock(locked_file_descriptor):
#     ''' release exclusive lock file access '''
#     locked_file_descriptor.close()


######################################################################################

# Disable


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore


def enablePrint():
    sys.stdout = sys.__stdout__


#########################get_default_args#############################################
import inspect

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }



#######################################################################################
###taken from https://github.com/kundajelab/deeplift/blob/b6ebbdd1a497698c96850c2e9f71e5a5e6860e17/deeplift/dinuc_shuffle.py#L4 #####


def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def dinuc_shuffle(seq, num_shufs=None, seed=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `seed`: a NumPy seed, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if seed is not None:
        np.random.seed(seed)

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            # Keep last index same
            inds[:-1] = np.random.permutation(len(inds) - 1)
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]
