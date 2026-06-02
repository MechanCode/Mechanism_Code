import os
import sys
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import numpy as np

_ZEROS_1 = np.zeros(1, dtype=np.float64)

# Add Laplace noise, vectorized over dim.

def add_noise(sensitivity, eps, histo, dim):
    noise = np.random.laplace(loc=0, scale=sensitivity / eps, size=dim)
    return np.asarray(histo, dtype=np.float64)[:dim] + noise

# Compute the distance between two histograms, vectorized.

def count_dis(histo_1, histo_2, dim):
    return np.sum(np.abs(histo_1[:dim] - histo_2[:dim])) / dim

# Compute the distance variance between two histograms, vectorized.

def count_vardis(histo_1, histo_2, dim):
    diff = histo_1[:dim] - histo_2[:dim]
    var_avgdis = np.dot(diff, diff) / dim
    var_exp_sum = np.sum(np.abs(diff)) / dim
    return var_avgdis, var_exp_sum

# Compute the average distance variance of a stream prefix, batch vectorized.

def agv_vardis(compute_num, window_size, published_stream, dim, length=None):
    length_ = length if length is not None else len(published_stream)
    total_ = window_size * compute_num

    if length_ >= total_ + 1:
        num_pairs = total_ - 1
    else:
        num_pairs = length_ - 2

    if num_pairs <= 0:
        return 0

    # When published_stream is a pre-allocated 2-D ndarray, this slice is a
    # zero-copy view and avoids the O(total*dim) memory copy every iteration.
    start_idx = length_ - 1 - num_pairs
    arr = published_stream[start_idx:length_]                # (num_pairs+1, dim)

    # Differences between all consecutive pairs at once
    diffs = arr[1:] - arr[:-1]                              # (num_pairs, dim)

    # vardis (E[X^2]) and varexp (E[|X|]) for every pair, no Python loop.
    vardis_arr = np.einsum('ij,ij->i', diffs, diffs) / dim  # (num_pairs,)
    varexp_arr = np.sum(np.abs(diffs), axis=1) / dim        # (num_pairs,)

    # Only count pairs where vardis > 0
    mask = vardis_arr > 0
    count_ = np.count_nonzero(mask)

    if count_ == 0:
        return 0

    num_var = np.sum(vardis_arr[mask])
    num_exp = np.sum(varexp_arr[mask])

    return num_var / count_ - (num_exp / count_) ** 2

# Update the optimal C
# epsilon_p: total privacy budegt in a sliding window
# sensitivity_p: sensitivity for adding noise to the released histogram

def update_optimalc(epsilon_p, sensitivity_p, compute_num, window_size, published_stream, dim, length=None):
    E_dis = agv_vardis(compute_num, window_size, published_stream, dim, length=length)

    #optimal_c = max(int(epsilon_p * np.sqrt(3 * E_dis) / (6 * sensitivity_p)), 1)
    theta_ = 1 / 2
    optimal_c = max(int((np.sqrt((1 - 2 * theta_)**2 * window_size**2 + (3 * theta_**2 * E_dis * epsilon_p**2) / sensitivity_p**2) - (1 - 2 * theta_) * window_size) / (6 * theta_)), 1)

    #test for using T to control svt
    # q = 10
    # optimal_c = max(int(epsilon_p * np.sqrt((theta_ * int(E_dis) + (1 - theta_) * q**2 * sensitivity_p**2) / (6 * (theta_ + 1))) / sensitivity_p), 1)

    return optimal_c

# Publish the data in the first window or first several windows

def warm_up_stage(epsilon, sensitivity_p, raw_stream, window_size, window_num, dim):
    c_init = window_size / 20
    sample_interval = int(window_size / c_init)
    total_for_warmup = window_size * window_num
    published_stream = []
    eps_consumed = []
    epsilon_warmup = epsilon / c_init

    svt_consumed = []

    for i in range(total_for_warmup):
        if i % sample_interval == 0:
            published_stream.append(add_noise(sensitivity_p, epsilon_warmup, raw_stream[i], dim))
            eps_consumed.append(epsilon_warmup)
            svt_consumed.append(0)
        else:
            published_stream.append(published_stream[i - 1])
            eps_consumed.append(0)
            svt_consumed.append(0)
    print("Finish warm-up stage!")
    return published_stream, eps_consumed, svt_consumed

# Compute the remaining epsilon at the last timestamp of a window, using a slice sum.

def compute_epsremain(epsilon_p, eps_consume, window_size):
    length_ = len(eps_consume)
    start = length_ - window_size + 1
    if start < 0:
        start = 0
    eps_con = sum(eps_consume[start:length_])
    return epsilon_p - eps_con


# Find the first sampled data in the current window.
def find_firstsample(eps_con, window_size):
    current = len(eps_con)
    for i in range(current - window_size + 1, current):
        if eps_con[i] > 0:
            return i

    return current

# The whole workflow of SPAS

def SPAS_workflow(epsilon, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, dim, seed):
    np.random.seed(seed)
    raw_stream = np.asarray(raw_stream, dtype=np.float64)
    n = len(raw_stream)

    epsilon_s = epsilon / 4
    epsilon_p = epsilon - epsilon_s
    eps_1 = epsilon_s / 2
    eps_2 = epsilon_s - eps_1

    # Pre-allocate contiguous numpy arrays.
    # This is the critical optimization: published_stream is now a 2-D ndarray.
    # In agv_vardis, slicing a ndarray produces a zero-copy view instead of
    # the previous O(window*dim) memory copy from list-to-array conversion.
    published_stream = np.empty((n, dim), dtype=np.float64)
    eps_consumed = np.zeros(n, dtype=np.float64)
    svt_consumed = np.zeros(n, dtype=np.float64)

    # Warm-up stage, inlined to write into pre-allocated arrays.
    c_init = window_size / 20
    sample_interval = int(window_size / c_init)
    total_for_warmup = window_size * windownum_warm
    epsilon_warmup = epsilon_p / c_init
    _warmup_scale = sensitivity_p / epsilon_warmup

    for i in range(total_for_warmup):
        if i % sample_interval == 0:
            published_stream[i] = raw_stream[i] + np.random.laplace(0, _warmup_scale, size=dim)
            eps_consumed[i] = epsilon_warmup
        else:
            published_stream[i] = published_stream[i - 1]
            # eps_consumed[i] and svt_consumed[i] are already 0
    print("Finish warm-up stage!")

    pub_len = total_for_warmup
    optimal_c = update_optimalc(epsilon_p, sensitivity_p, windownum_updateE, window_size,
                                published_stream, dim, length=pub_len)
    print("Finish warm-up stage! Optimal C:", optimal_c)

    # Follow-up stage.
    start_ = windownum_warm * window_size

    # Scalar noise for SVT threshold (same RNG position as original add_noise call)
    rho_0 = np.random.laplace(0, sensitivity_s / eps_1)

    for i in range(start_, n):
        # Print the current progress.
        if (i - start_) % 10000 == 0:
            print(f"Processing index {i}/{n}...{i/n:.2%} done.")
        # compute_epsremain inlined with a numpy slice view and np.sum.
        ws_start = i - window_size + 1
        if ws_start < 0:
            ws_start = 0
        eps_remain = epsilon_p - np.sum(eps_consumed[ws_start:i])

        T_ = optimal_c * sensitivity_p / epsilon_p
        eps_per_c = epsilon_p / optimal_c

        # count_dis inlined.
        diff = np.sum(np.abs(raw_stream[i] - raw_stream[i - 1])) / dim

        # Scalar SVT noise inlined.
        v_0 = np.random.laplace(0, sensitivity_s / (eps_2 / (2 * optimal_c)))

        # find_firstsample inlined and vectorized.
        seg = eps_consumed[ws_start:i]
        nz = np.flatnonzero(seg)
        first_sample_inwindow = (ws_start + nz[0]) if len(nz) > 0 else i

        remaining_steps = window_size - (i - first_sample_inwindow)

        if remaining_steps <= int(eps_remain / eps_per_c):
            svt_start = first_sample_inwindow  # = i - (i - first_sample_inwindow + 1) + 1
            eps_svt_remain = eps_2 - np.sum(svt_consumed[svt_start:i])

            eps_pub = eps_remain / remaining_steps + eps_svt_remain / optimal_c
            published_stream[i] = raw_stream[i] + np.random.laplace(0, sensitivity_p / eps_pub, size=dim)
            svt_consumed[i] = eps_svt_remain / optimal_c
            eps_consumed[i] = eps_remain / remaining_steps

        elif diff + v_0 > T_ + rho_0 and eps_remain >= eps_per_c:
            svt_consumed[i] = eps_2 / optimal_c
            # compute_epsremain after appending svt: length = i+1
            svt_ws_start = i - window_size + 2
            if svt_ws_start < 0:
                svt_ws_start = 0
            eps_svt_remain = eps_2 - np.sum(svt_consumed[svt_ws_start:i + 1])
            if eps_svt_remain > 0:
                published_stream[i] = raw_stream[i] + np.random.laplace(0, sensitivity_p / (eps_per_c + eps_svt_remain), size=dim)
            else:
                published_stream[i] = raw_stream[i] + np.random.laplace(0, sensitivity_p / eps_per_c, size=dim)
            eps_consumed[i] = eps_per_c

        else:
            published_stream[i] = published_stream[i - 1]
            # eps_consumed[i] and svt_consumed[i] are already 0

        pub_len = i + 1
        optimal_c = update_optimalc(epsilon_p, sensitivity_p, windownum_updateE, window_size,
                                    published_stream, dim, length=pub_len)
        #print(optimal_c)

    return published_stream

# Test the SPAS and compute the corresponding metrics
# Flag == 0, varying epsilon
# Flag ==1, varying window size

# def run_SPAS(epsilon, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, round_, Flag_ = 0):
#     dim = len(raw_stream[0])
#     MAE_list = []

#     if Flag_ == 0:
#         for eps in epsilon:
#             MAE_ = 0
#             for i in range(round_):
#                 published_result = SPAS_workflow(eps, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, dim)
#                 MAE_ += count_mre(raw_stream, published_result)

#             MAE_ = MAE_ / round_
#             print("epsilon:", eps, "Done!")

#             MAE_list.append(MAE_)

#         print('SPAS DONE!')

#     else:
#         for w in window_size:
#             MAE_ = 0
#             for i in range(round_):
#                 published_result = SPAS_workflow(epsilon, sensitivity_s, sensitivity_p, raw_stream, w, windownum_warm, windownum_updateE, dim)
#                 MAE_ += count_mre(raw_stream, published_result)

#             MAE_ = MAE_ / round_
#             print("window size:", w, "Done!")

#             MAE_list.append(MAE_)

#         print('SPAS DONE!')

#     return MAE_list

# def run_SPAS_sum_query(epsilon, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, round_, query_num, Flag_ = 0):
#     dim = len(raw_stream[0])
#     query_MRE_list = []

#     if Flag_ == 0:
#         for eps in epsilon:
#             query_MRE = 0
#             for i in range(round_):
#                 published_result = SPAS_workflow(eps, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, dim)
#                 query_MRE += sum_query(raw_stream, published_result, query_num)

#             query_MRE = query_MRE / round_
#             print("epsilon:", eps, "Done!")

#             query_MRE_list.append(query_MRE)

#         print('SPAS sum query DONE!')

#     else:
#         for w in window_size:
#             query_MRE = 0
#             for i in range(round_):
#                 published_result = SPAS_workflow(epsilon, sensitivity_s, sensitivity_p, raw_stream, w, windownum_warm, windownum_updateE, dim)
#                 query_MRE += sum_query(raw_stream, published_result, query_num)

#             query_MRE = query_MRE / round_
#             print("window size:", w, "Done!")

#             query_MRE_list.append(query_MRE)

#         print('SPAS sum query DONE!')

#     return query_MRE_list

# def run_SPAS_count_query(epsilon, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, round_, query_num, Flag_ = 0):
#     dim = len(raw_stream[0])
#     query_MRE_list = []

#     if Flag_ == 0:
#         for eps in epsilon:
#             query_MRE = 0
#             for i in range(round_):
#                 published_result = SPAS_workflow(eps, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, dim)
#                 query_MRE += count_query(raw_stream, published_result, query_num)

#             query_MRE = query_MRE / round_
#             print("epsilon:", eps, "Done!")

#             query_MRE_list.append(query_MRE)

#         print('SPAS count query DONE!')

#     else:
#         for w in window_size:
#             query_MRE = 0
#             for i in range(round_):
#                 published_result = SPAS_workflow(epsilon, sensitivity_s, sensitivity_p, raw_stream, w, windownum_warm, windownum_updateE, dim)
#                 query_MRE += count_query(raw_stream, published_result, query_num)

#             query_MRE = query_MRE / round_
#             print("window size:", w, "Done!")

#             query_MRE_list.append(query_MRE)

#         print('SPAS count query DONE!')

#     return query_MRE_list

if __name__ == "__main__":

    raw_stream = data_reader('syn_arbit1')
    #epsilon = [0.1, 0.3, 0.5, 0.7, 0.9]
    epsilon = [1]
    sensitivity_s = 1
    sensitivity_p = 1
    window_size = 120
    # c_init = window_size / 5
    windownum_warm = 1
    windownum_updateE = 2
    round_ = 5

    error_ = run_SPAS(epsilon, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, round_)
    print(error_)
    #sum_query_err = run_SPAS_sum_query(epsilon, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, round_, 1000)
    #print(sum_query_err)
