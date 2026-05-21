from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


@dataclass
class InferenceResult:
    dx: list
    icov: list
    s: np.ndarray
    s_joint: np.ndarray
    gamma_opt: float
    x_array: list
    error_bars: np.ndarray
    s_joint_error_bars: np.ndarray

    def as_list(self):
        return [
            self.dx,
            self.icov,
            self.s,
            self.s_joint,
            self.gamma_opt,
            self.x_array,
            self.error_bars,
            self.s_joint_error_bars,
        ]

    def __iter__(self):
        return iter(self.as_list())

    def __getitem__(self, index):
        return self.as_list()[index]

    def __len__(self):
        return len(self.as_list())


def find_last_below_threshold(nums_in, th=0.1):
    idx_out = 0
    for i, value in enumerate(nums_in):
        if value >= th:
            break
        idx_out = i
    return idx_out

def get_best_regularization(corrs, gamma_values, corr_cutoff_pct=0.5):
    '''
    Compute best regularization strength from correlation data.
    '''
    max_corrs = max(corrs)
    corr_thresh = (max_corrs**2 - corrs[0]**2) * corr_cutoff_pct
    
    if abs(max_corrs**2 - corrs[0]**2) < 0.01:
        gamma_opt = gamma_values[0]
    else:
        gamma_set = False
        
        delta_cor_set, i_set = [], []
        for i in range(np.argmax(corrs), 1, -1):
            delta_cor_num = corrs[i]**2 - corrs[i-1]**2
            delta_cor_den = (np.log10(gamma_values[i]) - np.log10(gamma_values[i-1]))
            delta_cor_ratio = abs(delta_cor_num / delta_cor_den)
            delta_cor_set.append(abs(max_corrs) - abs(corrs[i]))
            i_set.append(i)
            if delta_cor_ratio >= corr_thresh:
                gamma_opt = gamma_values[i]
                gamma_set = True
                break
        if not gamma_set:
            if not i_set:
                gamma_opt = gamma_values[np.argmax(corrs)]
            else:
                i_before_below10percent = find_last_below_threshold(delta_cor_set)
                idx = min(i_before_below10percent, len(i_set) - 1)
                gamma_opt = gamma_values[i_set[idx]]
    
    return gamma_opt


def plot_regularization(corrs, gamma_values):
    '''
    Plot correlation as a function of regularization strength.
    '''
    
    plt.plot(gamma_values, corrs)
    plt.xscale('log')
    plt.xlabel('Regularization strength (gamma)')
    plt.ylabel('Average correlation between replicates')
    plt.show()


def plot_regularization_all(corrs_list, gamma_values):
    '''
    Plot correlation as a function of regularization strength for multiple data sets.
    '''    
    corrs_list = np.array(corrs_list).T  # rows = replicate pairs, cols = gamma values

    for i in range(corrs_list.shape[0]):
        plt.plot(gamma_values, corrs_list[i], label=f"Pair {i + 1}")

    plt.xscale('log')
    plt.xlabel('Regularization strength (gamma)')
    plt.ylabel('Correlation between replicates')
    plt.legend()
    plt.show()


def _prepare_eigendecomp(dx, icov, n_replicates):
    dx_arr = np.array(dx)
    eig_lam = [None] * n_replicates
    eig_vec = [None] * n_replicates
    vt_dx = [None] * n_replicates

    # For each replicate, selection solves have the form
    # (C + gamma I) s = dx. Since C = V diag(lambda) V.T,
    # s = V @ ((V.T @ dx) / (lambda + gamma)). Cache V, lambda,
    # and V.T @ dx once so gamma sweeps avoid repeated matrix inversions.
    for r in range(n_replicates):
        eig_lam[r], eig_vec[r] = np.linalg.eigh(icov[r])
        vt_dx[r] = eig_vec[r].T @ dx_arr[r]

    icov_sum = np.sum(icov, axis=0)
    dx_sum = np.sum(dx_arr, axis=0)
    lam_j, V_j = np.linalg.eigh(icov_sum)
    vt_dx_j = V_j.T @ dx_sum

    return dx_arr, eig_lam, eig_vec, vt_dx, lam_j, V_j, vt_dx_j


def infer_gamma_range(sequence_dataframe, sequence_to_feature, n_replicates=1,
                      gamma_values=None, max_reads=1e2):
    """
    Infer selection coefficients across a range of gamma values.

    Returns
    -------
    gamma_values : np.ndarray
        The gamma values used.
    s_by_gamma : np.ndarray, shape (n_gamma, n_replicates, L)
        Per-replicate selection coefficients for each gamma.
    s_joint_by_gamma : np.ndarray, shape (n_gamma, L)
        Joint selection coefficients for each gamma.
    """
    dx, icov, _ = compute_dx_covariance_esm(sequence_dataframe, sequence_to_feature)

    if gamma_values is None:
        gamma_values = np.logspace(np.log10(1 / max_reads), 4, num=20)
    gamma_values = np.asarray(gamma_values)

    L = len(dx[0])
    n_gamma = len(gamma_values)
    _, eig_lam, eig_vec, vt_dx, lam_j, V_j, vt_dx_j = _prepare_eigendecomp(dx, icov, n_replicates)
    s_by_gamma = np.zeros((n_gamma, n_replicates, L))
    s_joint_by_gamma = np.zeros((n_gamma, L))

    for g_idx, g in enumerate(gamma_values):
        for r_idx in range(n_replicates):
            s_by_gamma[g_idx, r_idx] = eig_vec[r_idx] @ (vt_dx[r_idx] / (eig_lam[r_idx] + g))
        s_joint_by_gamma[g_idx] = V_j @ (vt_dx_j / (lam_j + g))

    return gamma_values, s_by_gamma, s_joint_by_gamma


# ==============================================================================
# Approach: full per-sequence covariance matrix (don't even bother with other methods)
# ==============================================================================

def compute_dx_covariance_esm(sequence_dataframe, sequence_to_feature, plot_icov=False):
    """Compute the mean change (dx) and time-integrated full covariance matrix
    (icov) from a sequence dataframe and sequence-to-feature mapping.

    The covariance matrix at each time point is computed directly from
    per-sequence features:

        C_kl(t) = E[z_k * z_l](t) - E[z_k](t) * E[z_l](t)

    and integrated over time via the trapezoidal rule.  Because C(t) is a sum
    of weighted outer products it is guaranteed positive semi-definite, so the
    resulting icov is a valid PSD matrix suitable for inversion and error bars.

    Parameters
    ----------
    sequence_dataframe : DataFrame with columns SequenceIndex, Generation, Frequency, Replicate
    sequence_to_feature : dict mapping SequenceIndex -> feature vector

    Returns
    -------
    dx      : list of ndarray, length n_replicates, each shape (d,)
    icov    : list of ndarray, length n_replicates, each shape (d, d)
    x_array : list of ndarray, length n_replicates, each shape (n_times, d)
    """

    sequence_indices = sorted(sequence_dataframe['SequenceIndex'].unique())
    sequence_to_feature_row = {seq_idx: i for i, seq_idx in enumerate(sequence_indices)}
    feature_table = np.vstack([sequence_to_feature[seq_idx] for seq_idx in sequence_indices])
    sequence_dataframe = sequence_dataframe.copy()
    sequence_dataframe['_FeatureRow'] = sequence_dataframe['SequenceIndex'].map(sequence_to_feature_row)

    rep_vals = sorted(sequence_dataframe['Replicate'].unique())
    reps = len(rep_vals)
    d    = feature_table.shape[1]

    dx      = [np.zeros(d)       for _ in range(reps)] # MEAN change in feature
    icov    = [np.zeros((d, d))  for _ in range(reps)] # INTEGRATED covariance matrix
    x_array = []

    for r_idx, (rep_val, df_rep) in enumerate(sequence_dataframe.groupby('Replicate', sort=True)):
        times   = np.sort(np.unique(df_rep['Generation']))
        n_times = len(times)
        generation_groups = {
            generation: df_t
            for generation, df_t in df_rep.groupby('Generation', sort=True)
        }

        # Trapezoid weights for time integration
        trap_weights        = np.zeros(n_times)
        trap_weights[0]     = (times[1] - times[0]) / 2
        trap_weights[-1]    = (times[-1] - times[-2]) / 2
        for i in range(1, n_times - 1):
            trap_weights[i] = (times[i + 1] - times[i - 1]) / 2

        x = np.zeros((n_times, d))     # population mean feature
        M = np.zeros((n_times, d, d))  # population second-moment matrix

        for i, t in enumerate(times):
            df_t       = generation_groups[t]
            total_freq = df_t['Frequency'].sum()
            feature_mat = feature_table[df_t['_FeatureRow'].to_numpy()] # (n_seqs, d)
            w          = df_t['Frequency'].values / total_freq  # (n_seqs,)
            x[i]       = w @ feature_mat                       # weighted mean feature (dot product of (1, n_seqs) and (n_seqs, d) -> (d,))
            M[i]       = (feature_mat.T * w) @ feature_mat      # weighted second-moment matrix (dot product of (d, n_seqs) and (n_seqs, d) -> (d, d))

        x_array.append(x)
        dx[r_idx] = x[-1] - x[0]

        # Compute C(t) for all time points and accumulate icov
        if plot_icov:
            C_all = np.zeros((n_times, d, d))
            for i in range(n_times):
                C_all[i]     = M[i] - np.outer(x[i], x[i])
                icov[r_idx] += trap_weights[i] * C_all[i]
        else:
            for i in range(n_times):
                icov[r_idx] += trap_weights[i] * (M[i] - np.outer(x[i], x[i]))

        if plot_icov:
            # trace(C(t)): instantaneous total variance across all feature dims
            trace_C = np.array([np.trace(C_all[i]) for i in range(n_times)])

            # Frobenius norm of cumulative integral up to each time point
            cumulative_icov = np.zeros((n_times, d, d))
            for i in range(n_times):
                cumulative_icov[i] = cumulative_icov[i - 1] + trap_weights[i] * C_all[i] if i > 0 else trap_weights[i] * C_all[i]
            frob_cumicov = np.array([np.linalg.norm(cumulative_icov[i], 'fro') for i in range(n_times)])

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'Replicate {r_idx + 1}')

            axes[0].plot(times, trace_C, marker='o', ms=4)
            axes[0].set_xlabel('Generation')
            axes[0].set_ylabel('tr(C(t))')
            axes[0].set_title('Instantaneous total variance')

            axes[1].plot(times, frob_cumicov, marker='o', ms=4)
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel(r'$\|\int_0^t C\, dt\|_F$')
            axes[1].set_title('Cumulative ||icov(t)||  (should plateau)')

            plt.tight_layout()
            plt.show()
            
    return dx, icov, x_array


def mini_infer_esm(sequence_dataframe, sequence_to_feature, 
                    gamma=None, corr_cutoff_pct=0.5,
                    max_reads=1e2, plot_gamma=False, 
                    verbose=False, calc_error_bars=False,
                    plot_icov=False):
    """Infer selection coefficients on ESM feature dimensions using the full
    per-sequence covariance matrix (Approach 2 / full-covariance).

    The covariance matrix is computed directly from per-sequence features so
    it is guaranteed PSD, and error bars from sqrt(diag(inv(C + gamma*I))) are
    always valid.

    Returns
    -------
    InferenceResult
    """
    n_replicates = len(sequence_dataframe['Replicate'].unique())

    dx, icov, x_array = compute_dx_covariance_esm(sequence_dataframe, sequence_to_feature, 
                                                    plot_icov=plot_icov)
    L = len(dx[0])
    _, eig_lam, eig_vec, vt_dx, lam_j, V_j, vt_dx_j = _prepare_eigendecomp(dx, icov, n_replicates)

    if gamma is not None:
        gamma_opt = gamma
    elif n_replicates == 1:
        if verbose:
            print('Only one replicate, setting gamma = 1')
        gamma_opt = 1
    else:
        gamma_vals = np.logspace(np.log10(1 / max_reads), 4, num=20)
        corrs = []
        for g in gamma_vals:
            s_temp = np.zeros((n_replicates, L))
            for r_idx in range(n_replicates):
                s_temp[r_idx] = eig_vec[r_idx] @ (vt_dx[r_idx] / (eig_lam[r_idx] + g))
            corrs.append(np.mean([
                pearsonr(s_temp[i], s_temp[j]).statistic
                for i in range(n_replicates)
                for j in range(i + 1, n_replicates)
            ]))
        if plot_gamma and verbose:
            print(f"gamma_values: {gamma_vals}")
            print(f"corrs: {corrs}")
        gamma_opt = get_best_regularization(corrs, gamma_vals, corr_cutoff_pct)

    s = np.zeros((n_replicates, L))
    error_bars = np.zeros((n_replicates, L))
    for r_idx in range(n_replicates):
        inv_denom = 1.0 / (eig_lam[r_idx] + gamma_opt)
        s[r_idx] = eig_vec[r_idx] @ (vt_dx[r_idx] * inv_denom)
        if calc_error_bars:
            error_bars[r_idx] = np.sqrt(eig_vec[r_idx] ** 2 @ inv_denom)

    inv_denom_j = 1.0 / (lam_j + n_replicates * gamma_opt)
    s_joint = V_j @ (vt_dx_j * inv_denom_j)
    s_joint_error_bars = np.sqrt(V_j ** 2 @ inv_denom_j**2) if calc_error_bars else np.full(L, np.nan)

    return InferenceResult(dx, icov, s, s_joint, gamma_opt, x_array, error_bars, s_joint_error_bars)
