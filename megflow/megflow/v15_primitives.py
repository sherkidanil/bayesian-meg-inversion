"""Pure functions copied from references/Dataset_generation_15_09_2026.py.

Only formatting changed (AST unparse); no top-level downloads or dataset writes.
Source SHA256: e43a62983f5d8884040c24b3b25bb55060cff1b43aaf94a6502060da598a0e84.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import gamma


def Modeling_Liedfield_Error(G, beta_rad=0.00025, beta_tan=2.5e-06, rng=None):
    """
    Добавляет радиальный и тангенциальный шум к матрице G

    Параметры:
    G (np.ndarray): Исходная матрица формы (M, N).
    beta_rad (float): Коэффициент радиального шума.
    beta_tan (float): Коэффициент тангенциального шума.
    rng (np.random.Generator): Объект генератора случайных чисел (опционально).

    Возвращает:
    G_error (np.ndarray): Матрица с добавленным шумом.
    p_rad (np.ndarray): Вектор относительных радиальных ошибок для каждого столбца.
    p_tan (np.ndarray): Вектор относительных тангенциальных ошибок для каждого столбца.
    p_total (np.ndarray): Вектор суммарных относительных ошибок для каждого столбца.
    """
    if rng is None:
        rng = np.random.default_rng()
    N_sns, N_vert = G.shape
    ksi_sca = rng.standard_normal((1, N_vert))
    ksi_vec = rng.standard_normal((N_sns, N_vert))
    norm_G = np.linalg.norm(G, axis=0, keepdims=True)
    G_rad = beta_rad * ksi_sca * G
    term = G + beta_tan * ksi_vec * norm_G
    norm_term = np.linalg.norm(term, axis=0, keepdims=True)
    G_tan = term * (norm_G / norm_term) - G
    G_error = G + G_rad + G_tan
    nG = np.sqrt(np.mean(G * G, axis=0))
    ndG_rad = np.sqrt(np.mean(G_rad * G_rad, axis=0))
    p_rad = ndG_rad / nG
    ndG_tan = np.sqrt(np.mean(G_tan * G_tan, axis=0))
    p_tan = ndG_tan / nG
    dG_total = G_error - G
    ndG_total = np.sqrt(np.mean(dG_total * dG_total, axis=0))
    p_total = ndG_total / nG
    return (G_error, p_rad, p_tan, p_total)


def Sample_Activation_Windows(N_src, ntpoints, dur_samp, overlap_frac, rng=None):
    """
    изменение №3. окна активации теперь случайные величины, равномерно распределённые на интервалах,
    которые формируются из длительности источника и параметра перекрытия:
    L = ntpoints // N_src -- номинальный слот источника,
    s = overlap_frac * dur_samp / 2 -- допустимый выход за границы слота,
    t_start_j ~ U[max(0, j*L - s), min(ntpoints - dur_samp, (j+1)*L + s - dur_samp)],
    t_stop_j = t_start_j + dur_samp
    """
    if rng is None:
        rng = np.random.default_rng()
    L = ntpoints // N_src
    s = overlap_frac * dur_samp / 2.0
    t_start = np.zeros(N_src, dtype=int)
    for j in range(N_src):
        lo = max(0, int(round(j * L - s)))
        hi = min(ntpoints - dur_samp, int(round((j + 1) * L + s - dur_samp)))
        if hi < lo:
            raise ValueError(
                "окна не помещаются: увеличьте overlap_frac или уменьшите N_src / dur_samp"
            )
        t_start[j] = rng.integers(lo, hi + 1)
    return (t_start, t_start + dur_samp)


def Temporal_Activation(params, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N_src = params["N_src"]
    ntpoints = params["ntpoints"]
    t_start_sources = params["t_start_sources"]
    t_stop_sources = params["t_stop_sources"]
    frec_source_band = params["frec_source_band"]
    pad = params["pad"]
    shape = params["shape"]
    scale = params["scale"]
    qL = params["qL"]
    qR = params["qR"]
    zL = gamma.ppf(qL, a=shape, scale=scale)
    zR = gamma.ppf(qR, a=shape, scale=scale)
    el_time_activation = np.zeros((N_src, ntpoints), dtype=float)
    for j in range(N_src):
        start = t_start_sources[j]
        stop = t_stop_sources[j]
        nseg = stop - start
        q = rng.standard_normal(nseg + 2 * pad)
        b, a = butter(3, frec_source_band[j, :], btype="bandpass")
        s = filtfilt(b, a, q) / np.mean(frec_source_band[j, :])
        x = np.arange(start, stop)
        z = zL + (zR - zL) * (x - start) / nseg
        mask = (
            (zR - zL)
            / nseg
            * (
                gamma.pdf(z, a=shape, scale=scale)
                / (
                    gamma.cdf(zR, a=shape, scale=scale)
                    - gamma.cdf(zL, a=shape, scale=scale)
                )
            )
        )
        mask = mask / (mask.std(ddof=1) + 1e-12)
        s_seg = s[:nseg]
        s_seg = s_seg / (s_seg.std(ddof=1) + 1e-12)
        el_time_activation[j, x] = mask * s_seg
    return el_time_activation


def Generate_Brain_Noise(G, params, rng=None):

    def Generate_noise_trial(
        G, noise_trial_length, add_len, N_noise_sources, Fs, rng=None
    ):
        if rng is None:
            rng = np.random.default_rng()
        Nsrc = G.shape[1]
        SrcIndex = rng.integers(0, Nsrc, size=N_noise_sources)
        q = rng.standard_normal((N_noise_sources, noise_trial_length + 2 * add_len))
        nyq = Fs / 2.0
        alpha_band = np.array([8.0, 12.0]) / nyq
        beta_band = np.array([15.0, 30.0]) / nyq
        gamma1_band = np.array([30.0, 50.0]) / nyq
        gamma2_band = np.array([50.0, 70.0]) / nyq
        b_alpha, a_alpha = butter(3, alpha_band, btype="bandpass")
        b_beta, a_beta = butter(3, beta_band, btype="bandpass")
        b_gamma1, a_gamma1 = butter(3, gamma1_band, btype="bandpass")
        b_gamma2, a_gamma2 = butter(3, gamma2_band, btype="bandpass")
        SourceNoise = (
            filtfilt(b_alpha, a_alpha, q, axis=1) / np.mean(alpha_band)
            + filtfilt(b_beta, a_beta, q, axis=1) / np.mean(beta_band)
            + filtfilt(b_gamma1, a_gamma1, q, axis=1) / np.mean(gamma1_band)
            + filtfilt(b_gamma2, a_gamma2, q, axis=1) / np.mean(gamma2_band)
        )
        BrainNoiseTrial = (
            G[:, SrcIndex]
            @ SourceNoise[:, add_len : add_len + noise_trial_length]
            / N_noise_sources
        )
        return BrainNoiseTrial

    Fs = params["Fs"]
    ntpoints = params["ntpoints"]
    window = params["window"]
    N_noise_sources = params["N_noise_sources"]
    pad = params["pad"]
    noise_trial_length = int(window * Fs)
    noise_trials_count = int(np.floor(ntpoints / noise_trial_length + 1))
    Noise = np.zeros((G.shape[0], noise_trials_count * noise_trial_length))
    for ntidx in range(noise_trials_count):
        start = ntidx * noise_trial_length
        stop = start + noise_trial_length
        Noise[:, start:stop] = Generate_noise_trial(
            G, noise_trial_length, pad, N_noise_sources, Fs, rng=rng
        )
    Noise = Noise[:, :ntpoints]
    Noise = Noise / (Noise.reshape(-1).std(ddof=1) + 1e-12)
    return Noise
