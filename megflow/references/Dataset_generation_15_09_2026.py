import numpy as np
import mne
from mne.datasets import sample
from mne import spatial_src_adjacency
import networkx as nx
from scipy.signal import butter, filtfilt
from scipy.stats import gamma
from scipy.stats import truncnorm # изменение №2. усечённое нормальное распределение для радиусов источников

#%% Functions

def Modeling_Liedfield_Error(G, beta_rad=0.00025, beta_tan=0.0000025, rng=None):
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

    # ksi_sca [1 x N_vert] one scalar by each vertex (column g_k)
    ksi_sca = rng.standard_normal((1, N_vert))

    # ksi_vec [N_sns, N_vert] one vector by each vertex (column g_k)
    ksi_vec = rng.standard_normal((N_sns, N_vert))

    norm_G = np.linalg.norm(G, axis=0, keepdims=True)

    # Radial component
    # G_rad = beta_rad * ksi_sca * norm_G * G
    G_rad = beta_rad * ksi_sca * G

    # Tangential component
    # term = G + beta_tan * ksi_vec
    term = G + beta_tan * ksi_vec * norm_G
    norm_term = np.linalg.norm(term, axis=0, keepdims=True)
    G_tan = term * (norm_G / norm_term) - G

    G_error = G + G_rad + G_tan

    # Базовый знаменатель (норма исходного сигнала)
    nG = np.sqrt(np.mean(G * G, axis=0))

    # 1. Радиальная ошибка
    ndG_rad = np.sqrt(np.mean(G_rad * G_rad, axis=0))
    p_rad = ndG_rad / nG

    # 2. Тангенциальная ошибка
    ndG_tan = np.sqrt(np.mean(G_tan * G_tan, axis=0))
    p_tan = ndG_tan / nG

    # 3. Суммарная ошибка
    dG_total = G_error - G
    ndG_total = np.sqrt(np.mean(dG_total * dG_total, axis=0))
    p_total = ndG_total / nG

    return G_error, p_rad, p_tan, p_total

def get_hemi(v, n_lh_vertices):
    return 'lh' if v < n_lh_vertices else 'rh'

def same_hemi(v1, v2, n_lh_vertices):
    return (v1 < n_lh_vertices and v2 < n_lh_vertices) or (v1 >= n_lh_vertices and v2 >= n_lh_vertices)

def Spatial_Activation(params, Graph, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    N_src = params["N_src"]
    min_dist = params["min_dist"]
    R_mm = params["R_mm"]
    sigma_mm = R_mm * params["sigma_mm_scale"]
    A0 = params["A0"]

    centroids = np.zeros(N_src, dtype=int)
    spatial_activation = np.zeros((N_src, N_vert))

    # first source
    centroids[0] = rng.integers(N_vert)
    d = nx.single_source_dijkstra_path_length(Graph, centroids[0], weight='weight')
    d_vec = np.full(N_vert, np.inf)
    for k, v in d.items():
        d_vec[k] = v
        # distances from centroids[0] to other vertices
        # may contain inf signifying the absence of connection (another hemisphere)
    idx_region = d_vec <= R_mm[0] # target area
    r = d_vec[idx_region]
    spatial_activation[0, idx_region] = A0[0] * np.exp(-0.5 * (r / sigma_mm[0])**2)
    # activation by the Gaussian function with a peak at the centroid

    # other sources
    for j in range(1, N_src):
        if j == 1:
            idx_centroids = d_vec >= min_dist
            # accessible vertices that are no closer than min_dist to the first centroid
        if j > 1:
            idx_centroids = idx_centroids & (d_vec >= min_dist)
            # accessible vertices that are no closer than min_dist to all previous centroids

        candidates = np.where(idx_centroids)[0]
        # изменение №7. защита от пустого списка кандидатов: при min_dist = 50 мм и большом N_src
        # подходящих вёртексов может не остаться, тогда симуляция генерируется заново (см. цикл генерации)
        if candidates.size == 0:
            raise RuntimeError("нет вёртексов, удовлетворяющих min_dist")
        centroids[j] = rng.choice(candidates)

        d = nx.single_source_dijkstra_path_length(Graph, centroids[j], weight='weight')
        d_vec = np.full(N_vert, np.inf)
        for k, v in d.items():
            d_vec[k] = v
            # distances from centroids[j] to other vertices

        idx_region = d_vec <= R_mm[j]
        r = d_vec[idx_region]
        spatial_activation[j, idx_region] = A0[j] * np.exp(-0.5 * (r / sigma_mm[j])**2) # [N_src x N_vert]

    return centroids, spatial_activation

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
            raise ValueError("окна не помещаются: увеличьте overlap_frac или уменьшите N_src / dur_samp")
        t_start[j] = rng.integers(lo, hi + 1)

    return t_start, t_start + dur_samp

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

    zL = gamma.ppf(qL, a=shape, scale=scale) # quantile corresponding to the qL
    zR = gamma.ppf(qR, a=shape, scale=scale) # quantile corresponding to the qR

    el_time_activation = np.zeros((N_src, ntpoints), dtype=float) # [N_src x ntpoints]

    for j in range(N_src):
        start = t_start_sources[j]
        stop  = t_stop_sources[j]
        nseg = stop - start # length of active segment (in samples)

        # white noise with time padding
        q = rng.standard_normal(nseg + 2*pad)

        b, a = butter(3, frec_source_band[j, :], btype='bandpass')
        s = filtfilt(b, a, q) / np.mean(frec_source_band[j, :])

        # construction of an envelope/modulating curve (truncated gamma between zL and zR)
        x = np.arange(start, stop)
        z = zL + (zR - zL) * (x - start) / nseg
        mask = ((zR - zL) / nseg) * ( gamma.pdf(z, a=shape, scale=scale) /
                                     ( gamma.cdf(zR, a=shape, scale=scale) - gamma.cdf(zL, a=shape, scale=scale) ) )

        # dispersion standardisation
        mask = mask / (mask.std(ddof=1) + 1e-12)
        s_seg = s[:nseg] # take only the required segment length
        s_seg = s_seg / (s_seg.std(ddof=1) + 1e-12)

        el_time_activation[j, x] = mask * s_seg # [N_src x ntpoints]

    return el_time_activation

def Generate_Brain_Noise(G, params, rng=None):

    def Generate_noise_trial(G, noise_trial_length, add_len, N_noise_sources, Fs, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        Nsrc = G.shape[1]

        SrcIndex = rng.integers(0, Nsrc, size=N_noise_sources) # indexes of selected noise sources

        q = rng.standard_normal((N_noise_sources, noise_trial_length + 2 * add_len)) # white noise with time padding

        nyq = Fs / 2.0
        alpha_band  = np.array([8.0, 12.0])  / nyq
        beta_band   = np.array([15.0, 30.0]) / nyq
        gamma1_band = np.array([30.0, 50.0]) / nyq
        gamma2_band = np.array([50.0, 70.0]) / nyq

        b_alpha,  a_alpha  = butter(3, alpha_band,  btype='bandpass')
        b_beta,   a_beta   = butter(3, beta_band,   btype='bandpass')
        b_gamma1, a_gamma1 = butter(3, gamma1_band, btype='bandpass')
        b_gamma2, a_gamma2 = butter(3, gamma2_band, btype='bandpass')

        SourceNoise = (
            filtfilt(b_alpha,  a_alpha,  q, axis=1) / np.mean(alpha_band) +
            filtfilt(b_beta,   a_beta,   q, axis=1) / np.mean(beta_band)  +
            filtfilt(b_gamma1, a_gamma1, q, axis=1) / np.mean(gamma1_band) +
            filtfilt(b_gamma2, a_gamma2, q, axis=1) / np.mean(gamma2_band)
        )

        BrainNoiseTrial = G[:, SrcIndex] @ SourceNoise[:, add_len:add_len + noise_trial_length] / N_noise_sources

        return BrainNoiseTrial

    Fs = params["Fs"]
    ntpoints = params["ntpoints"]
    window = params["window"]
    N_noise_sources = params["N_noise_sources"]
    pad = params["pad"]

    noise_trial_length =  int(window * Fs) # = 100 with Fs=1000
    noise_trials_count =  int(np.floor(ntpoints / noise_trial_length + 1))

    Noise = np.zeros((G.shape[0], noise_trials_count * noise_trial_length))

    for ntidx in range(noise_trials_count):
        start = ntidx * noise_trial_length
        stop  = start + noise_trial_length
        Noise[:, start:stop] = Generate_noise_trial(
            G, noise_trial_length, pad, N_noise_sources, Fs, rng=rng
        )

    Noise = Noise[:, :ntpoints] # take only the required length

    # dispersion standardisation
    Noise = Noise / (Noise.reshape(-1).std(ddof=1) + 1e-12)

    return Noise

#%% Loading data available in mne

data_path = sample.data_path()
subjects_dir = data_path / "subjects"
subject = "sample"
raw_fname = data_path / "MEG" / subject / "sample_audvis_raw.fif"
trans_fname = data_path / "MEG"/subject/"sample_audvis_raw-trans.fif"

src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir)
bem = mne.read_bem_solution(f'{data_path}/subjects/sample/bem/sample-5120-bem-sol.fif')
fwd = mne.make_forward_solution(raw_fname, trans=trans_fname, src=src, bem=bem,
                                 meg=True, eeg=False, mindist=5.0)

fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=True) # fix orientation, orthogonal to the cortical surface
src = fwd['src']

print(src)

#%% Getting the forward model matrix G

print(fwd)

fwd_mag = mne.pick_types_forward(fwd, meg='mag', eeg=False) # meg=True to include all MEG channels; meg='mag' to select only magnetometers; meg='grad' - all gradiometers; etc.
G = fwd_mag['sol']['data'] # [N_sensors x N_vert]; characterizes how a unit-amplitude neural source at a given location contributes to the measured signals at each sensor

N_sensors = G.shape[0]

#%% Parameters

# изменение №1. число источников теперь случайное от симуляции к симуляции,
# границы можно двигать не переписывая остальной код
N_src_min = 1
N_src_max = 5

# изменение №2. радиусы источников -- усечённое нормальное на [8, 25] мм со средним ~15 мм
r_min = 8.0
r_max = 25.0
mu = 14.515                                                  # параметры неусечённого распределения
sigma_r = 3.80                                               # (подобраны под целевое среднее 15.0 мм)
a_trunc = (r_min - mu) / sigma_r                             # границы в стандартной шкале z-scores (требование scipy)
b_trunc = (r_max - mu) / sigma_r
trunc_dist = truncnorm(a_trunc, b_trunc, loc=mu, scale=sigma_r)

# parameters for setting the source location (centroid and surrounding region) and spatial activation
# изменение №1. N_src, R_mm и A0 больше не задаются здесь: они свои у каждой симуляции
# и добавляются к этому словарю внутри цикла генерации
params_spatial_common = {
    "N_vert": G.shape[1],              # the number of using vertices
    "min_dist": 50.0,                  # mm, minimum distance between centroids
    "sigma_mm_scale": 1 / 3.0          # Gaussian width, 3*sigma = R_mm (99.72%), or 2*sigma = R_mm (95.44%)
    }

# parameters for time series of oscillations (temporal oscillatory activation)
duration = 2.0                                               # sec
Fs = 1000                                                    # Hz, sampling frequency of interest
ntpoints = round(Fs * duration)                              # the number of time samples
t = np.linspace(0, ntpoints / Fs * 1000 - 1, ntpoints)       # ms

# изменение №3. длительность активации источника и параметр перекрытия задают интервалы,
# на которых равномерно выбираются начало и конец активации каждого источника
dur_source_ms = 500                                          # ms, длительность активации одного источника
overlap_frac = 0.4                                           # доля перекрытия (0.4 -> 200 мс (100 с одной и 100 с другой) при 500 мс)
dur_samp = int(round(dur_source_ms * Fs / 1000))             # длительность активации в отсчётах

# изменение №4. частотный диапазон каждого источника выбирается случайно из четырёх возможных
freq_bands_hz = np.array([[4, 8],
                          [8, 12],
                          [12, 16],
                          [16, 20]], dtype=float)            # Hz, [n_bands x 2]

# изменение №1 и №3 и №4. t_start_sources, t_stop_sources и frec_source_band
# теперь тоже свои у каждой симуляции и добавляются к словарю внутри цикла генерации
params_temporal_common = {
    "ntpoints": ntpoints,
    "pad": 300,                                                            # padding
    "shape": 6.0,                                                          # parameters of the gamma distribution that modulates the oscillation
    "scale": 4.0,
    "qL": 1e-5,
    "qR": 0.999
    }

# parameters of noise generation
params_noise_generation = {
    "Fs": Fs,
    "ntpoints": ntpoints,
    "window": 0.1,
    "N_noise_sources": 100,
    "pad": 300,
    "noise_level": 0.3,
    "beta_rad": 0.13, # ожидаемый процент ошибки ~ beta_rad * 0.8 (где 0.8 ~ np.sqrt(2 / np.pi) среднее полунормального распределения N(0, 1))
    "beta_tan": 0.001 # ожидаемый процент ошибки ~ beta_tan * 10 (где 10 ~ np.sqrt(N_sensors), при N_sensors = 102)
    }

# изменение №5. параметры фильтрации, которая теперь применяется к каждому сэмплу датасета
params_filter = {
    "l_freq": 2.0,
    "h_freq": 40.0,
    "method": 'iir',
    "order": 4,
    "ftype": 'butter'
    }

noise_level = params_noise_generation["noise_level"]
beta_rad = params_noise_generation["beta_rad"]
beta_tan = params_noise_generation["beta_tan"]

# проверка разрешимости интервалов активации при самом плотном случае (см. изменение №3)
if ntpoints // N_src_max + overlap_frac * dur_samp < dur_samp:
    raise ValueError(f"при N_src = {N_src_max} окна активации не помещаются в {ntpoints} отсчётов")

#%% Creating a graph with vertices, with distances between them

rr_lh = src[0]['rr'][src[0]['vertno']] # left hemisphere, select only used vertices
rr_rh = src[1]['rr'][src[1]['vertno']] # right hemisphere
vertices = np.concatenate((rr_lh, rr_rh),axis = 0)
N_vert = vertices.shape[0]
n_lh_vertices = len(rr_lh)
VertConn = spatial_src_adjacency(src)
# [N_vert x N_vert], sparse adjacency matrix, VertConn(i,j)=1 if i and j are neighbors

nn_lh = src[0]['nn'][src[0]['vertno']]
nn_rh = src[1]['nn'][src[1]['vertno']]
normals = np.concatenate((nn_lh, nn_rh), axis=0) # [N_vert x 3]

ii, jj = VertConn.nonzero() # extract only the elements below the main diagonal
mask = ii > jj
ii = ii[mask]
jj = jj[mask]
W = np.linalg.norm(vertices[ii, :] - vertices[jj, :], axis=1) * 1000.0  # mm
# np.mean(W) ~ 6.3 mm, np.min(W) ~ 1 mm, np.max(W) ~ 14 mm

Graph = nx.Graph() # Graph with the weights equals the distance between vertices = the length of the edges
for i, j, w in zip(ii, jj, W):
    Graph.add_edge(i, j, weight=w)

edge_index = np.vstack((ii, jj)) # [2 x N_edges], геометрия сохраняется вместе с датасетом

#%% Creating our raw template

# изменение №5. raw_template создаётся один раз, внутри цикла из него берётся только info
raw_template = mne.io.read_raw_fif(raw_fname, preload=False).copy()
raw_template.pick('mag') # 'mag', 'grad', 'all', etc.
raw_template.resample(Fs)

info = raw_template.info

#%% Generating of a simulation data set

iter = 300
base_seed = 0                     # изменение №6. seed каждого сэмпла = base_seed + k, сэмпл полностью воспроизводим
max_attempts = 20                 # изменение №7. сколько раз пересобирать сэмпл, если центроиды не удалось расставить

# изменение №6. массивы датасета выделяются заранее (полные G во float64 занимают много памяти),
# per-source поля хранятся padded-массивами шириной N_src_max: int-поля заполнены -1, float -- NaN
Ys = np.zeros((iter, N_sensors, ntpoints))                       # изменение №5. здесь лежат отфильтрованные данные
Gs_for_simulation = np.zeros((iter, N_sensors, N_vert))
n_src_all = np.zeros(iter, dtype=int)
seeds_all = np.zeros(iter, dtype=int)
centroids_idx_all = np.full((iter, N_src_max), -1, dtype=int)
R_mm_all = np.full((iter, N_src_max), np.nan)
band_idx_all = np.full((iter, N_src_max), -1, dtype=int)
t_start_all = np.full((iter, N_src_max), -1, dtype=int)
t_stop_all = np.full((iter, N_src_max), -1, dtype=int)
snr_all = np.full((iter, N_src_max), np.nan)
thetas = np.full((iter, N_src_max * 3), np.nan)                  # координаты центроидов, NaN-padded

for k in range(iter):

    seed = base_seed + k
    rng = np.random.default_rng(seed=seed)

    for attempt in range(max_attempts):
        try:
            N_src = int(rng.integers(N_src_min, N_src_max + 1))                # изменение №1. случайное число источников
            R_mm = trunc_dist.rvs(size=N_src, random_state=rng)                # изменение №2. случайные радиусы источников
            band_idx = rng.integers(0, len(freq_bands_hz), size=N_src)         # изменение №4. случайные частотные диапазоны (с повторами)
            t_start, t_stop = Sample_Activation_Windows(                       # изменение №3. случайные окна активации
                N_src, ntpoints, dur_samp, overlap_frac, rng=rng)

            params_spatial_activation = {**params_spatial_common,
                                         "N_src": N_src,
                                         "R_mm": R_mm,
                                         "A0": np.ones(N_src)}                 # amplitude coefficient

            params_temporal_activation = {**params_temporal_common,
                                          "N_src": N_src,
                                          "t_start_sources": t_start,
                                          "t_stop_sources": t_stop,
                                          "frec_source_band": freq_bands_hz[band_idx] / (Fs / 2.0)}

            centroids_idx, spatial_activation = Spatial_Activation(params_spatial_activation, Graph, rng=rng)
            break
        except RuntimeError:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"[{k}] не удалось расставить центроиды за {max_attempts} попыток")
            continue

    el_time_activation = Temporal_Activation(params_temporal_activation, rng=rng)
    spatio_temporal_activation = el_time_activation.T @ spatial_activation
    G_for_simulation, _, _, _ = Modeling_Liedfield_Error(G, beta_rad, beta_tan, rng=rng)
    Y_data_unnoise = G_for_simulation @ spatio_temporal_activation.T
    Noise = Generate_Brain_Noise(G_for_simulation, params_noise_generation, rng=rng)
    Y_data = Y_data_unnoise / (Y_data_unnoise.reshape(-1).std(ddof=1) + 1e-12) + noise_level * Noise

    # изменение №5. в датасет пишутся отфильтрованные 2-40 Гц данные, размерность совпадает с Y_data
    raw_sample = mne.io.RawArray(Y_data, info, verbose=False)
    raw_sample.filter(params_filter["l_freq"], params_filter["h_freq"],
                      method=params_filter["method"],
                      iir_params=dict(order=params_filter["order"],
                                      ftype=params_filter["ftype"], output='sos'),
                      verbose=False)
    Y_data_filt = raw_sample.get_data()

    # изменение №6. per-source SNR считается аналитически в собственном окне каждого источника:
    # вклад источника j в сенсоры это b_j * a_j(t), поэтому мощность считается без пересборки данных
    c = 1.0 / (Y_data_unnoise.reshape(-1).std(ddof=1) + 1e-12) # та же нормировка, что и у Y_data
    snr_per_source = np.zeros(N_src)
    for j in range(N_src):
        b_j = G_for_simulation @ spatial_activation[j] # [N_sensors]
        win = slice(t_start[j], t_stop[j])
        p_sig = c**2 * np.sum(b_j**2) * np.mean(el_time_activation[j, win]**2) / N_sensors
        p_noi = noise_level**2 * np.mean(Noise[:, win]**2)
        snr_per_source[j] = p_sig / (p_noi + 1e-12)

    # изменение №6. вместе с данными сохраняются все параметры, участвовавшие в их создании
    Ys[k] = Y_data_filt
    Gs_for_simulation[k] = G_for_simulation
    n_src_all[k] = N_src
    seeds_all[k] = seed
    centroids_idx_all[k, :N_src] = centroids_idx
    R_mm_all[k, :N_src] = R_mm
    band_idx_all[k, :N_src] = band_idx
    t_start_all[k, :N_src] = t_start
    t_stop_all[k, :N_src] = t_stop
    snr_all[k, :N_src] = snr_per_source
    thetas[k, :N_src * 3] = vertices[centroids_idx].reshape(-1)

    print(f"[{k}] N_src = {N_src}, R_mm = {np.round(R_mm, 1)}, SNR = {np.round(snr_per_source, 1)}")

#%% Saving

save_path = 'simulation_dataset_15_09_2026.npz'

size_gb = (Ys.nbytes + Gs_for_simulation.nbytes) / 1024**3
print(f"Ys: {Ys.nbytes / 1024**3:.2f} ГБ, G_for_simulation: {Gs_for_simulation.nbytes / 1024**3:.2f} ГБ, всего ~{size_gb:.2f} ГБ")

meta = {
    "description": "N_src, радиусы, окна активации и частотные диапазоны случайны от симуляции к симуляции",
    "Ys": "данные сенсоров, отфильтрованные 2-40 Гц (IIR butterworth, order=4)",
    "padding": "per-source поля имеют ширину N_src_max: int-поля дополнены -1, float-поля -- NaN",
    "snr_per_source": "аналитический SNR источника в его собственном окне активации (по нефильтрованным данным)",
    "seeds": "np.random.default_rng(seed=seeds[k]) полностью воспроизводит сэмпл k"
    }

np.savez(
    save_path,
    # per-sample
    Ys                       = Ys,
    G_for_simulation         = Gs_for_simulation,
    n_src                    = n_src_all,
    centroids_idx            = centroids_idx_all,
    R_mm                     = R_mm_all,
    band_idx                 = band_idx_all,
    t_start_samples          = t_start_all,
    t_stop_samples           = t_stop_all,
    snr_per_source           = snr_all,
    thetas                   = thetas,
    seeds                    = seeds_all,
    # общее для всех симуляций
    vertices                 = vertices,
    normals                  = normals,
    edge_index               = edge_index,
    edge_weight_mm           = W,
    n_lh_vertices            = n_lh_vertices,
    freq_bands_hz            = freq_bands_hz,
    N_src_min                = N_src_min,
    N_src_max                = N_src_max,
    Fs                       = Fs,
    params_spatial           = params_spatial_common,
    params_temporal          = params_temporal_common,
    params_noise             = params_noise_generation,
    params_filter            = params_filter,
    meta                     = meta
)

#%% Saving without full G (вариант на случай, если мы передумаем хранить гигабайты)

# G_for_simulation восстанавливается один-в-один из seed сэмпла, поэтому его можно не хранить:
# rng = np.random.default_rng(seed=seeds[k]) и повторить порядок вызовов из цикла генерации

# np.savez(
#     'simulation_dataset_15_09_2026_no_G.npz',
#     Ys                       = Ys,
#     n_src                    = n_src_all,
#     centroids_idx            = centroids_idx_all,
#     R_mm                     = R_mm_all,
#     band_idx                 = band_idx_all,
#     t_start_samples          = t_start_all,
#     t_stop_samples           = t_stop_all,
#     snr_per_source           = snr_all,
#     thetas                   = thetas,
#     seeds                    = seeds_all,
#     base_seed                = base_seed,
#     beta_rad                 = beta_rad,
#     beta_tan                 = beta_tan,
#     vertices                 = vertices,
#     normals                  = normals,
#     edge_index               = edge_index,
#     edge_weight_mm           = W,
#     n_lh_vertices            = n_lh_vertices,
#     freq_bands_hz            = freq_bands_hz,
#     N_src_min                = N_src_min,
#     N_src_max                = N_src_max,
#     Fs                       = Fs,
#     params_spatial           = params_spatial_common,
#     params_temporal          = params_temporal_common,
#     params_noise             = params_noise_generation,
#     params_filter            = params_filter,
#     meta                     = meta
# )
