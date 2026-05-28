"""
HOSVD + GPR  vs  POD + GPR  on (Re, mf) — single held-out test at (Re=15000, mf=0.12)
========================================================================================
Both methods are trained on the same 16 cases: the 4×4 sub-grid that excludes
the entire Re=15000 row AND the entire mf=0.12 column.

  Re  training axis : [11000, 13000, 17000, 19000]   (4 values)
  mf  training axis : [0.04,  0.08,  0.16,  0.20]   (4 values)

HOSVD path: arrange the 16 training cases on a (4×4) grid, run HOSVD on the
            resulting (4, 4, Nz, Nx, Nsp) tensor.  Two separate 1D GPRs predict
            the new factor rows at Re=15000 and mf=0.12; these are contracted
            with the Tucker core to reconstruct the field.

POD path  : SVD of the flattened (Nz·Nx·Nsp × 16) training matrix.  A single
            2D GPR in (Re, mf) space predicts POD coefficients at the test point.

Saves:
  plots/singular_values.png
  plots/gpr_coeffs_hosvd_re.png
  plots/gpr_coeffs_hosvd_mf.png
  plots/gpr_coeffs_pod.png
  plots/relative_error_per_feat.png
  plots/field_{species}.png
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path('/home/isacco/DATA/ZOU_GPR')
CASES_DIR = DATA_DIR / 'Datasets_Isacco'
PLOT_DIR  = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)

# ── column layout ─────────────────────────────────────────────────────────────
COLS = [
    'x','y','z','alphat',
    'C2H','C2H2','C2H3','C2H4','C2H5','C2H6','C3H7','C3H8',
    'CH','CH2','CH2CHO','CH2CO','CH2O','CH2OH','CH2S','CH3','CH3CHO','CH3O','CH3OH',
    'CH4','CO','CO2','epsilon','H','H2','H2O','H2O2','HCCO','HCCOH','HCO','HO2',
    'k','N2','O2','OH','p','T','Ux','Uy','Uz'
]
COL_IDX = {name: i for i, name in enumerate(COLS)}

RE_VALS = [11000, 13000, 15000, 17000, 19000]
MF_VALS = [0.04,  0.08,  0.12,  0.16,  0.20]

RE_TEST = 15000
MF_TEST = 0.12

# Training axes: exclude the test Re row and test mf column entirely
RE_TRAIN_VALS = [v for v in RE_VALS if v != RE_TEST]   # [11000, 13000, 17000, 19000]
MF_TRAIN_VALS = [v for v in MF_VALS if v != MF_TEST]   # [0.04, 0.08, 0.16, 0.20]

IMPORTANT_FIELDS = ['T', 'CH4', 'O2', 'CO2', 'H2O',]


# ── helpers ───────────────────────────────────────────────────────────────────
def load_case(path):
    raw    = np.loadtxt(path, skiprows=1)
    x_vals = np.unique(raw[:, COL_IDX['x']])
    z_vals = np.unique(raw[:, COL_IDX['z']])
    idx    = np.lexsort((raw[:, COL_IDX['x']], raw[:, COL_IDX['z']]))
    return raw[idx].reshape(len(z_vals), len(x_vals), len(COLS)), x_vals, z_vals


def rank_by_energy(singular_values, threshold=0.99):
    cumulative = np.cumsum(singular_values)
    r = np.searchsorted(cumulative, threshold * cumulative[-1]) + 1
    return min(r, len(singular_values))


def hosvd(tensor, energy_threshold=0.99):
    """HOSVD (Tucker via mode-wise SVD). Returns (core, factors, sv_list)."""
    factors, sv_list = [], []
    for mode in tqdm(range(tensor.ndim), desc='HOSVD modes'):
        U, S, _ = np.linalg.svd(tl.unfold(tensor, mode), full_matrices=False)
        r = rank_by_energy(S, energy_threshold)
        factors.append(U[:, :r])
        sv_list.append(S)
    core = tl.tenalg.multi_mode_dot(
        tensor, [f.T for f in factors], modes=list(range(tensor.ndim))
    )
    return core, factors, sv_list


def rel_error(pred, true):
    denom = np.linalg.norm(true.ravel())
    return np.linalg.norm((pred - true).ravel()) / denom if denom > 0 else np.nan


def minmax(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a, float)


def make_mo_gpr_2d():
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(2), length_scale_bounds='fixed', nu=2.5)
    base   = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    return MultiOutputRegressor(base)


def make_mo_gpr_1d():
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds='fixed', nu=2.5)
    base   = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    return MultiOutputRegressor(base)


# ── 1. load all 25 cases ──────────────────────────────────────────────────────
print('Loading cases...')
sample_grid, x_vals, z_vals = load_case(sorted(CASES_DIR.glob('*.xy'))[0])
Nz, Nx, Nsp  = sample_grid.shape
x_min, x_max = x_vals.min(), x_vals.max()
z_min, z_max = z_vals.min(), z_vals.max()

params_all      = np.array([[re, mf] for re in RE_VALS for mf in MF_VALS])  # (25, 2)
tensor_flat_all = np.empty((len(params_all), Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params_all):
    path = next(CASES_DIR.glob(f'*_mfH2_{mf:.2f}_Re_{int(re)}.xy'))
    tensor_flat_all[k], _, _ = load_case(path)

print(f'Full tensor shape: {tensor_flat_all.shape}')


# ── 2. train / test split — 4×4 grid, same 16 cases for both methods ─────────
# Training: only cases where Re ∈ RE_TRAIN_VALS AND mf ∈ MF_TRAIN_VALS (16 cases)
# Test: Re=15000, mf=0.12 (1 case)
def in_train(re, mf):
    return (re in RE_TRAIN_VALS) and (round(mf, 2) in [round(v, 2) for v in MF_TRAIN_VALS])

def is_test(re, mf):
    return (re == RE_TEST) and (round(mf, 2) == round(MF_TEST, 2))

train_mask = np.array([in_train(re, mf) for re, mf in params_all])
test_mask  = np.array([is_test(re, mf)  for re, mf in params_all])

params_train = params_all[train_mask]       # (16, 2)
T_train      = tensor_flat_all[train_mask]  # (16, Nz, Nx, Nsp)
T_test_true  = tensor_flat_all[test_mask][0]  # (Nz, Nx, Nsp)

print(f'Training cases: {len(params_train)}  (4×4 grid, Re∈{RE_TRAIN_VALS}, mf∈{MF_TRAIN_VALS})')
print(f'Test case: Re={RE_TEST}, mf={MF_TEST}')


# ── 3. scale on training statistics only ─────────────────────────────────────
eps = 1e-12
mu  = T_train.mean(axis=(0, 1, 2), keepdims=True)   # (1,1,1,Nsp)
std = T_train.std(axis=(0, 1, 2),  keepdims=True)
std = np.where(std < eps, 1.0, std)

T_train_s = (T_train - mu) / std   # (16, Nz, Nx, Nsp)


# ── 4. build (4, 4, Nz, Nx, Nsp) grid tensor for HOSVD ──────────────────────
n_re_tr = len(RE_TRAIN_VALS)
n_mf_tr = len(MF_TRAIN_VALS)
re_to_idx_tr = {v: i for i, v in enumerate(RE_TRAIN_VALS)}
mf_to_idx_tr = {round(v, 2): j for j, v in enumerate(MF_TRAIN_VALS)}

T_grid = np.zeros((n_re_tr, n_mf_tr, Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params_train):
    i = re_to_idx_tr[int(re)]
    j = mf_to_idx_tr[round(mf, 2)]
    T_grid[i, j] = T_train[k]

T_grid_s = (T_grid - mu) / std   # (4, 4, Nz, Nx, Nsp)
print(f'\nHOSVD grid tensor shape: {T_grid_s.shape}')


# ── 5. HOSVD on the 4×4 grid tensor ──────────────────────────────────────────
print('Running HOSVD on (4, 4, Nz, Nx, Nsp) grid tensor...')
core, factors, sv_list = hosvd(T_grid_s)
U_re, U_mf, U_z, U_x, U_spec = factors
r_re, r_mf = U_re.shape[1], U_mf.shape[1]
print(f'U_re: {U_re.shape}   U_mf: {U_mf.shape}   core: {core.shape}')


# ── 6. POD on the same 16 training cases ─────────────────────────────────────
print('\nRunning POD (SVD on flattened 16-case training matrix)...')
mat_tr           = T_train_s.reshape(len(params_train), -1).T   # (Nz*Nx*Nsp, 16)
V_pod, S_pod, _  = np.linalg.svd(mat_tr, full_matrices=False)
r_pod            = rank_by_energy(S_pod)
V_pod            = V_pod[:, :r_pod]
a_tr             = (V_pod.T @ mat_tr).T                         # (16, r_pod)
print(f'POD rank: {r_pod}')


# ── 7. singular value decay ───────────────────────────────────────────────────
mode_names = [f'Re ({n_re_tr})', f'mf ({n_mf_tr})', f'z ({Nz})', f'x ({Nx})', f'species ({Nsp})']
fig, axs = plt.subplots(1, 6, figsize=(20, 3))

for ax, sv, name, trunc in zip(axs[:5], sv_list, mode_names, [f.shape[1] for f in factors]):
    ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], 'o-', ms=3)
    ax.axvline(trunc, color='red', ls='--', label=f'r={trunc}')
    ax.set_title(f'HOSVD mode: {name}')
    ax.set_xlabel('index')
    ax.legend(fontsize=8)
    ax.grid(True)

axs[5].semilogy(np.arange(1, len(S_pod) + 1), S_pod / S_pod[0], 's-', ms=3, color='darkorange')
axs[5].axvline(r_pod, color='red', ls='--', label=f'r={r_pod}')
axs[5].set_title('POD (16-case matrix)')
axs[5].set_xlabel('index')
axs[5].legend(fontsize=8)
axs[5].grid(True)

fig.suptitle('Singular value decay (normalised to σ₁)', y=1.02)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'singular_values.png', dpi=150, bbox_inches='tight')
print('Saved plots/singular_values.png')


# ── 8. HOSVD GPR: two 1D GPRs on U_re and U_mf rows ─────────────────────────
# Normalised training axes (all 4 points used for training, predict at new value)
Re_tr_nm   = minmax(np.array(RE_TRAIN_VALS, float))          # (4,) in [0,1]
Mf_tr_nm   = minmax(np.array(MF_TRAIN_VALS, float))          # (4,) in [0,1]

# Normalise test point onto the same scale as training axes
Re_test_nm = (RE_TEST - min(RE_TRAIN_VALS)) / (max(RE_TRAIN_VALS) - min(RE_TRAIN_VALS))
Mf_test_nm = (MF_TEST - min(MF_TRAIN_VALS)) / (max(MF_TRAIN_VALS) - min(MF_TRAIN_VALS))

print(f'\nHOSVD 1D GPR — Re: training on {Re_tr_nm}, predicting at {Re_test_nm:.3f}')
print(f'HOSVD 1D GPR — mf: training on {Mf_tr_nm}, predicting at {Mf_test_nm:.3f}')

# GPR_Re: all 4 training Re rows → predict at RE_TEST
print('Fitting HOSVD GPR for Re axis (1D)...')
gpr_re_h = make_mo_gpr_1d()
gpr_re_h.fit(Re_tr_nm.reshape(-1, 1), U_re)   # U_re is (4, r_re)
preds_re_h    = [est.predict([[Re_test_nm]], return_std=True)
                 for est in gpr_re_h.estimators_]
alpha_Re_pred = np.array([p[0].item() for p in preds_re_h])   # (r_re,)
alpha_Re_std  = np.array([p[1].item() for p in preds_re_h])

# GPR_mf: all 4 training mf rows → predict at MF_TEST
print('Fitting HOSVD GPR for mf axis (1D)...')
gpr_mf_h = make_mo_gpr_1d()
gpr_mf_h.fit(Mf_tr_nm.reshape(-1, 1), U_mf)   # U_mf is (4, r_mf)
preds_mf_h    = [est.predict([[Mf_test_nm]], return_std=True)
                 for est in gpr_mf_h.estimators_]
alpha_mf_pred = np.array([p[0].item() for p in preds_mf_h])   # (r_mf,)
alpha_mf_std  = np.array([p[1].item() for p in preds_mf_h])


# ── 9. POD GPR: 2D (Re, mf) → POD coefficients ───────────────────────────────
param_scaler = StandardScaler().fit(params_train)
P_train_s    = param_scaler.transform(params_train)
P_test_s     = param_scaler.transform([[RE_TEST, MF_TEST]])

print('\nFitting POD GPR (2D, Re × mf)...')
mo_gpr_pod = make_mo_gpr_2d()
mo_gpr_pod.fit(P_train_s, a_tr)

preds_pod = [est.predict(P_test_s, return_std=True) for est in mo_gpr_pod.estimators_]
a_pred    = np.array([p[0].item() for p in preds_pod])   # (r_pod,)
a_std     = np.array([p[1].item() for p in preds_pod])


# ── 10. true coefficients (for plotting only — not used in reconstruction) ────
# POD: project test case onto spatial modes
T_test_s = (T_test_true - mu.squeeze()) / std.squeeze()
a_true   = (V_pod.T @ T_test_s.ravel()).ravel()   # (r_pod,)

# Note: there are no "true" U_re / U_mf rows for the test point since it was
# never in the HOSVD training tensor.  We instead compare reconstruction quality.


# ── 11. HOSVD coefficient plots ───────────────────────────────────────────────
comp_re = np.arange(r_re)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(comp_re, alpha_Re_pred, 'r--s', ms=5, lw=1.2, label='1D GPR prediction')
ax.fill_between(comp_re, alpha_Re_pred - alpha_Re_std, alpha_Re_pred + alpha_Re_std,
                alpha=0.25, color='red', label='±1 σ')
ax.set_xlabel('Re component index')
ax.set_ylabel('$U_{re}$ coefficient')
ax.set_title(f'HOSVD 1D GPR (Re axis) — extrapolated to Re={RE_TEST}  |  r_re={r_re}')
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'gpr_coeffs_hosvd_re.png', dpi=150, bbox_inches='tight')
print('Saved plots/gpr_coeffs_hosvd_re.png')

comp_mf = np.arange(r_mf)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(comp_mf, alpha_mf_pred, 'r--s', ms=5, lw=1.2, label='1D GPR prediction')
ax.fill_between(comp_mf, alpha_mf_pred - alpha_mf_std, alpha_mf_pred + alpha_mf_std,
                alpha=0.25, color='red', label='±1 σ')
ax.set_xlabel('mf component index')
ax.set_ylabel('$U_{mf}$ coefficient')
ax.set_title(f'HOSVD 1D GPR (mf axis) — extrapolated to mf={MF_TEST}  |  r_mf={r_mf}')
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'gpr_coeffs_hosvd_mf.png', dpi=150, bbox_inches='tight')
print('Saved plots/gpr_coeffs_hosvd_mf.png')


# ── 12. POD coefficient plot ──────────────────────────────────────────────────
comp_pod = np.arange(r_pod)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(comp_pod, a_true, 'k-o',  ms=5, lw=1.2, label='True (projection onto POD modes)')
ax.plot(comp_pod, a_pred, 'r--s', ms=5, lw=1.2, label='2D GPR prediction')
ax.fill_between(comp_pod, a_pred - a_std, a_pred + a_std,
                alpha=0.25, color='red', label='±1 σ')
ax.set_xlabel('POD mode index')
ax.set_ylabel('POD coefficient')
ax.set_title(f'POD 2D GPR — predicted vs true coefficients at (Re={RE_TEST}, mf={MF_TEST})\n'
             f'trained on {len(params_train)} conditions (4×4 grid)')
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'gpr_coeffs_pod.png', dpi=150, bbox_inches='tight')
print('Saved plots/gpr_coeffs_pod.png')


# ── 13. reconstruct fields ────────────────────────────────────────────────────
# HOSVD: contract modes 0 and 1 of the Tucker core with predicted factor rows
Z_hosvd   = tl.tenalg.mode_dot(core, alpha_Re_pred, mode=0)    # (r_mf, r_z, r_x, r_sp)
Z_hosvd   = tl.tenalg.mode_dot(Z_hosvd, alpha_mf_pred, mode=0) # (r_z, r_x, r_sp)
recon_s_h = tl.tenalg.multi_mode_dot(Z_hosvd, [U_z, U_x, U_spec], modes=[0, 1, 2])
recon_h   = recon_s_h * std.squeeze() + mu.squeeze()            # (Nz, Nx, Nsp) physical

# POD: 2D GPR predicted coefficients → field
recon_s_p = (V_pod @ a_pred).reshape(Nz, Nx, Nsp)
recon_p   = recon_s_p * std.squeeze() + mu.squeeze()            # (Nz, Nx, Nsp) physical


# ── 14. per-feature relative error bar chart ──────────────────────────────────
errors_h = [rel_error(recon_h[:, :, COL_IDX[n]], T_test_true[:, :, COL_IDX[n]])
            for n in IMPORTANT_FIELDS]
errors_p = [rel_error(recon_p[:, :, COL_IDX[n]], T_test_true[:, :, COL_IDX[n]])
            for n in IMPORTANT_FIELDS]

print(f"\nMean relative error — HOSVD: {np.mean(errors_h):.4f}   POD: {np.mean(errors_p):.4f}")

x = np.arange(len(IMPORTANT_FIELDS))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w/2, errors_h, w, color='#2196F3', alpha=0.85,
       label=f'HOSVD 4×4 + 2×1D GPR  (r_re={r_re}, r_mf={r_mf}, mean={np.mean(errors_h):.4f})')
ax.bar(x + w/2, errors_p, w, color='#FF5722', alpha=0.85,
       label=f'POD + 2D GPR  (r_pod={r_pod}, mean={np.mean(errors_p):.4f})')
ax.set_xticks(x)
ax.set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right')
ax.set_ylabel('Relative L2 error')
ax.set_title(f'Per-feature relative error — (Re={RE_TEST}, mf={MF_TEST})\n'
             f'Both methods trained on same 16 cases (4×4 grid)')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'relative_error_per_feat.png', dpi=150, bbox_inches='tight')
print('Saved plots/relative_error_per_feat.png')


# ── 15. per-feature spatial figures ──────────────────────────────────────────
extent       = [x_min, x_max, z_min, z_max]
suptitle_sub = f'Re={RE_TEST}, mf={MF_TEST}  |  trained on {len(params_train)} conditions (4×4 grid)'

for name in IMPORTANT_FIELDS:
    sp = COL_IDX[name]

    F_true    = T_test_true[:, :, sp]
    F_hosvd   = recon_h    [:, :, sp]
    F_pod     = recon_p    [:, :, sp]
    err_h_fld = F_hosvd - F_true
    err_p_fld = F_pod   - F_true

    vmin_f  = min(F_true.min(), F_hosvd.min(), F_pod.min())
    vmax_f  = max(F_true.max(), F_hosvd.max(), F_pod.max())
    err_lim = max(np.abs(err_h_fld).max(), np.abs(err_p_fld).max())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=120)

    for ax, title, field in zip(axes[0],
                                 ['Original',
                                  'HOSVD 4×4 + 2×1D GPR',
                                  'POD + 2D GPR'],
                                 [F_true, F_hosvd, F_pod]):
        im = ax.imshow(field, origin='lower', aspect='auto', extent=extent,
                       vmin=vmin_f, vmax=vmax_f, cmap='hot')
        ax.set_title(title)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        plt.colorbar(im, ax=ax, label=name)

    axes[1, 0].hist(err_h_fld.ravel(), bins=80, alpha=0.7, color='#2196F3',
                    density=True, label='HOSVD')
    axes[1, 0].hist(err_p_fld.ravel(), bins=80, alpha=0.7, color='#FF5722',
                    density=True, label='POD')
    axes[1, 0].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    axes[1, 0].set_xlabel(f'Δ{name}')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error distributions')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)

    im_eh = axes[1, 1].imshow(err_h_fld, origin='lower', aspect='auto', extent=extent,
                               vmin=-err_lim, vmax=err_lim, cmap='RdBu_r')
    axes[1, 1].set_title('Spatial error — HOSVD')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('z')
    plt.colorbar(im_eh, ax=axes[1, 1], label=f'Δ{name}')

    im_ep = axes[1, 2].imshow(err_p_fld, origin='lower', aspect='auto', extent=extent,
                               vmin=-err_lim, vmax=err_lim, cmap='RdBu_r')
    axes[1, 2].set_title('Spatial error — POD')
    axes[1, 2].set_xlabel('r')
    axes[1, 2].set_ylabel('z')
    plt.colorbar(im_ep, ax=axes[1, 2], label=f'Δ{name}')

    fig.suptitle(f'{name} — {suptitle_sub}', fontsize=11)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f'field_{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plots/field_{name}.png')

#plt.show()