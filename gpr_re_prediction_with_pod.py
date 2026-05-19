"""
HOSVD + GPR  vs  POD + GPR  on (Re, mf) — single held-out test at (Re=15000, mf=0.12)
========================================================================================
Leave out one simulation, fit HOSVD **and** POD on the remaining 24, then predict
the held-out field via GPR.

HOSVD path: arrange the 24 training cases on a regular (5 × 5) (Re, mf) grid (filling
            the missing test entry by additive interpolation), then run HOSVD on the
            resulting (5, 5, Nz, Nx, Nsp) tensor.  Modes 0 and 1 of the decomposition
            give U_re (5 × r_re) and U_mf (5 × r_mf) directly.  Two **separate 1D
            GPRs** — one per axis — predict the factor rows at the test condition; the
            predictions are contracted with the Tucker core to reconstruct the field.

POD path  : SVD of the flattened (Nz·Nx·Nsp × 24) training matrix gives spatial modes
            V_pod.  A single **2D GPR** in (Re, mf) parameter space then predicts the
            POD coefficients a_pred for the test condition.

Saves:
  plots/singular_values.png            — HOSVD mode decay (5 modes) + POD sv decay
  plots/gpr_coeffs_hosvd_re.png        — predicted vs true U_re row at Re_TEST
  plots/gpr_coeffs_hosvd_mf.png        — predicted vs true U_mf row at MF_TEST
  plots/gpr_coeffs_pod.png             — predicted vs true POD coefficients (2D GPR)
  plots/relative_error_per_feat.png    — HOSVD vs POD relative L2 error per species
  plots/field_{species}.png            — original | HOSVD recon | POD recon | errors
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
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

IMPORTANT_FIELDS = ['T', 'CO2', 'CH4', 'H2O', 'CO', 'O2', 'H2']


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
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(2), length_scale_bounds=(1e-10, 1e5), nu=2.5)
    base   = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    return MultiOutputRegressor(base)


def make_mo_gpr_1d():
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-10, 1e5), nu=2.5)
    base   = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    return MultiOutputRegressor(base)


# ── 1. load all 25 cases ──────────────────────────────────────────────────────
print('Loading cases...')
sample_grid, x_vals, z_vals = load_case(sorted(CASES_DIR.glob('*.xy'))[0])
Nz, Nx, Nsp  = sample_grid.shape
x_min, x_max = x_vals.min(), x_vals.max()
z_min, z_max = z_vals.min(), z_vals.max()

params      = np.array([[re, mf] for re in RE_VALS for mf in MF_VALS])  # (25, 2)
tensor_flat = np.empty((len(params), Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params):
    path = next(CASES_DIR.glob(f'*_mfH2_{mf:.2f}_Re_{int(re)}.xy'))
    tensor_flat[k], _, _ = load_case(path)

print(f'Tensor shape: {tensor_flat.shape}')


# ── 2. train / test split ─────────────────────────────────────────────────────
test_mask  = (params[:, 0] == RE_TEST) & (params[:, 1] == MF_TEST)
test_idx   = int(np.where(test_mask)[0][0])
train_mask = ~test_mask

params_train = params[train_mask]       # (24, 2)
T_train      = tensor_flat[train_mask]  # (24, Nz, Nx, Nsp)  physical units
T_test_true  = tensor_flat[test_idx]    # (Nz, Nx, Nsp)      physical units


# ── 2b. grid helpers (shared) ─────────────────────────────────────────────────
n_re = len(RE_VALS)
n_mf = len(MF_VALS)
re_to_idx = {v: i for i, v in enumerate(RE_VALS)}
mf_to_idx = {round(v, 2): j for j, v in enumerate(MF_VALS)}

i_miss = re_to_idx[RE_TEST]
j_miss = mf_to_idx[MF_TEST]

has_val = np.zeros((n_re, n_mf), dtype=bool)
for re, mf in params_train:
    has_val[re_to_idx[int(re)], mf_to_idx[round(mf, 2)]] = True

# minmax-normalised axes for 1D GPRs
Re_nm         = minmax(np.array(RE_VALS, float))          # (5,)
Mf_nm         = minmax(np.array(MF_VALS, float))          # (5,)
re_train_mask = np.array([v != RE_TEST for v in RE_VALS])
mf_train_mask = np.array([v != MF_TEST for v in MF_VALS])


# ── 3. scale on training statistics only ─────────────────────────────────────
eps = 1e-12
mu  = T_train.mean(axis=(0, 1, 2), keepdims=True)   # (1,1,1,Nsp)
std = T_train.std(axis=(0, 1, 2),  keepdims=True)
std = np.where(std < eps, 1.0, std)

T_train_s = (T_train - mu) / std   # (24, Nz, Nx, Nsp) — used by POD


# ── 4. build (5, 5, Nz, Nx, Nsp) grid tensor for HOSVD ──────────────────────
T_grid = np.zeros((n_re, n_mf, Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params_train):
    i, j = re_to_idx[int(re)], mf_to_idx[round(mf, 2)]
    T_grid[i, j] = T_train[k]

# fill missing test entry by additive interpolation (physical space)
T_grid[i_miss, j_miss] = (
    T_grid[:, j_miss][has_val[:, j_miss]].mean(axis=0) +
    T_grid[i_miss, :][has_val[i_miss, :]].mean(axis=0) -
    T_grid[has_val].mean(axis=0)
)

T_grid_s = (T_grid - mu) / std   # (5, 5, Nz, Nx, Nsp)


# ── 5. HOSVD on the 5D grid tensor ───────────────────────────────────────────
# modes: 0=Re(5), 1=mf(5), 2=z(Nz), 3=x(Nx), 4=species(Nsp)
print('Running HOSVD on (5, 5, Nz, Nx, Nsp) grid tensor...')
core, factors, sv_list = hosvd(T_grid_s)
U_re, U_mf, U_z, U_x, U_spec = factors
r_re, r_mf = U_re.shape[1], U_mf.shape[1]
print(f'U_re: {U_re.shape}   U_mf: {U_mf.shape}   core: {core.shape}')


# ── 6. POD on the flattened training matrix ───────────────────────────────────
print('Running POD (SVD on flattened training matrix)...')
mat_tr           = T_train_s.reshape(len(params_train), -1).T   # (Nz*Nx*Nsp, 24)
V_pod, S_pod, _  = np.linalg.svd(mat_tr, full_matrices=False)
r_pod            = rank_by_energy(S_pod)
V_pod            = V_pod[:, :r_pod]
a_tr             = (V_pod.T @ mat_tr).T                         # (24, r_pod)
print(f'POD rank: {r_pod}')


# ── 7. singular value decay ───────────────────────────────────────────────────
mode_names = [f'Re ({n_re})', f'mf ({n_mf})', f'z ({Nz})', f'x ({Nx})', f'species ({Nsp})']
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
axs[5].set_title('POD (flattened matrix)')
axs[5].set_xlabel('index')
axs[5].legend(fontsize=8)
axs[5].grid(True)

fig.suptitle('Singular value decay (normalised to σ₁)', y=1.02)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'singular_values.png', dpi=150, bbox_inches='tight')
print('Saved plots/singular_values.png')


# ── 8. HOSVD GPR: two 1D GPRs on U_re and U_mf rows ─────────────────────────
# 1D GPR_Re: 4 training Re values → predict row of U_re at RE_TEST
print('Fitting HOSVD GPR for Re axis (1D)...')
gpr_re_h = make_mo_gpr_1d()
gpr_re_h.fit(Re_nm[re_train_mask].reshape(-1, 1), U_re[re_train_mask])
preds_re_h    = [est.predict(Re_nm[~re_train_mask].reshape(-1, 1), return_std=True)
                 for est in gpr_re_h.estimators_]
alpha_Re_pred = np.array([p[0].item() for p in preds_re_h])   # (r_re,)
alpha_Re_std  = np.array([p[1].item() for p in preds_re_h])

# 1D GPR_mf: 4 training mf values → predict row of U_mf at MF_TEST
print('Fitting HOSVD GPR for mf axis (1D)...')
gpr_mf_h = make_mo_gpr_1d()
gpr_mf_h.fit(Mf_nm[mf_train_mask].reshape(-1, 1), U_mf[mf_train_mask])
preds_mf_h    = [est.predict(Mf_nm[~mf_train_mask].reshape(-1, 1), return_std=True)
                 for est in gpr_mf_h.estimators_]
alpha_mf_pred = np.array([p[0].item() for p in preds_mf_h])   # (r_mf,)
alpha_mf_std  = np.array([p[1].item() for p in preds_mf_h])


# ── 9. POD GPR: 2D (Re, mf) → POD coefficients ───────────────────────────────
param_scaler = StandardScaler().fit(params_train)
P_train_s    = param_scaler.transform(params_train)
P_test_s     = param_scaler.transform([[RE_TEST, MF_TEST]])

print('Fitting POD GPR (2D, Re × mf)...')
mo_gpr_pod = make_mo_gpr_2d()
mo_gpr_pod.fit(P_train_s, a_tr)

preds_pod = [est.predict(P_test_s, return_std=True) for est in mo_gpr_pod.estimators_]
a_pred    = np.array([p[0].item() for p in preds_pod])   # (r_pod,)
a_std     = np.array([p[1].item() for p in preds_pod])


# ── 10. true coefficients ─────────────────────────────────────────────────────
# HOSVD: true factor rows at the test grid position
alpha_Re_true = U_re[i_miss]   # (r_re,)
alpha_mf_true = U_mf[j_miss]   # (r_mf,)

# POD: project test case onto spatial modes
T_test_s = (T_test_true - mu.squeeze()) / std.squeeze()
a_true   = (V_pod.T @ T_test_s.ravel()).ravel()   # (r_pod,)


# ── 11a. HOSVD coefficient plot — Re axis ─────────────────────────────────────
comp_re = np.arange(r_re)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(comp_re, alpha_Re_true, 'k-o',  ms=5, lw=1.2, label='True $U_{re}$ row')
ax.plot(comp_re, alpha_Re_pred, 'r--s', ms=5, lw=1.2, label='1D GPR prediction')
ax.fill_between(comp_re, alpha_Re_pred - alpha_Re_std, alpha_Re_pred + alpha_Re_std,
                alpha=0.25, color='red', label='±1 σ')
ax.set_xlabel('Re component index')
ax.set_ylabel('$U_{re}$ coefficient')
ax.set_title(f'HOSVD 1D GPR (Re axis) at Re={RE_TEST}  |  r_re={r_re}')
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'gpr_coeffs_hosvd_re.png', dpi=150, bbox_inches='tight')
print('Saved plots/gpr_coeffs_hosvd_re.png')


# ── 11b. HOSVD coefficient plot — mf axis ─────────────────────────────────────
comp_mf = np.arange(r_mf)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(comp_mf, alpha_mf_true, 'k-o',  ms=5, lw=1.2, label='True $U_{mf}$ row')
ax.plot(comp_mf, alpha_mf_pred, 'r--s', ms=5, lw=1.2, label='1D GPR prediction')
ax.fill_between(comp_mf, alpha_mf_pred - alpha_mf_std, alpha_mf_pred + alpha_mf_std,
                alpha=0.25, color='red', label='±1 σ')
ax.set_xlabel('mf component index')
ax.set_ylabel('$U_{mf}$ coefficient')
ax.set_title(f'HOSVD 1D GPR (mf axis) at mf={MF_TEST}  |  r_mf={r_mf}')
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'gpr_coeffs_hosvd_mf.png', dpi=150, bbox_inches='tight')
print('Saved plots/gpr_coeffs_hosvd_mf.png')


# ── 12. POD coefficient plot (2D GPR) ─────────────────────────────────────────
comp_pod = np.arange(r_pod)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(comp_pod, a_true, 'k-o',  ms=5, lw=1.2, label='True (projection onto POD modes)')
ax.plot(comp_pod, a_pred, 'r--s', ms=5, lw=1.2, label='2D GPR prediction')
ax.fill_between(comp_pod, a_pred - a_std, a_pred + a_std,
                alpha=0.25, color='red', label='±1 σ')
ax.set_xlabel('POD mode index')
ax.set_ylabel('POD coefficient')
ax.set_title(f'POD 2D GPR-predicted vs true coefficients at (Re={RE_TEST}, mf={MF_TEST})\n'
             f'trained on {len(params_train)} conditions')
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'gpr_coeffs_pod.png', dpi=150, bbox_inches='tight')
print('Saved plots/gpr_coeffs_pod.png')


# ── 13. reconstruct fields ────────────────────────────────────────────────────
# HOSVD: contract modes 0 and 1 of the Tucker core with predicted factor rows,
#        then expand with spatial/species factors.
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
       label=f'HOSVD 5D + 2×1D GPR  (r_re={r_re}, r_mf={r_mf}, mean={np.mean(errors_h):.4f})')
ax.bar(x + w/2, errors_p, w, color='#FF5722', alpha=0.85,
       label=f'POD + 2D GPR  (r_pod={r_pod}, mean={np.mean(errors_p):.4f})')
ax.set_xticks(x)
ax.set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right')
ax.set_ylabel('Relative L2 error')
ax.set_title(f'Per-feature relative error — (Re={RE_TEST}, mf={MF_TEST})')
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'relative_error_per_feat.png', dpi=150, bbox_inches='tight')
print('Saved plots/relative_error_per_feat.png')


# ── 15. per-feature spatial figures ──────────────────────────────────────────
extent       = [x_min, x_max, z_min, z_max]
suptitle_sub = f'Re={RE_TEST}, mf={MF_TEST}  |  trained on {len(params_train)} conditions'

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
                                  'HOSVD 5D + 2×1D GPR',
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

plt.show()
