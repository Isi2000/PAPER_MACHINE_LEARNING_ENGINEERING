"""
HOSVD + GPR on (Re, mf) jointly — leave-one-out at (Re=15000, mf=0.12)
=======================================================================
Flatten the Re × mf grid into a single parameter axis of 25 conditions.
Leave out one simulation, fit HOSVD on the remaining 24, train a 2D GPR
that maps (Re, mf) → U_param row, and reconstruct the held-out field.

Saves:
  plots/singular_values.png         — decay per tensor mode
  plots/gpr_coeffs.png              — predicted vs true U_param coefficients
  plots/relative_error_per_feat.png — relative L2 error per species
  plots/field_{species}.png         — original | recon | error field | histogram
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

IMPORTANT_FIELDS = ['T', 'OH', 'CO2', 'CH4', 'H2O', 'CO']


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


# ── 1. load all 25 cases into a flat (25, Nz, Nx, Nsp) tensor ────────────────
print('Loading cases...')
sample_grid, x_vals, z_vals = load_case(sorted(CASES_DIR.glob('*.xy'))[0])
Nz, Nx, Nsp  = sample_grid.shape
x_min, x_max = x_vals.min(), x_vals.max()
z_min, z_max = z_vals.min(), z_vals.max()

# params[k] = (Re, mf) for case k, ordered Re-major
params      = np.array([[re, mf] for re in RE_VALS for mf in MF_VALS])  # (25, 2)
tensor_flat = np.empty((len(params), Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params):
    path = next(CASES_DIR.glob(f'*_mfH2_{mf:.2f}_Re_{int(re)}.xy'))
    tensor_flat[k], _, _ = load_case(path)

print(f'Tensor shape: {tensor_flat.shape}')


# ── 2. train / test split ─────────────────────────────────────────────────────
test_mask  = (params[:, 0] == RE_TEST) & (params[:, 1] == MF_TEST)
test_idx   = int(np.where(test_mask)[0])
train_mask = ~test_mask

params_train = params[train_mask]       # (24, 2)
T_train      = tensor_flat[train_mask]  # (24, Nz, Nx, Nsp)  physical units
T_test_true  = tensor_flat[test_idx]    # (Nz, Nx, Nsp)      physical units


# ── 3. scale on training statistics only ─────────────────────────────────────
eps = 1e-12
mu  = T_train.mean(axis=(0, 1, 2), keepdims=True)   # (1,1,1,Nsp)
std = T_train.std(axis=(0, 1, 2),  keepdims=True)
std = np.where(std < eps, 1.0, std)

T_train_s = (T_train - mu) / std   # (24, Nz, Nx, Nsp)


# ── 4. HOSVD on the 4D training tensor ───────────────────────────────────────
# modes: 0=param(24), 1=z(101), 2=x(84), 3=species(44)
core, factors, sv_list = hosvd(T_train_s)
U_param, U_z, U_x, U_spec = factors
r_p = U_param.shape[1]
print(f'U_param shape: {U_param.shape}  (one row per training condition)')


# ── 5. singular value decay ───────────────────────────────────────────────────
mode_names = ['param (24)', 'z (101)', 'x (84)', 'species (44)']
fig, axs = plt.subplots(1, 4, figsize=(14, 3))
for ax, sv, name, trunc in zip(axs, sv_list, mode_names, [f.shape[1] for f in factors]):
    ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], 'o-', ms=3)
    ax.axvline(trunc, color='red', ls='--', label=f'r={trunc}')
    ax.set_title(f'mode: {name}')
    ax.set_xlabel('index')
    ax.legend(fontsize=8)
    ax.grid(True)
fig.suptitle('Singular value decay per mode (normalised to σ₁)', y=1.02)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'singular_values.png', dpi=150, bbox_inches='tight')
print('Saved plots/singular_values.png')


# ── 6. GPR: 2D (Re, mf) → U_param row ───────────────────────────────────────
# Standardise inputs so Re and mf have comparable scales for the kernel.
# length_scale is a 2-vector so the kernel learns separate scales for Re and mf.
param_scaler = StandardScaler().fit(params_train)
P_train_s    = param_scaler.transform(params_train)          # (24, 2)
P_test_s     = param_scaler.transform([[RE_TEST, MF_TEST]])  # (1, 2)

kernel   = ConstantKernel(1.0) * Matern(length_scale=np.ones(2), nu=2.5)
base_gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

print('Fitting GPR...')
mo_gpr = MultiOutputRegressor(base_gpr)
mo_gpr.fit(P_train_s, U_param)

preds      = [est.predict(P_test_s, return_std=True) for est in mo_gpr.estimators_]
alpha_pred = np.array([p[0].item() for p in preds])   # (r_p,)
alpha_std  = np.array([p[1].item() for p in preds])   # (r_p,)


# ── 7. true U_param via projection of the test case onto the training basis ───
# The Tucker model: T_train_s[k] ≈ Σ_p U_param[k,p] * core[p] ×_z U_z ×_x U_x ×_s U_spec
# For a new scaled case x*, contract it with the spatial/species factors:
#   G_test = x* ×_z U_z.T ×_x U_x.T ×_s U_spec.T     shape (r_z, r_x, r_s)
# then solve  core.reshape(r_p, -1).T @ alpha = G_test.ravel()  (least squares)

T_test_s = (T_test_true - mu.squeeze()) / std.squeeze()
G_test   = tl.tenalg.multi_mode_dot(T_test_s, [U_z.T, U_x.T, U_spec.T], modes=[0, 1, 2])
alpha_true, _, _, _ = np.linalg.lstsq(
    core.reshape(r_p, -1).T, G_test.ravel(), rcond=None
)


# ── 8. plot predicted vs true U_param coefficients ───────────────────────────
comp = np.arange(r_p)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(comp, alpha_true, 'k-o',  ms=5, lw=1.2, label='True (projection onto training basis)')
ax.plot(comp, alpha_pred, 'r--s', ms=5, lw=1.2, label='GPR prediction')
ax.fill_between(comp, alpha_pred - alpha_std, alpha_pred + alpha_std,
                alpha=0.25, color='red', label='±1 σ')
ax.set_xlabel('Component index')
ax.set_ylabel('$U_{param}$ coefficient')
ax.set_title(f'GPR-predicted vs true $U_{{param}}$ at (Re={RE_TEST}, mf={MF_TEST})\n'
             f'trained on {len(params_train)} conditions')
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'gpr_coeffs.png', dpi=150, bbox_inches='tight')
print('Saved plots/gpr_coeffs.png')


# ── 9. reconstruct field ──────────────────────────────────────────────────────
# Contract core along mode 0 with alpha_pred (vector, reduces dim by 1),
# then expand remaining modes with the spatial and species factors.
Z       = tl.tenalg.mode_dot(core, alpha_pred, mode=0)               # (r_z, r_x, r_s)
recon_s = tl.tenalg.multi_mode_dot(Z, [U_z, U_x, U_spec], modes=[0, 1, 2])  # (Nz, Nx, Nsp)
recon   = recon_s * std.squeeze() + mu.squeeze()                      # physical units


# ── 10. per-feature relative error bar chart ──────────────────────────────────
errors = [rel_error(recon[:, :, COL_IDX[n]], T_test_true[:, :, COL_IDX[n]])
          for n in IMPORTANT_FIELDS]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(IMPORTANT_FIELDS)), errors, color='steelblue')
ax.set_xticks(range(len(IMPORTANT_FIELDS)))
ax.set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right')
ax.set_ylabel('Relative L2 error')
ax.set_title(f'Per-feature relative error — (Re={RE_TEST}, mf={MF_TEST})')
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'relative_error_per_feat.png', dpi=150, bbox_inches='tight')
print('Saved plots/relative_error_per_feat.png')


# ── 11. per-feature spatial figures ───────────────────────────────────────────
extent       = [x_min, x_max, z_min, z_max]
suptitle_sub = f'Re={RE_TEST}, mf={MF_TEST}  |  trained on {len(params_train)} conditions'

for name in IMPORTANT_FIELDS:
    sp = COL_IDX[name]

    F_true  = T_test_true[:, :, sp]
    F_recon = recon      [:, :, sp]
    err     = F_recon - F_true

    vmin_f  = min(F_true.min(), F_recon.min())
    vmax_f  = max(F_true.max(), F_recon.max())
    err_lim = np.abs(err).max()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=120)

    for ax, title, field in zip(axes[0].flat,
                                ['Original', 'HOSVD 4D + 2D GPR'],
                                [F_true, F_recon]):
        im = ax.imshow(field, origin='lower', aspect='auto', extent=extent,
                       vmin=vmin_f, vmax=vmax_f, cmap='hot')
        ax.set_title(title)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        plt.colorbar(im, ax=ax, label=name)

    axes[1, 0].hist(err.ravel(), bins=80, alpha=0.7, color='steelblue', density=True)
    axes[1, 0].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    axes[1, 0].set_xlabel(f'Δ{name}')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error distribution')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)

    im_err = axes[1, 1].imshow(err, origin='lower', aspect='auto', extent=extent,
                                vmin=-err_lim, vmax=err_lim, cmap='RdBu_r')
    axes[1, 1].set_title('Spatial error')
    axes[1, 1].set_xlabel('r')
    axes[1, 1].set_ylabel('z')
    plt.colorbar(im_err, ax=axes[1, 1], label=f'Δ{name}')

    fig.suptitle(f'{name} — {suptitle_sub}', fontsize=11)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f'field_{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plots/field_{name}.png')

plt.show()
