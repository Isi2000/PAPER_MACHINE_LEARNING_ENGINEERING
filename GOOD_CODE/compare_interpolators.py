"""
compare_interpolators.py
========================
GPR vs deterministic interpolation for HOSVD 1D and POD 2D surrogates.

  HOSVD 1D: GPR | Cubic Spline | Linear
  POD 2D:   GPR | RBF thin-plate | RBF multiquadric | Bilinear | Nearest

Training: Re in {11000,15000,17000,19000} x mf in {0.04,0.08,0.16,0.20}
Test Re:  13000 across all mf values
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import (CubicSpline, interp1d,
                                RBFInterpolator, RegularGridInterpolator)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)

from utils import (
    COLS, COL_IDX, load_case,
    hosvd, run_pod, reconstruct_hosvd, reconstruct_pod,
    minmax, minmax_scale_point, standardise_train,
    rel_error, make_mo_gpr,
)

# ─────────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR  = Path('/home/isacco/DATA/')
CASES_DIR = DATA_DIR / 'Datasets_Isacco'
PLOT_DIR  = Path('plots_interp')
PLOT_DIR.mkdir(exist_ok=True)

RE_VALS       = [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
MF_VALS       = [0.04,  0.06,  0.08,  0.10,  0.12,  0.14,  0.16,  0.18,  0.20,  0.22]
RE_TRAIN_VALS = [11000, 12000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
MF_TRAIN_VALS = [0.04,  0.06,  0.08,  0.10,  0.12,  0.14,  0.16,  0.18,  0.20,  0.22]
RE_TEST       = 13000
MF_TEST_VALS  = MF_VALS

IMPORTANT_FIELDS = ['T', 'CH4', 'O2', 'CO2', 'H2O']
E_THRESHOLD      = 0.99

KERNEL_1D = ConstantKernel(1.0) * Matern(length_scale=1.0,
                                          length_scale_bounds='fixed', nu=2.5)
KERNEL_2D = ConstantKernel(1.0) * Matern(length_scale=np.ones(2),
                                          length_scale_bounds='fixed', nu=2.5)

HOSVD_METHODS = ['HOSVD GPR', 'HOSVD Spline', 'HOSVD Linear']
POD_METHODS   = ['POD GPR', 'POD RBF tps', 'POD RBF mq', 'POD Bilinear', 'POD Nearest']
HOSVD_COLORS  = ['#2196F3', '#FF9800', '#E91E63']
POD_COLORS    = ['#FF5722', '#009688', '#795548', '#607D8B', '#9E9E9E']

# ─────────────────────────────────────────────────────────────────────────────
# 1. load data
# ─────────────────────────────────────────────────────────────────────────────

print('Loading cases...')
sample_grid, x_vals, z_vals = load_case(sorted(CASES_DIR.glob('*.xy'))[0])
Nz, Nx, Nsp = sample_grid.shape
x_min, x_max = x_vals.min(), x_vals.max()
z_min, z_max = z_vals.min(), z_vals.max()

params_all      = np.array([[re, mf] for re in RE_VALS for mf in MF_VALS])
tensor_flat_all = np.empty((len(params_all), Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params_all):
    path = next(CASES_DIR.glob(f'*_mfH2_{mf:.2f}_Re_{int(re)}.xy'))
    tensor_flat_all[k], _, _ = load_case(path)
print(f'Loaded {len(params_all)} cases, shape {tensor_flat_all.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# 2. train / test split
# ─────────────────────────────────────────────────────────────────────────────

mf_train_set = {round(v, 2) for v in MF_TRAIN_VALS}

def in_train(re, mf):
    return (re in RE_TRAIN_VALS) and (round(mf, 2) in mf_train_set)

def get_test_field(re_test, mf_test):
    mask = np.array([(re == re_test) and (round(mf, 2) == round(mf_test, 2))
                     for re, mf in params_all])
    return tensor_flat_all[mask][0]

train_mask   = np.array([in_train(re, mf) for re, mf in params_all])
params_train = params_all[train_mask]
T_train      = tensor_flat_all[train_mask]
print(f'Training cases: {len(params_train)}')

# ─────────────────────────────────────────────────────────────────────────────
# 3. standardise
# ─────────────────────────────────────────────────────────────────────────────

T_train_s, mu, std = standardise_train(T_train)

# ─────────────────────────────────────────────────────────────────────────────
# 4. HOSVD
# ─────────────────────────────────────────────────────────────────────────────

n_re_tr      = len(RE_TRAIN_VALS)
n_mf_tr      = len(MF_TRAIN_VALS)
re_to_idx_tr = {v: i for i, v in enumerate(RE_TRAIN_VALS)}
mf_to_idx_tr = {round(v, 2): j for j, v in enumerate(MF_TRAIN_VALS)}

T_grid_s = np.zeros((n_re_tr, n_mf_tr, Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params_train):
    i = re_to_idx_tr[int(re)]
    j = mf_to_idx_tr[round(mf, 2)]
    T_grid_s[i, j] = T_train_s[k]

print('Running HOSVD...')
core, factors, sv_list = hosvd(T_grid_s, energy_threshold=E_THRESHOLD)
U_re, U_mf = factors[0], factors[1]
r_re, r_mf = U_re.shape[1], U_mf.shape[1]
print(f'U_re: {U_re.shape}   U_mf: {U_mf.shape}   core: {core.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# 5. POD
# ─────────────────────────────────────────────────────────────────────────────

print('\nRunning POD...')
V_pod, S_pod, a_tr = run_pod(T_train_s, energy_threshold=E_THRESHOLD)
r_pod = V_pod.shape[1]
print(f'POD rank: {r_pod}')

# ─────────────────────────────────────────────────────────────────────────────
# 6. fit surrogates
# ─────────────────────────────────────────────────────────────────────────────

Re_tr_nm = minmax(np.array(RE_TRAIN_VALS, float))
Mf_tr_nm = minmax(np.array(MF_TRAIN_VALS, float))

# ── HOSVD 1D GPR ─────────────────────────────────────────────────────────────
print('\nFitting HOSVD 1D GPR...')
gpr_re = make_mo_gpr(KERNEL_1D)
gpr_re.fit(Re_tr_nm.reshape(-1, 1), U_re)
gpr_mf = make_mo_gpr(KERNEL_1D)
gpr_mf.fit(Mf_tr_nm.reshape(-1, 1), U_mf)

# ── HOSVD 1D Cubic Spline ────────────────────────────────────────────────────
cs_re = CubicSpline(Re_tr_nm, U_re)
cs_mf = CubicSpline(Mf_tr_nm, U_mf)

# ── HOSVD 1D Linear ──────────────────────────────────────────────────────────
li_re = interp1d(Re_tr_nm, U_re, axis=0, kind='linear', fill_value='extrapolate')
li_mf = interp1d(Mf_tr_nm, U_mf, axis=0, kind='linear', fill_value='extrapolate')

# ── POD 2D GPR ────────────────────────────────────────────────────────────────
print('Fitting POD 2D GPR...')
param_scaler = StandardScaler().fit(params_train)
P_train_s    = param_scaler.transform(params_train)
gpr_pod      = make_mo_gpr(KERNEL_2D)
gpr_pod.fit(P_train_s, a_tr)

# ── POD 2D RBF ───────────────────────────────────────────────────────────────
# normalise (Re, mf) each to [0,1] for RBF
Re_tr_nm_all = np.array([minmax_scale_point(re, RE_TRAIN_VALS)
                          for re in params_train[:, 0]])
Mf_tr_nm_all = np.array([minmax_scale_point(mf, MF_TRAIN_VALS)
                          for mf in params_train[:, 1]])
X_rbf_tr = np.column_stack([Re_tr_nm_all, Mf_tr_nm_all])

print('Fitting POD RBF interpolants...')
rbf_tps = RBFInterpolator(X_rbf_tr, a_tr, kernel='thin_plate_spline', degree=1)
rbf_mq  = RBFInterpolator(X_rbf_tr, a_tr, kernel='multiquadric', epsilon=1.0)

# ── POD bilinear / nearest (regular grid) ────────────────────────────────────
# params_train is ordered (Re slow, mf fast) so reshape gives (n_re, n_mf, r_pod)
a_tr_grid    = a_tr.reshape(n_re_tr, n_mf_tr, r_pod)
rgi_bilinear = RegularGridInterpolator((Re_tr_nm, Mf_tr_nm), a_tr_grid, method='linear')
rgi_nearest  = RegularGridInterpolator((Re_tr_nm, Mf_tr_nm), a_tr_grid, method='nearest')

# ─────────────────────────────────────────────────────────────────────────────
# 7. pre-compute Re-axis predictions (same Re_test for all mf)
# ─────────────────────────────────────────────────────────────────────────────

Re_test_nm = minmax_scale_point(RE_TEST, RE_TRAIN_VALS)

alpha_Re_gpr = np.array([est.predict([[Re_test_nm]])[0].item()
                          for est in gpr_re.estimators_])
alpha_Re_cs  = cs_re(Re_test_nm)
alpha_Re_li  = li_re(Re_test_nm)

# ─────────────────────────────────────────────────────────────────────────────
# 8. evaluation loop over mf
# ─────────────────────────────────────────────────────────────────────────────

summary = {m: [] for m in HOSVD_METHODS + POD_METHODS}

for mf_test in MF_TEST_VALS:
    mf_in_grid = round(mf_test, 2) in mf_train_set
    label      = f'Re{RE_TEST}_mf{str(mf_test).replace(".", "")}'
    print(f'\n── Re={RE_TEST}, mf={mf_test} '
          f'({"in grid" if mf_in_grid else "unseen"}) ──')

    test_plot_dir = PLOT_DIR / label
    test_plot_dir.mkdir(exist_ok=True)

    T_test_true = get_test_field(RE_TEST, mf_test)
    Mf_test_nm  = minmax_scale_point(mf_test, MF_TRAIN_VALS)
    X_test_rbf  = np.array([[Re_test_nm, Mf_test_nm]])

    # ── HOSVD reconstructions ────────────────────────────────────────────

    alpha_mf_gpr = np.array([est.predict([[Mf_test_nm]])[0].item()
                              for est in gpr_mf.estimators_])
    alpha_mf_cs  = cs_mf(Mf_test_nm)
    alpha_mf_li  = li_mf(Mf_test_nm)

    def hosvd_recon(alpha_re, alpha_mf):
        s = reconstruct_hosvd(core, factors, alpha_re, alpha_mf)
        return s * std.squeeze() + mu.squeeze()

    recon_hosvd_gpr = hosvd_recon(alpha_Re_gpr, alpha_mf_gpr)
    recon_hosvd_cs  = hosvd_recon(alpha_Re_cs,  alpha_mf_cs)
    recon_hosvd_li  = hosvd_recon(alpha_Re_li,  alpha_mf_li)

    # ── POD reconstructions ──────────────────────────────────────────────

    def pod_recon(a_pred):
        s = reconstruct_pod(V_pod, a_pred, Nz, Nx, Nsp)
        return s * std.squeeze() + mu.squeeze()

    a_gpr = gpr_pod.predict(param_scaler.transform([[RE_TEST, mf_test]])).ravel()
    a_tps = rbf_tps(X_test_rbf).ravel()
    a_mq  = rbf_mq(X_test_rbf).ravel()
    a_bi  = rgi_bilinear(X_test_rbf).ravel()
    a_nn  = rgi_nearest(X_test_rbf).ravel()

    recon_pod_gpr = pod_recon(a_gpr)
    recon_pod_tps = pod_recon(a_tps)
    recon_pod_mq  = pod_recon(a_mq)
    recon_pod_bi  = pod_recon(a_bi)
    recon_pod_nn  = pod_recon(a_nn)

    # ── errors ───────────────────────────────────────────────────────────

    all_recons = [recon_hosvd_gpr, recon_hosvd_cs,  recon_hosvd_li,
                  recon_pod_gpr,   recon_pod_tps,    recon_pod_mq,
                  recon_pod_bi,    recon_pod_nn]

    errors = {}
    for name, recon in zip(HOSVD_METHODS + POD_METHODS, all_recons):
        errs = [rel_error(recon[:, :, COL_IDX[f]], T_test_true[:, :, COL_IDX[f]])
                for f in IMPORTANT_FIELDS]
        errors[name] = errs
        m = np.nanmean(errs)
        summary[name].append(m)
        print(f'  {name:20s}: mean={m:.4f}')

    # ── per-field bar chart ───────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x = np.arange(len(IMPORTANT_FIELDS))

    w = 0.25
    for i, (m, c) in enumerate(zip(HOSVD_METHODS, HOSVD_COLORS)):
        axes[0].bar(x + (i - 1) * w, errors[m], w, color=c, alpha=0.85,
                    label=f'{m} (mean={np.nanmean(errors[m]):.4f})')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right')
    axes[0].set_ylabel('Relative L2 error')
    axes[0].set_title(f'HOSVD 1D — Re={RE_TEST}, mf={mf_test}')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    w = 0.15
    for i, (m, c) in enumerate(zip(POD_METHODS, POD_COLORS)):
        axes[1].bar(x + (i - 2) * w, errors[m], w, color=c, alpha=0.85,
                    label=f'{m} (mean={np.nanmean(errors[m]):.4f})')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right')
    axes[1].set_ylabel('Relative L2 error')
    axes[1].set_title(f'POD 2D — Re={RE_TEST}, mf={mf_test}')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    status = 'in training grid' if mf_in_grid else 'unseen mf'
    fig.suptitle(f'Re={RE_TEST}, mf={mf_test}  |  {status}', fontsize=10)
    plt.tight_layout()
    fig.savefig(test_plot_dir / 'error_per_field.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {test_plot_dir}/error_per_field.png')

# ─────────────────────────────────────────────────────────────────────────────
# 9. summary plot: error vs mf for all methods
# ─────────────────────────────────────────────────────────────────────────────

markers = ['o', 's', '^', 'D', 'v']

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for m, c in zip(HOSVD_METHODS, HOSVD_COLORS):
    axes[0].plot(MF_TEST_VALS, summary[m], 'o-', color=c, lw=1.8, ms=6, label=m)
for mf in MF_TEST_VALS:
    if round(mf, 2) in mf_train_set:
        axes[0].axvline(mf, color='gray', ls=':', lw=1, alpha=0.5)
axes[0].set_xlabel('mf')
axes[0].set_ylabel('Mean relative L2 error')
axes[0].set_title(f'HOSVD 1D — Error vs mf at Re={RE_TEST}')
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle='--', alpha=0.5)

for m, c, mk in zip(POD_METHODS, POD_COLORS, markers):
    axes[1].plot(MF_TEST_VALS, summary[m], f'{mk}-', color=c, lw=1.8, ms=6, label=m)
for mf in MF_TEST_VALS:
    if round(mf, 2) in mf_train_set:
        axes[1].axvline(mf, color='gray', ls=':', lw=1, alpha=0.5)
axes[1].set_xlabel('mf')
axes[1].set_ylabel('Mean relative L2 error')
axes[1].set_title(f'POD 2D — Error vs mf at Re={RE_TEST}')
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle='--', alpha=0.5)

fig.suptitle(
    f'Interpolation method comparison — Re={RE_TEST} (interpolated)\n'
    f'Training: Re={RE_TRAIN_VALS}, mf={MF_TRAIN_VALS}  '
    f'(vertical dotted lines = training mf values)',
    fontsize=10,
)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'summary_error_vs_mf.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {PLOT_DIR}/summary_error_vs_mf.png')
print('Done.')
