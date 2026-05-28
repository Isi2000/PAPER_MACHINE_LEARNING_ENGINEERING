"""
main_sweep_re.py  —  HOSVD+GPR vs POD+GPR on (Re, mf)
=======================================================
For a single interpolated mf (not in the training grid), evaluate both methods
across ALL Re values [11000, 13000, 15000, 17000, 19000].

  Training: Re in {11000, 13000, 15000, 17000, 19000} x mf in {0.04, 0.08, 0.16, 0.20}
  Interpolated mf: 0.12

For each Re:
  - if Re in training grid -> U_re row read directly (no GPR)
  - if Re not in training grid -> U_re row predicted by GPR
  POD always uses the 2D GPR.

Outputs per Re:
  plots/<label>/relative_error_per_feat.png
  plots/<label>/field_{species}.png
  plots/<label>/gpr_coeffs_*.png

Summary:
  plots/summary_error_vs_re.png   -- mean relative error vs Re for both methods
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)

from utils import (
    COLS, COL_IDX, load_case,
    hosvd, run_pod, reconstruct_hosvd, reconstruct_pod,
    minmax, minmax_scale_point, standardise_train,
    rel_error,
    make_mo_gpr,
    plot_singular_values, plot_hosvd_coeffs, plot_pod_coeffs,
    plot_error_bars, plot_field,
)

# ─────────────────────────────────────────────────────────────────────────────
# configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR  = Path('/home/isacco/DATA/')
CASES_DIR = DATA_DIR / 'Datasets_Isacco'
PLOT_DIR  = Path('plots_sweep_re')
PLOT_DIR.mkdir(exist_ok=True)

RE_VALS = [11000, 13000, 15000, 17000, 19000]
MF_VALS = [0.04,  0.08,  0.12,  0.16,  0.20]

RE_TRAIN_VALS = [11000, 13000, 15000, 17000, 19000]   # all Re in training
MF_TRAIN_VALS = [0.04,  0.08,  0.16,  0.20]           # 0.12 held out

MF_TEST      = 0.12       # interpolated mf (not in training grid)
RE_TEST_VALS = RE_VALS    # evaluate at all 5 Re values

IMPORTANT_FIELDS = [
    'x', 'y',
    'Ux', 'Uy',
    'p', 'T',

    # RANS turbulence
    'k', 'epsilon', 'alphat',

    # Main species
    'CH4', 'O2', 'CO', 'CO2', 'H2O', 'N2',

    # Key radicals/intermediates
    'OH', 'HCO', 'CH3', 'HO2', 'H2', 'H2O2'
]
KERNEL_1D = ConstantKernel(1.0) * Matern(length_scale=1.0,
                                          length_scale_bounds='fixed', nu=2.5)
KERNEL_2D = ConstantKernel(1.0) * Matern(length_scale=np.ones(2),
                                          length_scale_bounds='fixed', nu=2.5)
E_THRESHOLD = 0.99


# ─────────────────────────────────────────────────────────────────────────────
# 1. load all 25 cases
# ─────────────────────────────────────────────────────────────────────────────

print('Loading cases...')
sample_grid, x_vals, z_vals = load_case(sorted(CASES_DIR.glob('*.xy'))[0])
Nz, Nx, Nsp  = sample_grid.shape
x_min, x_max = x_vals.min(), x_vals.max()
z_min, z_max = z_vals.min(), z_vals.max()

params_all      = np.array([[re, mf] for re in RE_VALS for mf in MF_VALS])
tensor_flat_all = np.empty((len(params_all), Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params_all):
    path = next(CASES_DIR.glob(f'*_mfH2_{mf:.2f}_Re_{int(re)}.xy'))
    tensor_flat_all[k], _, _ = load_case(path)

print(f'Full tensor shape: {tensor_flat_all.shape}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. train / test split
# ─────────────────────────────────────────────────────────────────────────────

re_train_set = set(RE_TRAIN_VALS)
mf_train_set = {round(v, 2) for v in MF_TRAIN_VALS}

def in_train(re, mf):
    return (re in re_train_set) and (round(mf, 2) in mf_train_set)

def get_test_field(re_test, mf_test):
    mask = np.array([(re == re_test) and (round(mf, 2) == round(mf_test, 2))
                     for re, mf in params_all])
    return tensor_flat_all[mask][0]

train_mask   = np.array([in_train(re, mf) for re, mf in params_all])
params_train = params_all[train_mask]       # (20, 2)
T_train      = tensor_flat_all[train_mask]

print(f'Training cases : {len(params_train)}  '
      f'(Re={RE_TRAIN_VALS}, mf={MF_TRAIN_VALS})')
print(f'Interpolated mf: {MF_TEST}')
print(f'Test Re values : {RE_TEST_VALS}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. standardise on training statistics
# ─────────────────────────────────────────────────────────────────────────────

T_train_s, mu, std = standardise_train(T_train)


# ─────────────────────────────────────────────────────────────────────────────
# 4. build (5, 4, Nz, Nx, Nsp) grid tensor and run HOSVD
# ─────────────────────────────────────────────────────────────────────────────

n_re_tr      = len(RE_TRAIN_VALS)
n_mf_tr      = len(MF_TRAIN_VALS)
re_to_idx_tr = {v: i for i, v in enumerate(RE_TRAIN_VALS)}
mf_to_idx_tr = {round(v, 2): j for j, v in enumerate(MF_TRAIN_VALS)}

T_grid = np.zeros((n_re_tr, n_mf_tr, Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params_train):
    i = re_to_idx_tr[int(re)]
    j = mf_to_idx_tr[round(mf, 2)]
    T_grid[i, j] = T_train[k]

T_grid_s = (T_grid - mu) / std
print(f'\nHOSVD grid tensor shape: {T_grid_s.shape}')

print('Running HOSVD...')
core, factors, sv_list = hosvd(T_grid_s, energy_threshold=E_THRESHOLD)
U_re, U_mf = factors[0], factors[1]
r_re, r_mf = U_re.shape[1], U_mf.shape[1]
print(f'U_re: {U_re.shape}   U_mf: {U_mf.shape}   core: {core.shape}')


# ─────────────────────────────────────────────────────────────────────────────
# 5. POD on the same training cases
# ─────────────────────────────────────────────────────────────────────────────

print('\nRunning POD...')
V_pod, S_pod, a_tr = run_pod(T_train_s, energy_threshold=1.0)
r_pod = V_pod.shape[1]
print(f'POD rank: {r_pod}')

plot_singular_values(sv_list, factors, S_pod, r_pod,
                     n_re_tr, n_mf_tr, Nz, Nx, Nsp, PLOT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 6. fit GPRs — once, reused for all Re test values
# ─────────────────────────────────────────────────────────────────────────────

Re_tr_nm = minmax(np.array(RE_TRAIN_VALS, float))
Mf_tr_nm = minmax(np.array(MF_TRAIN_VALS, float))

print('\nFitting HOSVD GPR for Re axis (1D)...')
gpr_re = make_mo_gpr(KERNEL_1D)
gpr_re.fit(Re_tr_nm.reshape(-1, 1), U_re)

print('Fitting HOSVD GPR for mf axis (1D)...')
gpr_mf = make_mo_gpr(KERNEL_1D)
gpr_mf.fit(Mf_tr_nm.reshape(-1, 1), U_mf)

param_scaler = StandardScaler().fit(params_train)
P_train_s    = param_scaler.transform(params_train)

print('Fitting POD GPR (2D, Re x mf)...')
gpr_pod = make_mo_gpr(KERNEL_2D)
gpr_pod.fit(P_train_s, a_tr)

# predict U_mf once — mf=MF_TEST is the same for all Re evaluations
Mf_test_nm    = minmax_scale_point(MF_TEST, MF_TRAIN_VALS)
preds_mf      = [est.predict([[Mf_test_nm]], return_std=True)
                 for est in gpr_mf.estimators_]
alpha_mf_pred = np.array([p[0].item() for p in preds_mf])
alpha_mf_std  = np.array([p[1].item() for p in preds_mf])


# ─────────────────────────────────────────────────────────────────────────────
# 7. evaluate at every Re value
# ─────────────────────────────────────────────────────────────────────────────

summary_errors_h = []
summary_errors_p = []
all_errors_h = []   # shape: (n_re, n_fields) after loop
all_errors_p = []

for re_test in RE_TEST_VALS:

    re_in_grid = re_test in re_train_set
    label      = f'Re{re_test}_mf{str(MF_TEST).replace(".", "")}'

    print(f'\n{"─"*60}')
    print(f'Re={re_test}, mf={MF_TEST}  '
          f'(Re {"IN" if re_in_grid else "NOT IN"} training grid)')
    print(f'{"─"*60}')

    test_plot_dir = PLOT_DIR / label
    test_plot_dir.mkdir(exist_ok=True)

    T_test_true = get_test_field(re_test, MF_TEST)
    Re_test_nm  = minmax_scale_point(re_test, RE_TRAIN_VALS)

    # ── HOSVD Re factor row ───────────────────────────────────────────────
    if re_in_grid:
        i_re          = re_to_idx_tr[re_test]
        alpha_Re_pred = U_re[i_re]
        alpha_Re_std  = np.zeros(r_re)
        print(f'  HOSVD Re: read U_re[{i_re}] directly')
    else:
        preds_re      = [est.predict([[Re_test_nm]], return_std=True)
                         for est in gpr_re.estimators_]
        alpha_Re_pred = np.array([p[0].item() for p in preds_re])
        alpha_Re_std  = np.array([p[1].item() for p in preds_re])
        print(f'  HOSVD Re: predicted by GPR')

    # ── POD prediction ────────────────────────────────────────────────────
    P_test_s  = param_scaler.transform([[re_test, MF_TEST]])
    preds_pod = [est.predict(P_test_s, return_std=True)
                 for est in gpr_pod.estimators_]
    a_pred = np.array([p[0].item() for p in preds_pod])
    a_std  = np.array([p[1].item() for p in preds_pod])

    # true POD coefficients for plotting
    T_test_s = (T_test_true - mu.squeeze()) / std.squeeze()
    a_true   = (V_pod.T @ T_test_s.ravel()).ravel()

    # ── reconstruct ───────────────────────────────────────────────────────
    recon_s_h = reconstruct_hosvd(core, factors, alpha_Re_pred, alpha_mf_pred)
    recon_h   = recon_s_h * std.squeeze() + mu.squeeze()

    recon_s_p = reconstruct_pod(V_pod, a_pred, Nz, Nx, Nsp)
    recon_p   = recon_s_p * std.squeeze() + mu.squeeze()

    # ── per-field errors ──────────────────────────────────────────────────
    errors_h = [rel_error(recon_h[:, :, COL_IDX[n]],
                          T_test_true[:, :, COL_IDX[n]]) for n in IMPORTANT_FIELDS]
    errors_p = [rel_error(recon_p[:, :, COL_IDX[n]],
                          T_test_true[:, :, COL_IDX[n]]) for n in IMPORTANT_FIELDS]

    all_errors_h.append(errors_h)
    all_errors_p.append(errors_p)

    mean_h = np.nanmean(errors_h)
    mean_p = np.nanmean(errors_p)
    summary_errors_h.append(mean_h)
    summary_errors_p.append(mean_p)

    print(f'  Mean relative error — HOSVD: {mean_h:.4f}   POD: {mean_p:.4f}')

    # ── per-Re plots ──────────────────────────────────────────────────────
    re_src = 'direct' if re_in_grid else 'GPR'

    plot_hosvd_coeffs(alpha_Re_pred, alpha_Re_std,
                      axis_label='Re', test_val=re_test, r=r_re,
                      plot_dir=test_plot_dir,
                      filename=f'gpr_coeffs_hosvd_re_{re_src}.png')

    plot_hosvd_coeffs(alpha_mf_pred, alpha_mf_std,
                      axis_label='mf', test_val=MF_TEST, r=r_mf,
                      plot_dir=test_plot_dir,
                      filename='gpr_coeffs_hosvd_mf_GPR.png')

    plot_pod_coeffs(a_true, a_pred, a_std,
                    re_test, MF_TEST, len(params_train), test_plot_dir)

    plot_error_bars(errors_h, errors_p, IMPORTANT_FIELDS,
                    r_re, r_mf, r_pod, re_test, MF_TEST, test_plot_dir)

    suptitle_sub = (f'Re={re_test}, mf={MF_TEST}  |  '
                    f'{len(params_train)} training cases  |  Re {re_src}')
    for name in IMPORTANT_FIELDS:
        plot_field(name, T_test_true, recon_h, recon_p,
                   COL_IDX, [x_min, x_max, z_min, z_max],
                   suptitle_sub, test_plot_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 8. summary plot: mean error vs Re
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(RE_TEST_VALS, summary_errors_h, 'o-', color='#2196F3', lw=1.8,
        ms=7, label='HOSVD + 1D GPR')
ax.plot(RE_TEST_VALS, summary_errors_p, 's-', color='#FF5722', lw=1.8,
        ms=7, label='POD + 2D GPR')

for re in RE_TEST_VALS:
    if re in re_train_set:
        ax.axvline(re, color='gray', ls=':', lw=1, alpha=0.6)

for re, eh, ep in zip(RE_TEST_VALS, summary_errors_h, summary_errors_p):
    tag = 'in grid' if re in re_train_set else 'unseen'
    ax.annotate(tag, xy=(re, max(eh, ep)),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=7, color='gray')

ax.set_xlabel('Re')
ax.set_ylabel('Mean relative L2 error')
ax.set_title(f'Error vs Re at mf={MF_TEST} (interpolated)\n'
             f'Training: Re={RE_TRAIN_VALS}, mf={MF_TRAIN_VALS}')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'summary_error_vs_re.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {PLOT_DIR}/summary_error_vs_re.png')

# ── bar plot: mean error per field over all test Re values ────────────────
mean_per_field_h = np.nanmean(all_errors_h, axis=0)   # (n_fields,)
mean_per_field_p = np.nanmean(all_errors_p, axis=0)

x = np.arange(len(IMPORTANT_FIELDS))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width / 2, mean_per_field_h, width, label='HOSVD + 1D GPR', color='#2196F3', alpha=0.85)
ax.bar(x + width / 2, mean_per_field_p, width, label='POD + 2D GPR',   color='#FF5722', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean relative L2 error')
ax.set_title(f'Mean error per field over test set (mf={MF_TEST}, Re={RE_TEST_VALS})')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'summary_error_per_field.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {PLOT_DIR}/summary_error_per_field.png')

print('\nDone.')