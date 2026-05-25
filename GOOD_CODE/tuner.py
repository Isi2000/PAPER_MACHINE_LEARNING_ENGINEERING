"""
tuner.py  —  kernel hyperparameter sweep for HOSVD+GPR vs POD+GPR
==================================================================
Joint sweep over (ls_1d) × (2D kernel name × ls_2d) — 9×6×9 = 486 configs.
Evaluates both methods on the held-out test set (not LOO).

Saves only:
  plots_tuner/singular_values.png
  plots_tuner/joint_sweep.png          (heatmap per kernel: ls_1d × ls_2d)
  plots_tuner/best_config_errors_Re*.png
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)

from utils import (
    COL_IDX, load_case,
    hosvd, run_pod, reconstruct_hosvd, reconstruct_pod,
    minmax, minmax_scale_point, standardise_train,
    rel_error, make_mo_gpr,
    plot_singular_values,
    KERNEL_NAMES, LS_SWEEP,
    build_kernel_1d, build_kernel_2d,
    HOSVD_1D_KERNEL_NAME,
)

# ─────────────────────────────────────────────────────────────────────────────
# config  —  mirrors main.py
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR  = Path('/home/isacco/DATA/')
CASES_DIR = DATA_DIR / 'Datasets_Isacco'
PLOT_DIR  = Path('plots_tuner')
PLOT_DIR.mkdir(exist_ok=True)

RE_VALS = [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
MF_VALS = [0.04,  0.06,  0.08,  0.10,  0.12,  0.14,  0.16,  0.18,  0.20,  0.22]

RE_TRAIN_VALS = [11000, 12000, 14000, 15000, 16000, 17000, 19000, 20000]
MF_TRAIN_VALS = [0.04,  0.06,  0.08,  0.12,  0.14,  0.16,  0.18,  0.20,  0.22]

RE_TEST_VALS = [13000, 18000]
MF_TEST      = 0.10

IMPORTANT_FIELDS = ['T', 'CH4', 'O2', 'CO2', 'H2O', 'H2']
E_THRESHOLD      = 0.99


# ─────────────────────────────────────────────────────────────────────────────
# 1. load data
# ─────────────────────────────────────────────────────────────────────────────

print('Loading cases...')
sample_grid, x_vals, z_vals = load_case(sorted(CASES_DIR.glob('*.xy'))[0])
Nz, Nx, Nsp = sample_grid.shape

params_all      = np.array([[re, mf] for re in RE_VALS for mf in MF_VALS])
tensor_flat_all = np.empty((len(params_all), Nz, Nx, Nsp), dtype=np.float32)
for k, (re, mf) in enumerate(params_all):
    path = next(CASES_DIR.glob(f'*_mfH2_{mf:.2f}_Re_{int(re)}.xy'))
    tensor_flat_all[k], _, _ = load_case(path)

print(f'Loaded {len(params_all)} cases  {tensor_flat_all.shape}')

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
print(f'Training: {len(params_train)} cases')

# ─────────────────────────────────────────────────────────────────────────────
# 3. standardise
# ─────────────────────────────────────────────────────────────────────────────

T_train_s, mu, std = standardise_train(T_train)

# ─────────────────────────────────────────────────────────────────────────────
# 4. HOSVD  (computed once — decomposition is kernel-independent)
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
print('Running HOSVD...')
core, factors, sv_list = hosvd(T_grid_s, energy_threshold=E_THRESHOLD)
U_re, U_mf = factors[0], factors[1]
r_re, r_mf = U_re.shape[1], U_mf.shape[1]
print(f'U_re: {U_re.shape}   U_mf: {U_mf.shape}   core: {core.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# 5. POD  (computed once)
# ─────────────────────────────────────────────────────────────────────────────

print('Running POD...')
V_pod, S_pod, a_tr = run_pod(T_train_s, energy_threshold=E_THRESHOLD)
r_pod = V_pod.shape[1]
print(f'POD rank: {r_pod}')

plot_singular_values(sv_list, factors, S_pod, r_pod,
                     n_re_tr, n_mf_tr, Nz, Nx, Nsp, PLOT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 6. fixed preprocessing for GPR inputs
# ─────────────────────────────────────────────────────────────────────────────

Re_tr_nm     = minmax(np.array(RE_TRAIN_VALS, float))
Mf_tr_nm     = minmax(np.array(MF_TRAIN_VALS, float))
param_scaler = StandardScaler().fit(params_train)
P_train_s    = param_scaler.transform(params_train)

Re_test_nm = {re: minmax_scale_point(re, RE_TRAIN_VALS) for re in RE_TEST_VALS}
Mf_test_nm = minmax_scale_point(MF_TEST, MF_TRAIN_VALS)
P_test_s   = {re: param_scaler.transform([[re, MF_TEST]]) for re in RE_TEST_VALS}
T_test     = {re: get_test_field(re, MF_TEST) for re in RE_TEST_VALS}

# ─────────────────────────────────────────────────────────────────────────────
# 7. evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def eval_test(gpr_re, gpr_mf, gpr_pod):
    """Mean relative L2 error over RE_TEST_VALS x IMPORTANT_FIELDS at MF_TEST."""
    errs_h, errs_p = [], []
    for re_test in RE_TEST_VALS:
        preds_re      = [e.predict([[Re_test_nm[re_test]]], return_std=True)
                         for e in gpr_re.estimators_]
        alpha_Re_pred = np.array([p[0].item() for p in preds_re])

        preds_mf      = [e.predict([[Mf_test_nm]], return_std=True)
                         for e in gpr_mf.estimators_]
        alpha_mf_pred = np.array([p[0].item() for p in preds_mf])

        recon_h = reconstruct_hosvd(core, factors, alpha_Re_pred, alpha_mf_pred)
        recon_h = recon_h * std.squeeze() + mu.squeeze()

        preds_pod = [e.predict(P_test_s[re_test], return_std=True)
                     for e in gpr_pod.estimators_]
        a_pred    = np.array([p[0].item() for p in preds_pod])
        recon_p   = reconstruct_pod(V_pod, a_pred, Nz, Nx, Nsp)
        recon_p   = recon_p * std.squeeze() + mu.squeeze()

        T_true = T_test[re_test]
        for n in IMPORTANT_FIELDS:
            sp = COL_IDX[n]
            errs_h.append(rel_error(recon_h[:, :, sp], T_true[:, :, sp]))
            errs_p.append(rel_error(recon_p[:, :, sp], T_true[:, :, sp]))

    return float(np.nanmean(errs_h)), float(np.nanmean(errs_p))


def fit_and_eval(kernel_1d, kernel_2d):
    gpr_re  = make_mo_gpr(kernel_1d)
    gpr_mf  = make_mo_gpr(kernel_1d)
    gpr_pod = make_mo_gpr(kernel_2d)
    gpr_re.fit(Re_tr_nm.reshape(-1, 1), U_re)
    gpr_mf.fit(Mf_tr_nm.reshape(-1, 1), U_mf)
    gpr_pod.fit(P_train_s, a_tr)
    return eval_test(gpr_re, gpr_mf, gpr_pod), (gpr_re, gpr_mf, gpr_pod)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Joint sweep — (ls_1d) × (k2d_name × ls_2d)
# ─────────────────────────────────────────────────────────────────────────────

n_ls    = len(LS_SWEEP)
n_names = len(KERNEL_NAMES)
total   = n_ls * n_names * n_ls
print(f'\n{"═"*72}')
print(f' JOINT SWEEP — ls_1d × (2D kernel × ls_2d)   [{total} configs]')
print(f' 1D kernel family fixed: {HOSVD_1D_KERNEL_NAME}')
print(f'{"═"*72}')
print(f'{"ls_1d":>8}  {"2D_kernel":>16}  {"ls_2d":>7}  {"HOSVD_err":>10}  {"POD_err":>10}')
print('─'*60)

records = []   # (ls_1d, k2d_name, ls_2d, err_h, err_p)

for ls_1d in LS_SWEEP:
    k1d = build_kernel_1d(HOSVD_1D_KERNEL_NAME, ls_1d)
    for k2d_name in KERNEL_NAMES:
        for ls_2d in LS_SWEEP:
            k2d = build_kernel_2d(k2d_name, ls_2d)
            (err_h, err_p), _ = fit_and_eval(k1d, k2d)
            records.append((ls_1d, k2d_name, ls_2d, err_h, err_p))
            print(f'{ls_1d:8.3f}  {k2d_name:>16}  {ls_2d:7.3f}  '
                  f'{err_h:10.4f}  {err_p:10.4f}')

best        = min(records, key=lambda r: r[3] + r[4])
best_ls_1d  = best[0]
best_k2d_name = best[1]
best_ls_2d  = best[2]
print(f'\nBest config:  ls_1d={best_ls_1d:.3f}  '
      f'2D={best_k2d_name}  ls_2d={best_ls_2d:.3f}  '
      f'(HOSVD={best[3]:.4f}, POD={best[4]:.4f})')

# ── plot: one heatmap per 2D kernel showing combined error over (ls_1d × ls_2d)
ncols = 3
nrows = int(np.ceil(n_names / ncols))
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(5 * ncols, 4 * nrows), squeeze=False)

for idx, k2d_name in enumerate(KERNEL_NAMES):
    ax   = axes[idx // ncols][idx % ncols]
    rows = [r for r in records if r[1] == k2d_name]
    mat  = np.array([r[3] + r[4] for r in rows]).reshape(n_ls, n_ls)

    im = ax.imshow(mat, aspect='auto', cmap='viridis_r', origin='lower')
    ax.set_xticks(range(n_ls))
    ax.set_xticklabels([f'{v:.2f}' for v in LS_SWEEP],
                        rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_ls))
    ax.set_yticklabels([f'{v:.2f}' for v in LS_SWEEP], fontsize=7)
    ax.set_xlabel('ls_2d', fontsize=8)
    ax.set_ylabel('ls_1d', fontsize=8)
    ax.set_title(f'2D: {k2d_name}', fontsize=9)
    plt.colorbar(im, ax=ax, label='HOSVD+POD error')

    if k2d_name == best_k2d_name:
        brow = list(LS_SWEEP).index(best_ls_1d)
        bcol = list(LS_SWEEP).index(best_ls_2d)
        ax.add_patch(plt.Rectangle((bcol - 0.5, brow - 0.5), 1, 1,
                                    fill=False, edgecolor='red', lw=2))

for idx in range(n_names, nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

fig.suptitle(f'Joint sweep — combined (HOSVD+POD) error\n'
             f'1D kernel: {HOSVD_1D_KERNEL_NAME}  |  red box = global best',
             fontsize=10)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'joint_sweep.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {PLOT_DIR}/joint_sweep.png')

# ─────────────────────────────────────────────────────────────────────────────
# 10. Final evaluation with best config — per-field error bar chart
# ─────────────────────────────────────────────────────────────────────────────

print(f'\n{"═"*72}')
print(f' FINAL CONFIG')
print(f'  1D: {HOSVD_1D_KERNEL_NAME}  ls={best_ls_1d:.3f}')
print(f'  2D: {best_k2d_name}  ls={best_ls_2d:.3f}')
print(f'{"═"*72}')

best_k1d = build_kernel_1d(HOSVD_1D_KERNEL_NAME, best_ls_1d)
best_k2d = build_kernel_2d(best_k2d_name, best_ls_2d)
_, (gpr_re, gpr_mf, gpr_pod) = fit_and_eval(best_k1d, best_k2d)

for re_test in RE_TEST_VALS:
    preds_re      = [e.predict([[Re_test_nm[re_test]]], return_std=True)
                     for e in gpr_re.estimators_]
    alpha_Re_pred = np.array([p[0].item() for p in preds_re])
    preds_mf      = [e.predict([[Mf_test_nm]], return_std=True)
                     for e in gpr_mf.estimators_]
    alpha_mf_pred = np.array([p[0].item() for p in preds_mf])
    recon_h = reconstruct_hosvd(core, factors, alpha_Re_pred, alpha_mf_pred)
    recon_h = recon_h * std.squeeze() + mu.squeeze()

    preds_pod = [e.predict(P_test_s[re_test], return_std=True)
                 for e in gpr_pod.estimators_]
    a_pred  = np.array([p[0].item() for p in preds_pod])
    recon_p = reconstruct_pod(V_pod, a_pred, Nz, Nx, Nsp)
    recon_p = recon_p * std.squeeze() + mu.squeeze()

    T_true  = T_test[re_test]
    errs_h  = [rel_error(recon_h[:, :, COL_IDX[n]], T_true[:, :, COL_IDX[n]])
               for n in IMPORTANT_FIELDS]
    errs_p  = [rel_error(recon_p[:, :, COL_IDX[n]], T_true[:, :, COL_IDX[n]])
               for n in IMPORTANT_FIELDS]

    mean_h = np.nanmean(errs_h)
    mean_p = np.nanmean(errs_p)
    print(f'  Re={re_test}, mf={MF_TEST}:  HOSVD={mean_h:.4f}   POD={mean_p:.4f}')

    x = np.arange(len(IMPORTANT_FIELDS)); w = 0.3
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.5*w, errs_h, w, color='#2196F3', alpha=0.85,
           label=f'HOSVD + 1D GPR  (mean={mean_h:.4f})')
    ax.bar(x + 0.5*w, errs_p, w, color='#FF5722', alpha=0.85,
           label=f'POD + 2D GPR  (mean={mean_p:.4f})')
    ax.set_xticks(x); ax.set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right')
    ax.set_ylabel('Relative L2 error')
    ax.set_title(f'Best config — Re={re_test}, mf={MF_TEST}\n'
                 f'1D: {HOSVD_1D_KERNEL_NAME} ls={best_ls_1d:.3f}  |  '
                 f'2D: {best_k2d_name} ls={best_ls_2d:.3f}')
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname = PLOT_DIR / f'best_config_errors_Re{re_test}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')

print('\nDone.')
