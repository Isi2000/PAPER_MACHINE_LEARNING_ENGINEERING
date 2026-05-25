"""
main.py  —  HOSVD+GPR vs POD+GPR
=================================
For two interpolated Re values, evaluate both methods across all mf values.

  Training: Re in {11000,12000,14000,15000,16000,17000,19000,20000}
            mf in {0.04,0.06,0.08,0.12,0.14,0.16,0.18,0.20,0.22}
  Test Re:  13000, 18000  (interpolated)
  Test mf:  0.10          (interpolated, held out from training)
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
PLOT_DIR  = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)

RE_VALS = [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
MF_VALS = [0.04,  0.06,  0.08,  0.10,  0.12,  0.14,  0.16,  0.18,  0.20,  0.22]

RE_TRAIN_VALS = [11000, 12000, 13000, 14000, 16000, 17000, 18000, 19000, 20000]
MF_TRAIN_VALS = [0.04,  0.06,  0.08, 0.12,  0.14,  0.16,  0.18,  0.20,  0.22]

RE_TEST_VALS = [15000]
MF_TEST      = 0.10
MF_TEST_VALS = MF_VALS

IMPORTANT_FIELDS = [ 'T',

    # Main species
    'CH4', 'O2', 'CO', 'CO2', 'H2O',
]
KERNEL_1D  = ConstantKernel(1.0) * Matern(length_scale=0.5,
                                            length_scale_bounds='fixed', nu=0.5)
KERNEL_2D  = ConstantKernel(1.0) * Matern(length_scale=np.full(2, 0.5),
                                            length_scale_bounds='fixed', nu=0.5)
E_THRESHOLD = 0.99


# ─────────────────────────────────────────────────────────────────────────────
# 1. load all 100 cases
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

print(f'Training cases : {len(params_train)}  '
      f'(Re={RE_TRAIN_VALS}, mf={MF_TRAIN_VALS})')


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
# 5. POD
# ─────────────────────────────────────────────────────────────────────────────

print('\nRunning POD...')
V_pod, S_pod, a_tr = run_pod(T_train_s, energy_threshold=E_THRESHOLD)
r_pod = V_pod.shape[1]
print(f'POD rank: {r_pod}')

plot_singular_values(sv_list, factors, S_pod, r_pod,
                     n_re_tr, n_mf_tr, Nz, Nx, Nsp, PLOT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 6. fit all surrogates
# ─────────────────────────────────────────────────────────────────────────────

Re_tr_nm = minmax(np.array(RE_TRAIN_VALS, float))
Mf_tr_nm = minmax(np.array(MF_TRAIN_VALS, float))

# ── HOSVD 1D GPRs ────────────────────────────────────────────────────────────
print('\nFitting HOSVD 1D GPRs...')
gpr_re = make_mo_gpr(KERNEL_1D)
gpr_re.fit(Re_tr_nm.reshape(-1, 1), U_re)
gpr_mf = make_mo_gpr(KERNEL_1D)
gpr_mf.fit(Mf_tr_nm.reshape(-1, 1), U_mf)

# ── POD 2D GPR ────────────────────────────────────────────────────────────────
param_scaler = StandardScaler().fit(params_train)
P_train_s    = param_scaler.transform(params_train)
print('Fitting POD 2D GPR...')
gpr_pod = make_mo_gpr(KERNEL_2D)
gpr_pod.fit(P_train_s, a_tr)


# ─────────────────────────────────────────────────────────────────────────────
# 7. evaluate at every (Re_test, mf) combination
# ─────────────────────────────────────────────────────────────────────────────

all_summary_h = {}   # {re_test: [mean_h per mf]}
all_summary_p = {}   # {re_test: [mean_p per mf]}
all_errors_h  = []   # accumulated per-field errors over all (re, mf) test cases
all_errors_p  = []

for re_test in RE_TEST_VALS:

    re_in_grid = re_test in RE_TRAIN_VALS
    Re_test_nm = minmax_scale_point(re_test, RE_TRAIN_VALS)

    # predict U_re for this Re test value
    preds_re      = [est.predict([[Re_test_nm]], return_std=True)
                     for est in gpr_re.estimators_]
    alpha_Re_pred = np.array([p[0].item() for p in preds_re])
    alpha_Re_std  = np.array([p[1].item() for p in preds_re])

    summary_errors_h = []
    summary_errors_p = []

    for mf_test in MF_TEST_VALS:

        mf_in_grid = round(mf_test, 2) in mf_train_set
        label      = f'Re{re_test}_mf{str(mf_test).replace(".", "")}'
        mf_src     = 'direct' if mf_in_grid else 'GPR'

        print(f'\n{"─"*62}')
        print(f'Re={re_test}, mf={mf_test}  '
              f'(mf {"IN" if mf_in_grid else "NOT IN"} training grid)')
        print(f'{"─"*62}')

        test_plot_dir = PLOT_DIR / label
        test_plot_dir.mkdir(exist_ok=True)

        T_test_true = get_test_field(re_test, mf_test)
        Mf_test_nm  = minmax_scale_point(mf_test, MF_TRAIN_VALS)

        # ── HOSVD 1D GPR ─────────────────────────────────────────────────────
        if mf_in_grid:
            j_mf          = mf_to_idx_tr[round(mf_test, 2)]
            alpha_mf_pred = U_mf[j_mf]
            alpha_mf_std  = np.zeros(r_mf)
        else:
            preds_mf      = [est.predict([[Mf_test_nm]], return_std=True)
                             for est in gpr_mf.estimators_]
            alpha_mf_pred = np.array([p[0].item() for p in preds_mf])
            alpha_mf_std  = np.array([p[1].item() for p in preds_mf])

        recon_s_h = reconstruct_hosvd(core, factors, alpha_Re_pred, alpha_mf_pred)
        recon_h   = recon_s_h * std.squeeze() + mu.squeeze()

        # ── POD 2D GPR ────────────────────────────────────────────────────────
        P_test_s  = param_scaler.transform([[re_test, mf_test]])
        preds_pod = [est.predict(P_test_s, return_std=True)
                     for est in gpr_pod.estimators_]
        a_pred = np.array([p[0].item() for p in preds_pod])
        a_std  = np.array([p[1].item() for p in preds_pod])
        T_test_s = (T_test_true - mu.squeeze()) / std.squeeze()
        a_true   = (V_pod.T @ T_test_s.ravel()).ravel()
        recon_s_p = reconstruct_pod(V_pod, a_pred, Nz, Nx, Nsp)
        recon_p   = recon_s_p * std.squeeze() + mu.squeeze()

        # ── errors ────────────────────────────────────────────────────────────
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

        print(f'  HOSVD 1D  : {mean_h:.4f}')
        print(f'  POD       : {mean_p:.4f}')

        # ── error bar chart ───────────────────────────────────────────────────
        x = np.arange(len(IMPORTANT_FIELDS)); w = 0.3
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.5*w, errors_h, w, color='#2196F3', alpha=0.85,
               label=f'HOSVD 1D  (mean={mean_h:.4f})')
        ax.bar(x + 0.5*w, errors_p, w, color='#FF5722', alpha=0.85,
               label=f'POD 2D    (mean={mean_p:.4f})')
        ax.set_xticks(x); ax.set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right')
        ax.set_ylabel('Relative L2 error')
        ax.set_title(f'Per-feature error — Re={re_test}, mf={mf_test}')
        ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        fig.savefig(test_plot_dir / 'relative_error_per_feat.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── spatial field plots ───────────────────────────────────────────────
        extent = [x_min, x_max, z_min, z_max]
        suptitle_sub = (f'Re={re_test}, mf={mf_test}  |  '
                        f'{len(params_train)} training cases  |  mf {mf_src}')

        for name in IMPORTANT_FIELDS:
            sp = COL_IDX[name]
            F_true = T_test_true[:, :, sp]
            F_h    = recon_h[:, :, sp]
            F_p    = recon_p[:, :, sp]
            eh = F_h - F_true
            ep = F_p - F_true

            vmin_f  = min(F_true.min(), F_h.min(), F_p.min())
            vmax_f  = max(F_true.max(), F_h.max(), F_p.max())
            err_lim = max(np.abs(e).max() for e in [eh, ep])

            fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=100)

            for ax, title, field in zip(axes[0],
                                         ['Original', 'HOSVD 1D', 'POD'],
                                         [F_true, F_h, F_p]):
                im = ax.imshow(field, origin='lower', aspect='auto', extent=extent,
                               vmin=vmin_f, vmax=vmax_f, cmap='hot')
                ax.set_title(title); ax.set_xlabel('r'); ax.set_ylabel('z')
                plt.colorbar(im, ax=ax, label=name)

            axes[1, 0].hist(eh.ravel(), bins=80, alpha=0.6,
                            color='#2196F3', density=True, label='HOSVD 1D')
            axes[1, 0].hist(ep.ravel(), bins=80, alpha=0.6,
                            color='#FF5722', density=True, label='POD')
            axes[1, 0].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
            axes[1, 0].set_title('Error distributions')
            axes[1, 0].legend(fontsize=7)

            for ax, err, title in zip(
                    axes[1, 1:],
                    [eh, ep],
                    ['Error HOSVD 1D', 'Error POD']):
                im = ax.imshow(err, origin='lower', aspect='auto', extent=extent,
                               vmin=-err_lim, vmax=err_lim, cmap='RdBu_r')
                ax.set_title(title); ax.set_xlabel('r'); ax.set_ylabel('z')
                plt.colorbar(im, ax=ax, label=f'Delta {name}')

            fig.suptitle(f'{name} — {suptitle_sub}', fontsize=11)
            plt.tight_layout()
            fig.savefig(test_plot_dir / f'field_{name}.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved {test_plot_dir}/field_{name}.png')

    all_summary_h[re_test] = summary_errors_h
    all_summary_p[re_test] = summary_errors_p


# ─────────────────────────────────────────────────────────────────────────────
# 8. summary plot
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(RE_TEST_VALS), figsize=(8 * len(RE_TEST_VALS), 4),
                         sharey=True)
if len(RE_TEST_VALS) == 1:
    axes = [axes]

for ax, re_test in zip(axes, RE_TEST_VALS):
    errs_h = all_summary_h[re_test]
    errs_p = all_summary_p[re_test]

    ax.plot(MF_TEST_VALS, errs_h, 'o-', color='#2196F3', lw=1.8, ms=7,
            label='HOSVD + 1D GPR')
    ax.plot(MF_TEST_VALS, errs_p, 's-', color='#FF5722', lw=1.8, ms=7,
            label='POD + 2D GPR')

    for mf in MF_TEST_VALS:
        if round(mf, 2) in mf_train_set:
            ax.axvline(mf, color='gray', ls=':', lw=1, alpha=0.6)

    for mf, eh, ep in zip(MF_TEST_VALS, errs_h, errs_p):
        tag = 'in grid' if round(mf, 2) in mf_train_set else 'unseen'
        ax.annotate(tag, xy=(mf, max(eh, ep)),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7, color='gray')

    re_tag = 'interpolated' if re_test not in RE_TRAIN_VALS else 'in grid'
    ax.set_xlabel('mf')
    ax.set_ylabel('Mean relative L2 error')
    ax.set_title(f'Re={re_test} ({re_tag})')
    ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.5)

fig.suptitle(f'Error vs mf — test Re={RE_TEST_VALS}, test mf={MF_TEST}\n'
             f'Training: Re={RE_TRAIN_VALS}, mf={MF_TRAIN_VALS}', fontsize=10)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'summary_error_vs_mf.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved plots/summary_error_vs_mf.png')

# ── bar plot: mean error per field over the whole test set ────────────────────
mean_per_field_h = np.nanmean(all_errors_h, axis=0)   # (n_fields,)
mean_per_field_p = np.nanmean(all_errors_p, axis=0)

x = np.arange(len(IMPORTANT_FIELDS))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width / 2, mean_per_field_h, width,
       label='HOSVD + 1D GPR', color='#2196F3', alpha=0.85)
ax.bar(x + width / 2, mean_per_field_p, width,
       label='POD + 2D GPR',   color='#FF5722', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean relative L2 error')
ax.set_title(f'Mean error per field over all test cases\n'
             f'Test Re={RE_TEST_VALS}, test mf={MF_TEST_VALS}')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'summary_error_per_field.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved plots/summary_error_per_field.png')

print('\nDone.')