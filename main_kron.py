"""
main.py  —  HOSVD+GPR vs HOSVD+Kronecker vs Kronecker+Residual vs POD+GPR
===========================================================================
For a single interpolated Re, evaluate all four methods across all mf values.

  Training: Re in {11000, 15000, 17000, 19000} x mf in {0.04, 0.08, 0.16, 0.20}
  Interpolated Re: 13000

Methods
-------
  HOSVD + 1D GPR       : two independent 1D GPRs on U_re and U_mf rows
  HOSVD + Kronecker    : joint GPR on Tucker coefficients, kernel k_Re * k_mf
  Kronecker + Residual : Kronecker reconstruction + POD GPR on Tucker residuals
  POD   + 2D GPR       : standard unstructured 2D GPR on POD coefficients
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)

from utils_kron import (
    COLS, COL_IDX, load_case,
    hosvd, run_pod, reconstruct_hosvd, reconstruct_pod,
    minmax, minmax_scale_point, standardise_train,
    rel_error,
    make_mo_gpr,
    fit_kronecker_gpr, reconstruct_kronecker,
    fit_residual_gpr, predict_residual_gpr,
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

RE_VALS = [11000, 13000, 15000, 17000, 19000]
MF_VALS = [0.04,  0.08,  0.12,  0.16,  0.20]

RE_TRAIN_VALS = [11000, 15000, 17000, 19000]
MF_TRAIN_VALS = [0.04,  0.08,  0.16,  0.20]

RE_TEST      = 13000
MF_TEST_VALS = MF_VALS

IMPORTANT_FIELDS = ['T', 'CH4', 'O2', 'CO2', 'H2O']

KERNEL_1D  = ConstantKernel(1.0) * Matern(length_scale=1.0,
                                            length_scale_bounds='fixed', nu=2.5)
KERNEL_2D  = ConstantKernel(1.0) * Matern(length_scale=np.ones(2),
                                            length_scale_bounds='fixed', nu=2.5)
KERNEL_KRE = Matern(length_scale=1.0, length_scale_bounds='fixed', nu=2.5)
KERNEL_KMF = Matern(length_scale=1.0, length_scale_bounds='fixed', nu=2.5)

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

# ── Kronecker GPR ─────────────────────────────────────────────────────────────
print('Fitting Kronecker GPR...')
kgpr_list, _, _ = fit_kronecker_gpr(
    Re_tr_nm, Mf_tr_nm, core, factors,
    kernel_re=KERNEL_KRE, kernel_mf=KERNEL_KMF,
)
print(f'  Fitted {len(kgpr_list)} Kronecker GPRs ({r_re}x{r_mf})')

# ── Residual GPR (on Tucker in-sample residuals) ──────────────────────────────
print('Fitting residual GPR on Tucker reconstruction errors...')
res_V, res_a_tr, res_gpr, res_scaler, r_res, S_res = fit_residual_gpr(
    params_train, T_train, core, factors,
    re_to_idx_tr, mf_to_idx_tr,
    mu, std, KERNEL_2D,
    energy_threshold=E_THRESHOLD,
)

# ── POD 2D GPR ────────────────────────────────────────────────────────────────
param_scaler = StandardScaler().fit(params_train)
P_train_s    = param_scaler.transform(params_train)
print('Fitting POD 2D GPR...')
gpr_pod = make_mo_gpr(KERNEL_2D)
gpr_pod.fit(P_train_s, a_tr)

# predict HOSVD U_re once — Re=RE_TEST same for all mf
Re_test_nm    = minmax_scale_point(RE_TEST, RE_TRAIN_VALS)
preds_re      = [est.predict([[Re_test_nm]], return_std=True)
                 for est in gpr_re.estimators_]
alpha_Re_pred = np.array([p[0].item() for p in preds_re])
alpha_Re_std  = np.array([p[1].item() for p in preds_re])


# ─────────────────────────────────────────────────────────────────────────────
# 7. evaluate at every mf value
# ─────────────────────────────────────────────────────────────────────────────

summary_errors_h = []   # HOSVD + 1D GPR
summary_errors_k = []   # HOSVD + Kronecker
summary_errors_r = []   # Kronecker + Residual
summary_errors_p = []   # POD + 2D GPR

for mf_test in MF_TEST_VALS:

    mf_in_grid = round(mf_test, 2) in mf_train_set
    label      = f'Re{RE_TEST}_mf{str(mf_test).replace(".", "")}'
    mf_src     = 'direct' if mf_in_grid else 'GPR'

    print(f'\n{"─"*62}')
    print(f'Re={RE_TEST}, mf={mf_test}  '
          f'(mf {"IN" if mf_in_grid else "NOT IN"} training grid)')
    print(f'{"─"*62}')

    test_plot_dir = PLOT_DIR / label
    test_plot_dir.mkdir(exist_ok=True)

    T_test_true = get_test_field(RE_TEST, mf_test)
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

    # ── Kronecker GPR ─────────────────────────────────────────────────────
    recon_s_k, alpha_Re_k, alpha_mf_k, std_k = reconstruct_kronecker(
        core, factors, kgpr_list, r_re, r_mf, Re_test_nm, Mf_test_nm
    )
    recon_k = recon_s_k * std.squeeze() + mu.squeeze()

    # ── Kronecker + Residual ──────────────────────────────────────────────
    res_pred_s, a_res_pred, a_res_std = predict_residual_gpr(
        res_V, res_gpr, res_scaler, RE_TEST, mf_test, Nz, Nx, Nsp
    )
    # add residual correction in scaled space, then unscale
    recon_s_r = recon_s_k + res_pred_s
    recon_r   = recon_s_r * std.squeeze() + mu.squeeze()

    # ── POD 2D GPR ────────────────────────────────────────────────────────
    P_test_s  = param_scaler.transform([[RE_TEST, mf_test]])
    preds_pod = [est.predict(P_test_s, return_std=True)
                 for est in gpr_pod.estimators_]
    a_pred = np.array([p[0].item() for p in preds_pod])
    a_std  = np.array([p[1].item() for p in preds_pod])
    T_test_s = (T_test_true - mu.squeeze()) / std.squeeze()
    a_true   = (V_pod.T @ T_test_s.ravel()).ravel()
    recon_s_p = reconstruct_pod(V_pod, a_pred, Nz, Nx, Nsp)
    recon_p   = recon_s_p * std.squeeze() + mu.squeeze()

    # ── errors ────────────────────────────────────────────────────────────
    def field_errors(recon):
        return [rel_error(recon[:, :, COL_IDX[n]],
                          T_test_true[:, :, COL_IDX[n]])
                for n in IMPORTANT_FIELDS]

    errors_h = field_errors(recon_h)
    errors_k = field_errors(recon_k)
    errors_r = field_errors(recon_r)
    errors_p = field_errors(recon_p)

    mean_h = np.nanmean(errors_h)
    mean_k = np.nanmean(errors_k)
    mean_r = np.nanmean(errors_r)
    mean_p = np.nanmean(errors_p)

    summary_errors_h.append(mean_h)
    summary_errors_k.append(mean_k)
    summary_errors_r.append(mean_r)
    summary_errors_p.append(mean_p)

    print(f'  HOSVD 1D  : {mean_h:.4f}')
    print(f'  Kronecker : {mean_k:.4f}')
    print(f'  Kron+Res  : {mean_r:.4f}')
    print(f'  POD       : {mean_p:.4f}')

    # ── error bar chart ───────────────────────────────────────────────────
    x = np.arange(len(IMPORTANT_FIELDS)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 1.5*w, errors_h, w, color='#2196F3', alpha=0.85,
           label=f'HOSVD 1D    (mean={mean_h:.4f})')
    ax.bar(x - 0.5*w, errors_k, w, color='#9C27B0', alpha=0.85,
           label=f'Kronecker   (mean={mean_k:.4f})')
    ax.bar(x + 0.5*w, errors_r, w, color='#4CAF50', alpha=0.85,
           label=f'Kron+Resid  (mean={mean_r:.4f})')
    ax.bar(x + 1.5*w, errors_p, w, color='#FF5722', alpha=0.85,
           label=f'POD 2D      (mean={mean_p:.4f})')
    ax.set_xticks(x); ax.set_xticklabels(IMPORTANT_FIELDS, rotation=45, ha='right')
    ax.set_ylabel('Relative L2 error')
    ax.set_title(f'Per-feature error — Re={RE_TEST}, mf={mf_test}')
    ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(test_plot_dir / 'relative_error_per_feat.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── spatial field plots ───────────────────────────────────────────────
    extent = [x_min, x_max, z_min, z_max]
    suptitle_sub = (f'Re={RE_TEST}, mf={mf_test}  |  '
                    f'{len(params_train)} training cases  |  mf {mf_src}')

    for name in IMPORTANT_FIELDS:
        sp = COL_IDX[name]
        F_true = T_test_true[:, :, sp]
        F_h    = recon_h[:, :, sp]
        F_k    = recon_k[:, :, sp]
        F_r    = recon_r[:, :, sp]
        F_p    = recon_p[:, :, sp]
        eh = F_h - F_true; ek = F_k - F_true
        er = F_r - F_true; ep = F_p - F_true

        vmin_f  = min(F_true.min(), F_h.min(), F_k.min(), F_r.min(), F_p.min())
        vmax_f  = max(F_true.max(), F_h.max(), F_k.max(), F_r.max(), F_p.max())
        err_lim = max(np.abs(e).max() for e in [eh, ek, er, ep])

        fig, axes = plt.subplots(2, 5, figsize=(25, 8), dpi=100)

        for ax, title, field in zip(axes[0],
                                     ['Original', 'HOSVD 1D',
                                      'Kronecker', 'Kron+Resid', 'POD'],
                                     [F_true, F_h, F_k, F_r, F_p]):
            im = ax.imshow(field, origin='lower', aspect='auto', extent=extent,
                           vmin=vmin_f, vmax=vmax_f, cmap='hot')
            ax.set_title(title); ax.set_xlabel('r'); ax.set_ylabel('z')
            plt.colorbar(im, ax=ax, label=name)

        axes[1, 0].hist(eh.ravel(), bins=80, alpha=0.6,
                        color='#2196F3', density=True, label='HOSVD 1D')
        axes[1, 0].hist(ek.ravel(), bins=80, alpha=0.6,
                        color='#9C27B0', density=True, label='Kronecker')
        axes[1, 0].hist(er.ravel(), bins=80, alpha=0.6,
                        color='#4CAF50', density=True, label='Kron+Resid')
        axes[1, 0].hist(ep.ravel(), bins=80, alpha=0.6,
                        color='#FF5722', density=True, label='POD')
        axes[1, 0].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        axes[1, 0].set_title('Error distributions')
        axes[1, 0].legend(fontsize=7)

        for ax, err, title in zip(
                axes[1, 1:],
                [eh, ek, er, ep],
                ['Error HOSVD 1D', 'Error Kronecker',
                 'Error Kron+Resid', 'Error POD']):
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


# ─────────────────────────────────────────────────────────────────────────────
# 8. summary plot
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(MF_TEST_VALS, summary_errors_h, 'o-', color='#2196F3', lw=1.8,
        ms=7, label='HOSVD + 1D GPR')
ax.plot(MF_TEST_VALS, summary_errors_k, 'D-', color='#9C27B0', lw=1.8,
        ms=7, label='HOSVD + Kronecker')
ax.plot(MF_TEST_VALS, summary_errors_r, 's-', color='#4CAF50', lw=1.8,
        ms=7, label='Kronecker + Residual')
ax.plot(MF_TEST_VALS, summary_errors_p, '^-', color='#FF5722', lw=1.8,
        ms=7, label='POD + 2D GPR')

for mf in MF_TEST_VALS:
    if round(mf, 2) in mf_train_set:
        ax.axvline(mf, color='gray', ls=':', lw=1, alpha=0.6)

all_vals = summary_errors_h + summary_errors_k + summary_errors_r + summary_errors_p
for mf, eh, ek, er, ep in zip(MF_TEST_VALS, summary_errors_h,
                                summary_errors_k, summary_errors_r,
                                summary_errors_p):
    tag = 'in grid' if round(mf, 2) in mf_train_set else 'unseen'
    ax.annotate(tag, xy=(mf, max(eh, ek, er, ep)),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=7, color='gray')

ax.set_xlabel('mf')
ax.set_ylabel('Mean relative L2 error')
ax.set_title(f'Error vs mf at Re={RE_TEST} (interpolated)\n'
             f'Training: Re={RE_TRAIN_VALS}, mf={MF_TRAIN_VALS}')
ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(PLOT_DIR / 'summary_error_vs_mf.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved plots/summary_error_vs_mf.png')
print('\nDone.')