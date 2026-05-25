"""
utils.py  —  shared helpers for HOSVD+GPR vs POD+GPR comparison
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, RBF, RationalQuadratic, ConstantKernel
)
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

COLS = [
    'point_x', 'point_y', 'point_z', 'alphat',
    'CH', 'CH2', 'CH2O', 'CH3', 'CH4',
    'CO', 'CO2', 'epsilon',
    'H', 'H2', 'H2O', 'H2O2',
    'HCO', 'HO2', 'k', 'N2', 'nut',
    'O', 'O2', 'OH', 'p', 'T',
    'U_x', 'U_y', 'U_z',
]
COL_IDX = {name: i for i, name in enumerate(COLS)}


def load_case(path):
    """Load one .xy file → (Nz, Nx, Nsp) array, x_vals, z_vals."""
    raw    = np.loadtxt(path, skiprows=1)
    x_vals = np.unique(raw[:, COL_IDX['point_x']])
    z_vals = np.unique(raw[:, COL_IDX['point_z']])
    idx    = np.lexsort((raw[:, COL_IDX['point_x']], raw[:, COL_IDX['point_z']]))
    return (raw[idx].reshape(len(z_vals), len(x_vals), len(COLS)),
            x_vals, z_vals)


# ─────────────────────────────────────────────────────────────────────────────
# decompositions
# ─────────────────────────────────────────────────────────────────────────────

def rank_by_energy(singular_values, threshold=0.99):
    cumulative = np.cumsum(singular_values)
    r = np.searchsorted(cumulative, threshold * cumulative[-1]) + 1
    return int(min(r, len(singular_values)))


def hosvd(tensor, energy_threshold=0.99):
    """
    HOSVD (Tucker via mode-wise SVD).
    Returns (core, factors, sv_list).
    """
    factors, sv_list = [], []
    for mode in tqdm(range(tensor.ndim), desc='HOSVD modes'):
        U, S, _ = np.linalg.svd(tl.unfold(tensor, mode), full_matrices=False)
        r = rank_by_energy(S, energy_threshold)
        factors.append(U[:, :])
        sv_list.append(S)
    core = tl.tenalg.multi_mode_dot(
        tensor, [f.T for f in factors], modes=list(range(tensor.ndim))
    )
    return core, factors, sv_list


def run_pod(T_train_s, energy_threshold=0.99, n_modes=None):
    """
    POD via SVD of flattened training matrix.
    T_train_s : (N, Nz, Nx, Nsp)
    n_modes   : if given, use exactly this many modes (overrides energy_threshold)
    Returns (V_pod, S_pod, a_tr) where a_tr : (N, r_pod).
    """
    N = T_train_s.shape[0]
    mat          = T_train_s.reshape(N, -1).T        # (Nz*Nx*Nsp, N)
    V, S, _      = np.linalg.svd(mat, full_matrices=False)
    r            = min(n_modes, V.shape[1]) if n_modes is not None \
                   else rank_by_energy(S, energy_threshold)
    V_pod        = V[:, :r]
    a_tr         = (V_pod.T @ mat).T                 # (N, r)
    return V_pod, S, a_tr


def reconstruct_hosvd(core, factors, alpha_Re_pred, alpha_mf_pred):
    """
    Contract Tucker core with predicted 1D factor rows for modes 0 and 1,
    then expand with spatial/species factors (modes 2, 3, 4).
    Returns (Nz, Nx, Nsp) scaled reconstruction.
    """
    U_z, U_x, U_spec = factors[2], factors[3], factors[4]
    Z = tl.tenalg.mode_dot(core,   alpha_Re_pred, mode=0)
    Z = tl.tenalg.mode_dot(Z,      alpha_mf_pred, mode=0)
    return tl.tenalg.multi_mode_dot(Z, [U_z, U_x, U_spec], modes=[0, 1, 2])


def reconstruct_pod(V_pod, a_pred, Nz, Nx, Nsp):
    """Reconstruct (Nz, Nx, Nsp) scaled field from POD modes and coefficients."""
    return (V_pod @ a_pred).reshape(Nz, Nx, Nsp)


# ─────────────────────────────────────────────────────────────────────────────
# normalisation
# ─────────────────────────────────────────────────────────────────────────────

def minmax(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a, float)


def minmax_scale_point(val, train_vals):
    lo, hi = float(min(train_vals)), float(max(train_vals))
    return (val - lo) / (hi - lo) if hi > lo else 0.0


def standardise_train(T_train, eps=1e-12):
    """
    Compute mean/std over all training cases and spatial points (per species).
    Returns (T_train_s, mu, std) with keepdims shapes (1,1,1,Nsp).
    """
    mu  = T_train.mean(axis=(0, 1, 2), keepdims=True)
    std = T_train.std( axis=(0, 1, 2), keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (T_train - mu) / std, mu, std


# ─────────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────────

def rel_error(pred, true):
    denom = np.linalg.norm(true.ravel())
    return np.linalg.norm((pred - true).ravel()) / denom if denom > 0 else np.nan


def rel_error_vec(pred, true):
    """Relative L2 error between two 1-D vectors."""
    d = np.linalg.norm(true)
    return np.linalg.norm(pred - true) / d if d > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# GPR factories
# ─────────────────────────────────────────────────────────────────────────────

def make_mo_gpr(kernel, n_restarts=3):
    base = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=n_restarts)
    return MultiOutputRegressor(base)


def _make_base_kernel_1d(name, ls):
    if name.startswith('Matern'):
        nu = float(name.split('-')[1])
        return Matern(length_scale=ls, length_scale_bounds='fixed', nu=nu)
    if name == 'RBF':
        return RBF(length_scale=ls, length_scale_bounds='fixed')
    if name.startswith('RatQuad'):
        alpha = float(name.split('-a')[1])
        return RationalQuadratic(length_scale=ls, alpha=alpha,
                                  length_scale_bounds='fixed',
                                  alpha_bounds='fixed')
    raise ValueError(f'Unknown kernel name: {name}')


def _make_base_kernel_2d(name, ls):
    ls2 = np.ones(2) * ls
    if name.startswith('Matern'):
        nu = float(name.split('-')[1])
        return Matern(length_scale=ls2, length_scale_bounds='fixed', nu=nu)
    if name == 'RBF':
        return RBF(length_scale=ls2, length_scale_bounds='fixed')
    if name.startswith('RatQuad'):
        alpha = float(name.split('-a')[1])
        return RationalQuadratic(length_scale=ls, alpha=alpha,
                                  length_scale_bounds='fixed',
                                  alpha_bounds='fixed')
    raise ValueError(f'Unknown kernel name: {name}')


def build_kernel_1d(name, ls):
    return ConstantKernel(1.0) * _make_base_kernel_1d(name, ls)


def build_kernel_2d(name, ls):
    return ConstantKernel(1.0) * _make_base_kernel_2d(name, ls)


KERNEL_NAMES = [
    'Matern-0.5', 'Matern-1.5', 'Matern-2.5',
    'RBF',
    'RatQuad-a1', 'RatQuad-a5',
]
LS_SWEEP = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])

# Fixed kernel family for 1D GPRs: with only 4 training points LOO-3 cannot
# reliably rank kernel families, so we fix Matern-2.5 (smooth, well-behaved
# for interpolation between 4 points) and only tune length_scale.
HOSVD_1D_KERNEL_NAME = 'Matern-2.5'


# ─────────────────────────────────────────────────────────────────────────────
# LOO cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def loo_2d(X_tr, Y_tr, kernel):
    """
    LOO-CV for a 2D multi-output GPR.
    X_tr : (n, 2)   Y_tr : (n, d)
    Returns mean relative L2 error over n folds (n=16 for POD).
    """
    n = len(X_tr)
    errs = []
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        gpr  = make_mo_gpr(kernel)
        gpr.fit(X_tr[mask], Y_tr[mask])
        pred = np.array([est.predict(X_tr[[i]])[0]
                         for est in gpr.estimators_])
        errs.append(rel_error_vec(pred, Y_tr[i]))
    return float(np.nanmean(errs))


def loo_1d_recon(X_tr, U_factor, core, factors, factor_idx,
                 other_factor_row, mu, std, T_grid_phys):
    """
    Reconstruction-based LOO for a 1D HOSVD GPR.

    Holds out one row of U_factor at a time, fits GPR on the remaining 3,
    predicts the held-out row, reconstructs the corresponding slice of the
    physical field, and scores against the true slice.

    Parameters
    ----------
    X_tr            : (4, 1)  normalised axis values
    U_factor        : (4, r)  full factor matrix (U_re or U_mf)
    core            : Tucker core
    factors         : list of all 5 factor matrices
    factor_idx      : 0 for Re axis, 1 for mf axis
    other_factor_row: the *fixed* predicted row for the other parameter axis
                      (used only to keep reconstruction self-consistent;
                       pass the full-fit prediction or the mean row)
    mu, std         : (1,1,1,Nsp) standardisation arrays
    T_grid_phys     : (4, 4, Nz, Nx, Nsp) physical-space grid tensor
                      used to extract the true held-out slice

    Returns
    -------
    mean relative L2 reconstruction error over 4 LOO folds
    """
    n = len(X_tr)
    errs = []
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        kernel = build_kernel_1d(HOSVD_1D_KERNEL_NAME, ls=0.3)
        gpr = make_mo_gpr(kernel)
        gpr.fit(X_tr[mask], U_factor[mask])
        pred_row = np.array([est.predict(X_tr[[i]])[0]
                             for est in gpr.estimators_])

        # reconstruct the held-out slice
        if factor_idx == 0:
            alpha_re, alpha_mf = pred_row, other_factor_row
        else:
            alpha_re, alpha_mf = other_factor_row, pred_row

        recon_s = reconstruct_hosvd(core, factors, alpha_re, alpha_mf)
        recon   = recon_s * std.squeeze() + mu.squeeze()

        # true slice: average over the orthogonal axis to get a comparable field
        if factor_idx == 0:
            true_slice = T_grid_phys[i].mean(axis=0)   # mean over mf
            recon_ref  = recon                          # (Nz, Nx, Nsp)
        else:
            true_slice = T_grid_phys[:, i].mean(axis=0)  # mean over Re
            recon_ref  = recon

        errs.append(rel_error(recon_ref, true_slice))
    return float(np.nanmean(errs))


def search_kernel_1d(X_tr, U_factor, core, factors, factor_idx,
                     other_factor_row, mu, std, T_grid_phys,
                     plot_dir, label):
    """
    Length-scale sweep for a 1D HOSVD GPR using reconstruction-based LOO.
    Kernel family is fixed to HOSVD_1D_KERNEL_NAME (Matern-2.5).

    Returns best_kernel, best_ls.
    """
    print(f'\n── 1D kernel ls-sweep: {label} '
          f'(kernel={HOSVD_1D_KERNEL_NAME}) ──')
    ls_scores = []
    for ls in LS_SWEEP:
        kernel = build_kernel_1d(HOSVD_1D_KERNEL_NAME, ls=ls)
        gpr    = make_mo_gpr(kernel)
        gpr.fit(X_tr, U_factor)
        # use reconstruction LOO to score
        s = loo_1d_recon(X_tr, U_factor, core, factors, factor_idx,
                         other_factor_row, mu, std, T_grid_phys)
        ls_scores.append(s)
        print(f'  ls={ls:.3f}  recon-LOO err = {s:.4f}')

    best_ls = LS_SWEEP[int(np.nanargmin(ls_scores))]
    print(f'  → Best length_scale: {best_ls}')
    _sweep_plot(LS_SWEEP, ls_scores,
                f'{label} — ls sweep ({HOSVD_1D_KERNEL_NAME})',
                plot_dir / f'kernel_sweep_{label}.png',
                best_ls)
    best_kernel = build_kernel_1d(HOSVD_1D_KERNEL_NAME, ls=best_ls)
    return best_kernel, best_ls


def search_kernel_2d(X_tr, Y_tr, plot_dir, label):
    """
    Full kernel search for the POD 2D GPR (16 training points — LOO is reliable).

    Stage 1: LOO over kernel families at fixed ls=0.3.
    Stage 2: LOO ls-sweep for the winning family.

    Returns best_kernel, best_name, best_ls.
    """
    print(f'\n── 2D kernel search: {label} ──')

    # stage 1: family
    scores = []
    for name in KERNEL_NAMES:
        k = build_kernel_2d(name, ls=0.3)
        s = loo_2d(X_tr, Y_tr, k)
        scores.append(s)
        print(f'  {name:15s}  LOO err = {s:.4f}')

    wi        = int(np.nanargmin(scores))
    best_name = KERNEL_NAMES[wi]
    print(f'  → Winner: {best_name}  (err={scores[wi]:.4f})')
    _bar_plot(KERNEL_NAMES, scores,
              f'{label} — kernel LOO search',
              plot_dir / f'kernel_search_{label}.png',
              wi)

    # stage 2: ls sweep
    ls_scores = []
    for ls in LS_SWEEP:
        k = build_kernel_2d(best_name, ls=ls)
        s = loo_2d(X_tr, Y_tr, k)
        ls_scores.append(s)

    best_ls = LS_SWEEP[int(np.nanargmin(ls_scores))]
    print(f'  → Best length_scale: {best_ls}')
    _sweep_plot(LS_SWEEP, ls_scores,
                f'{label} — ls sweep ({best_name})',
                plot_dir / f'kernel_sweep_{label}.png',
                best_ls)

    best_kernel = build_kernel_2d(best_name, ls=best_ls)
    return best_kernel, best_name, best_ls


# ─────────────────────────────────────────────────────────────────────────────
# plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar_plot(names, scores, title, path, winner_idx):
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    colors = ['#4CAF50' if i == winner_idx else '#2196F3'
              for i in range(len(names))]
    ax.bar(names, scores, color=colors, alpha=0.85, edgecolor='k', lw=0.5)
    ax.set_ylabel('Mean LOO relative L2 error')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for i, s in enumerate(scores):
        ax.text(i, s + max(scores) * 0.01, f'{s:.4f}',
                ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=25, ha='right', fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def _sweep_plot(ls_vals, scores, title, path, best_ls):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ls_vals, scores, 'o-', color='#2196F3', lw=1.5, ms=5)
    ax.axvline(best_ls, color='#4CAF50', ls='--', lw=1.2,
               label=f'best ls={best_ls:.3f}')
    ax.set_xlabel('length_scale (normalised)')
    ax.set_ylabel('Mean LOO relative L2 error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_singular_values(sv_list, factors, S_pod, r_pod,
                         n_re_tr, n_mf_tr, Nz, Nx, Nsp, plot_dir):
    mode_names = [
        f'Re ({n_re_tr})', f'mf ({n_mf_tr})',
        f'z ({Nz})', f'x ({Nx})', f'species ({Nsp})',
    ]
    fig, axs = plt.subplots(1, 6, figsize=(20, 3))
    for ax, sv, name, trunc in zip(
            axs[:5], sv_list, mode_names, [f.shape[1] for f in factors]):
        ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], 'o-', ms=3)
        ax.axvline(trunc, color='red', ls='--', label=f'r={trunc}')
        ax.set_title(f'HOSVD mode: {name}')
        ax.set_xlabel('index')
        ax.legend(fontsize=8)
        ax.grid(True)
    axs[5].semilogy(np.arange(1, len(S_pod) + 1), S_pod / S_pod[0],
                    's-', ms=3, color='darkorange')
    axs[5].axvline(r_pod, color='red', ls='--', label=f'r={r_pod}')
    axs[5].set_title('POD (flattened matrix)')
    axs[5].set_xlabel('index')
    axs[5].legend(fontsize=8)
    axs[5].grid(True)
    fig.suptitle('Singular value decay (normalised to σ₁)', y=1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / 'singular_values.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved plots/singular_values.png')


def plot_hosvd_coeffs(alpha_pred, alpha_std, axis_label, test_val,
                      r, plot_dir, filename):
    comp = np.arange(r)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(comp, alpha_pred, 'r--s', ms=5, lw=1.2, label='1D GPR prediction')
    ax.fill_between(comp,
                    alpha_pred - alpha_std,
                    alpha_pred + alpha_std,
                    alpha=0.25, color='red', label='±1 σ')
    ax.set_xlabel(f'{axis_label} component index')
    ax.set_ylabel(f'$U_{{{axis_label}}}$ coefficient')
    ax.set_title(f'HOSVD 1D GPR ({axis_label} axis) — predicted at '
                 f'{axis_label}={test_val}  |  r={r}')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    fig.savefig(plot_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {plot_dir / filename}')


def plot_pod_coeffs(a_true, a_pred, a_std, re_test, mf_test,
                    n_train, plot_dir):
    comp = np.arange(len(a_pred))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(comp, a_true, 'k-o',  ms=5, lw=1.2,
            label='True (projection onto POD modes)')
    ax.plot(comp, a_pred, 'r--s', ms=5, lw=1.2, label='2D GPR prediction')
    ax.fill_between(comp, a_pred - a_std, a_pred + a_std,
                    alpha=0.25, color='red', label='±1 σ')
    ax.set_xlabel('POD mode index')
    ax.set_ylabel('POD coefficient')
    ax.set_title(f'POD 2D GPR — predicted vs true at '
                 f'(Re={re_test}, mf={mf_test})\n'
                 f'trained on {n_train} conditions (4×4 grid)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    fig.savefig(plot_dir / 'gpr_coeffs_pod.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved plots/gpr_coeffs_pod.png')


def plot_error_bars(errors_h, errors_p, fields, r_re, r_mf, r_pod,
                    re_test, mf_test, plot_dir):
    x = np.arange(len(fields))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, errors_h, w, color='#2196F3', alpha=0.85,
           label=(f'HOSVD 4×4 + 2×1D GPR  '
                  f'(r_re={r_re}, r_mf={r_mf}, '
                  f'mean={np.nanmean(errors_h):.4f})'))
    ax.bar(x + w/2, errors_p, w, color='#FF5722', alpha=0.85,
           label=(f'POD + 2D GPR  '
                  f'(r_pod={r_pod}, mean={np.nanmean(errors_p):.4f})'))
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha='right')
    ax.set_ylabel('Relative L2 error')
    ax.set_title(f'Per-feature relative error — (Re={re_test}, mf={mf_test})\n'
                 f'Both methods trained on same 16 cases (4×4 grid)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(plot_dir / 'relative_error_per_feat.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved plots/relative_error_per_feat.png')


def plot_field(name, T_test_true, recon_h, recon_p, col_idx,
               extent, suptitle_sub, plot_dir):
    sp        = col_idx[name]
    F_true    = T_test_true[:, :, sp]
    F_hosvd   = recon_h    [:, :, sp]
    F_pod     = recon_p    [:, :, sp]
    err_h_fld = F_hosvd - F_true
    err_p_fld = F_pod   - F_true

    vmin_f  = min(F_true.min(),  F_hosvd.min(),  F_pod.min())
    vmax_f  = max(F_true.max(),  F_hosvd.max(),  F_pod.max())
    err_lim = max(np.abs(err_h_fld).max(), np.abs(err_p_fld).max())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=120)

    for ax, title, field in zip(
            axes[0],
            ['Original', 'HOSVD 4×4 + 2×1D GPR', 'POD + 2D GPR'],
            [F_true, F_hosvd, F_pod]):
        im = ax.imshow(field, origin='lower', aspect='auto', extent=extent,
                       vmin=vmin_f, vmax=vmax_f, cmap='hot')
        ax.set_title(title); ax.set_xlabel('r'); ax.set_ylabel('z')
        plt.colorbar(im, ax=ax, label=name)

    axes[1, 0].hist(err_h_fld.ravel(), bins=80, alpha=0.7,
                    color='#2196F3', density=True, label='HOSVD')
    axes[1, 0].hist(err_p_fld.ravel(), bins=80, alpha=0.7,
                    color='#FF5722', density=True, label='POD')
    axes[1, 0].axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    axes[1, 0].set_xlabel(f'Δ{name}'); axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error distributions')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)

    im_eh = axes[1, 1].imshow(err_h_fld, origin='lower', aspect='auto',
                               extent=extent, vmin=-err_lim, vmax=err_lim,
                               cmap='RdBu_r')
    axes[1, 1].set_title('Spatial error — HOSVD')
    axes[1, 1].set_xlabel('r'); axes[1, 1].set_ylabel('z')
    plt.colorbar(im_eh, ax=axes[1, 1], label=f'Δ{name}')

    im_ep = axes[1, 2].imshow(err_p_fld, origin='lower', aspect='auto',
                               extent=extent, vmin=-err_lim, vmax=err_lim,
                               cmap='RdBu_r')
    axes[1, 2].set_title('Spatial error — POD')
    axes[1, 2].set_xlabel('r'); axes[1, 2].set_ylabel('z')
    plt.colorbar(im_ep, ax=axes[1, 2], label=f'Δ{name}')

    fig.suptitle(f'{name} — {suptitle_sub}', fontsize=11)
    plt.tight_layout()
    fig.savefig(plot_dir / f'field_{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plots/field_{name}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Kronecker GPR
# ─────────────────────────────────────────────────────────────────────────────

def _kernel_matrix(kernel, X):
    """Evaluate kernel matrix K(X, X) using sklearn kernel."""
    return kernel(X.reshape(-1, 1))


def _kernel_vector(kernel, x_star, X_tr):
    """Evaluate kernel vector k(x*, X_tr)."""
    return kernel(np.array([[x_star]]), X_tr.reshape(-1, 1)).ravel()


class KroneckerGPR:
    """
    Kronecker-structured GPR for data on a regular (Re, mf) grid.

    The joint kernel factorises as:
        k((Re,mf),(Re',mf')) = k_Re(Re,Re') * k_mf(mf,mf')

    This allows exact inference in O(n_re^3 + n_mf^3) instead of O((n_re*n_mf)^3)
    via the eigendecomposition trick:
        K = K_Re ⊗ K_mf  =>  (K + σ²I)^{-1} = (Q_Re ⊗ Q_mf) diag(1/(λ_i*λ_j+σ²)) (Q_Re ⊗ Q_mf)^T

    Parameters
    ----------
    kernel_re   : sklearn kernel for the Re axis (1D)
    kernel_mf   : sklearn kernel for the mf axis (1D)
    noise_var   : observation noise variance σ²
    normalize_y : subtract mean of training targets before fitting

    Usage
    -----
    kgpr = KroneckerGPR(kernel_re, kernel_mf)
    kgpr.fit(Re_tr_nm, Mf_tr_nm, C_train)   # C_train : (n_re, n_mf) or (n_re, n_mf, d)
    mean, std = kgpr.predict(Re_test_nm, Mf_test_nm)
    """

    def __init__(self, kernel_re, kernel_mf, noise_var=1e-6, normalize_y=True):
        self.kernel_re   = kernel_re
        self.kernel_mf   = kernel_mf
        self.noise_var   = noise_var
        self.normalize_y = normalize_y

    def fit(self, Re_tr_nm, Mf_tr_nm, C_train):
        """
        Parameters
        ----------
        Re_tr_nm : (n_re,)  normalised Re training values
        Mf_tr_nm : (n_mf,)  normalised mf training values
        C_train  : (n_re, n_mf) or (n_re, n_mf, d)
                   Tucker coefficient matrix/tensor at each grid point.
                   For d>1 each output is treated independently (same kernel).
        """
        self.Re_tr = Re_tr_nm
        self.Mf_tr = Mf_tr_nm
        n_re, n_mf = len(Re_tr_nm), len(Mf_tr_nm)

        # handle multi-output: reshape to (n_re, n_mf, d)
        scalar = C_train.ndim == 2
        if scalar:
            C_train = C_train[:, :, np.newaxis]
        self._scalar = scalar
        self._d = C_train.shape[2]

        # normalise targets per output
        if self.normalize_y:
            self._ymean = C_train.mean(axis=(0, 1), keepdims=True)  # (1,1,d)
        else:
            self._ymean = np.zeros((1, 1, C_train.shape[2]))
        Y = C_train - self._ymean   # (n_re, n_mf, d)

        # kernel matrices
        K_re = _kernel_matrix(self.kernel_re, Re_tr_nm)   # (n_re, n_re)
        K_mf = _kernel_matrix(self.kernel_mf, Mf_tr_nm)   # (n_mf, n_mf)

        # eigendecompositions
        lam_re, Q_re = np.linalg.eigh(K_re)   # (n_re,), (n_re, n_re)
        lam_mf, Q_mf = np.linalg.eigh(K_mf)   # (n_mf,), (n_mf, n_mf)

        # Kronecker eigenvalues: λ_ij = λ_re_i * λ_mf_j
        lam_kron = np.outer(lam_re, lam_mf)           # (n_re, n_mf)
        D_inv    = 1.0 / (lam_kron + self.noise_var)  # (n_re, n_mf)

        # store for prediction
        self._Q_re  = Q_re
        self._Q_mf  = Q_mf
        self._D_inv = D_inv
        self._lam_re = lam_re
        self._lam_mf = lam_mf

        # alpha = (K + σ²I)^{-1} y  via Kronecker trick, per output
        # alpha[i,j,t] = sum_{p,q} Q_re[i,p] Q_mf[j,q] D_inv[p,q] (Q_re^T Y Q_mf)[p,q,t]
        # compute QY = Q_re^T @ Y @ Q_mf  for each output
        QY = np.einsum('ip,pqd,jq->ijd', Q_re.T,
                       np.einsum('ip,pqd->iqd', Q_re.T, Y),   # Q_re^T @ Y per output... rewrite:
                       Q_mf)
        # simpler: QY[p,q,d] = (Q_re.T @ Y[:,:,d] @ Q_mf)[p,q]
        QY = np.stack([Q_re.T @ Y[:, :, t] @ Q_mf
                       for t in range(self._d)], axis=2)       # (n_re, n_mf, d)

        # alpha in rotated space
        self._alpha_rot = QY * D_inv[:, :, np.newaxis]         # (n_re, n_mf, d)

    def predict(self, re_star, mf_star):
        """
        Predict at a single new point (re_star, mf_star) — both normalised.

        Returns
        -------
        mean : (d,) or scalar   posterior mean
        std  : (d,) or scalar   posterior std (marginal)
        """
        k_re = _kernel_vector(self.kernel_re, re_star, self.Re_tr)  # (n_re,)
        k_mf = _kernel_vector(self.kernel_mf, mf_star, self.Mf_tr)  # (n_mf,)

        # posterior mean: k_re^T @ alpha @ k_mf  (in original space)
        # = (Q_re^T k_re)^T diag(D_inv) (Q_mf^T k_mf) contracted with alpha_rot
        qk_re = self._Q_re.T @ k_re   # (n_re,)
        qk_mf = self._Q_mf.T @ k_mf   # (n_mf,)

        # mean[t] = sum_{p,q} qk_re[p] * alpha_rot[p,q,t] * qk_mf[q]
        mean_rot = np.einsum('p,pqd,q->d', qk_re, self._alpha_rot, qk_mf)
        mean = mean_rot + self._ymean.squeeze(axis=(0, 1))   # add back mean

        # posterior variance: k** - k_*^T (K+σ²I)^{-1} k_*
        k_ss_re = float(self.kernel_re(np.array([[re_star]]))[0, 0])
        k_ss_mf = float(self.kernel_mf(np.array([[mf_star]]))[0, 0])
        k_star_star = k_ss_re * k_ss_mf

        # k_*^T (K+σ²I)^{-1} k_* = (qk_re ⊗ qk_mf)^T D_inv (qk_re ⊗ qk_mf)
        #                          = sum_{p,q} qk_re[p]^2 * D_inv[p,q] * qk_mf[q]^2
        var = k_star_star - float(np.einsum('p,pq,q', qk_re**2, self._D_inv, qk_mf**2))
        var = max(var, 0.0)
        std_scalar = float(np.sqrt(var))

        if self._scalar:
            return float(mean[0]), std_scalar
        return mean, np.full(self._d, std_scalar)


def fit_kronecker_gpr(Re_tr_nm, Mf_tr_nm, core, factors,
                      kernel_re, kernel_mf, noise_var=1e-6):
    """
    Build and fit one KroneckerGPR per Tucker coefficient pair (p, q).

    The training targets for pair (p,q) are the (n_re x n_mf) matrix of
    Tucker coefficients:
        C[i,j] = core[p,q,...] scalar — but since we want to predict the
                 full contracted field, we actually regress the core slice
                 core[p, q, :, :, :] weighted by each (U_re[i,p], U_mf[j,q]).

    In practice we regress the outer-product coefficient:
        C[i,j] = U_re[i,p] * U_mf[j,q]   for each (p,q)

    and at prediction time reconstruct:
        alpha_Re[p] * alpha_mf[q] ≈ predicted C_pq(Re*, mf*)

    Parameters
    ----------
    Re_tr_nm, Mf_tr_nm : normalised 1D training axes
    core               : Tucker core (r_re, r_mf, r_z, r_x, r_sp)
    factors            : list of factor matrices
    kernel_re, kernel_mf : sklearn kernels (1D each)

    Returns
    -------
    kgpr_list : list of KroneckerGPR, one per (p,q) pair
                indexed as kgpr_list[p * r_mf + q]
    r_re, r_mf : ranks
    """
    U_re, U_mf = factors[0], factors[1]
    r_re, r_mf = U_re.shape[1], U_mf.shape[1]

    kgpr_list = []
    for p in range(r_re):
        for q in range(r_mf):
            # training targets: outer product of factor rows at each grid point
            C_pq = np.outer(U_re[:, p], U_mf[:, q])   # (n_re, n_mf)
            kgpr = KroneckerGPR(kernel_re, kernel_mf,
                                noise_var=noise_var, normalize_y=True)
            kgpr.fit(Re_tr_nm, Mf_tr_nm, C_pq)
            kgpr_list.append(kgpr)

    return kgpr_list, r_re, r_mf


def predict_kronecker_gpr(kgpr_list, r_re, r_mf, re_star, mf_star):
    """
    Predict Tucker coefficient outer product at (re_star, mf_star).

    Returns
    -------
    alpha_Re : (r_re,)  effective Re factor row
    alpha_mf : (r_mf,)  effective mf factor row
    std      : scalar   mean posterior std across all (p,q) pairs
    """
    C_pred = np.zeros((r_re, r_mf))
    stds   = []
    for p in range(r_re):
        for q in range(r_mf):
            idx = p * r_mf + q
            val, s = kgpr_list[idx].predict(re_star, mf_star)
            C_pred[p, q] = float(val)
            stds.append(s)

    # extract effective factor rows via SVD of predicted coefficient matrix
    # C_pred ≈ α_Re ⊗ α_mf  — best rank-1 approximation
    U, S, Vt = np.linalg.svd(C_pred, full_matrices=False)
    alpha_Re = U[:, 0] * S[0]
    alpha_mf = Vt[0, :]

    return alpha_Re, alpha_mf, float(np.mean(stds))


def reconstruct_kronecker(core, factors, kgpr_list, r_re, r_mf,
                          re_star, mf_star):
    """
    Full reconstruction pipeline using Kronecker GPR.
    Returns (Nz, Nx, Nsp) scaled reconstruction and mean std.
    """
    alpha_Re, alpha_mf, std_k = predict_kronecker_gpr(
        kgpr_list, r_re, r_mf, re_star, mf_star
    )
    recon_s = reconstruct_hosvd(core, factors, alpha_Re, alpha_mf)
    return recon_s, alpha_Re, alpha_mf, std_k


# ─────────────────────────────────────────────────────────────────────────────
# Residual GPR correction for Kronecker+Tucker
# ─────────────────────────────────────────────────────────────────────────────

def fit_residual_gpr(params_train, T_train, core, factors,
                     re_to_idx_tr, mf_to_idx_tr,
                     mu, std, kernel_2d,
                     energy_threshold=0.99):
    """
    Fit a POD+GPR surrogate on the Tucker reconstruction residuals.

    For each training case the Tucker in-sample reconstruction is computed
    using the exact factor rows (no GPR involved — pure Tucker approximation).
    The residual is:
        R_k = T_train_scaled[k] - Tucker_recon_scaled[k]

    These 16 residual fields are collected into a matrix, compressed via SVD
    (POD on residuals), and a 2D GPR is fit to predict residual coefficients
    from (Re, mf).

    Parameters
    ----------
    params_train    : (N, 2)   training (Re, mf) pairs
    T_train         : (N, Nz, Nx, Nsp)  physical training fields
    core, factors   : Tucker decomposition from HOSVD
    re_to_idx_tr    : dict  Re  -> grid row index
    mf_to_idx_tr    : dict  round(mf,2) -> grid col index
    mu, std         : (1,1,1,Nsp) standardisation arrays
    kernel_2d       : sklearn kernel for the residual 2D GPR
    energy_threshold: SVD truncation threshold for residual modes

    Returns
    -------
    res_V    : (Nz*Nx*Nsp, r_res)  residual spatial modes
    res_a_tr : (N, r_res)          residual coefficients per training case
    res_gpr  : fitted MultiOutputRegressor (2D GPR on residual coefficients)
    r_res    : int  number of residual modes retained
    """
    N   = len(params_train)
    U_re, U_mf = factors[0], factors[1]

    # ── 1. compute Tucker in-sample residuals (scaled space) ──────────────
    T_train_s = (T_train - mu) / std                 # (N, Nz, Nx, Nsp)
    Nz, Nx, Nsp = T_train_s.shape[1:]

    residuals = np.zeros_like(T_train_s)             # (N, Nz, Nx, Nsp)
    for k, (re, mf) in enumerate(params_train):
        i = re_to_idx_tr[int(re)]
        j = mf_to_idx_tr[round(mf, 2)]
        alpha_re = U_re[i]                           # exact row — no GPR
        alpha_mf = U_mf[j]
        recon_s  = reconstruct_hosvd(core, factors, alpha_re, alpha_mf)
        residuals[k] = T_train_s[k] - recon_s

    # ── 2. POD on residuals ───────────────────────────────────────────────
    mat_res = residuals.reshape(N, -1).T             # (Nz*Nx*Nsp, N)
    V_res, S_res, _ = np.linalg.svd(mat_res, full_matrices=False)
    r_res    = rank_by_energy(S_res, energy_threshold)
    res_V    = V_res[:, :r_res]                      # (Nz*Nx*Nsp, r_res)
    res_a_tr = (res_V.T @ mat_res).T                 # (N, r_res)

    res_norms = np.array([np.linalg.norm(residuals[k]) for k in range(N)])
    print(f'  Residual POD rank: {r_res}  '
          f'(max residual norm: {res_norms.max():.4f})')

    # ── 3. 2D GPR on residual coefficients ───────────────────────────────
    from sklearn.preprocessing import StandardScaler
    param_scaler = StandardScaler().fit(params_train)
    P_train_s    = param_scaler.transform(params_train)

    res_gpr = make_mo_gpr(kernel_2d)
    res_gpr.fit(P_train_s, res_a_tr)

    return res_V, res_a_tr, res_gpr, param_scaler, r_res, S_res


def predict_residual_gpr(res_V, res_gpr, param_scaler,
                         re_test, mf_test, Nz, Nx, Nsp):
    """
    Predict the residual field at a new (Re, mf) point and return it
    in scaled space (same space as Tucker reconstruction).

    Returns
    -------
    residual_pred_s : (Nz, Nx, Nsp)  predicted residual in scaled space
    a_res_pred      : (r_res,)        predicted residual coefficients
    a_res_std       : (r_res,)        posterior std per coefficient
    """
    P_test_s  = param_scaler.transform([[re_test, mf_test]])
    preds     = [est.predict(P_test_s, return_std=True)
                 for est in res_gpr.estimators_]
    a_res_pred = np.array([p[0].item() for p in preds])
    a_res_std  = np.array([p[1].item() for p in preds])

    residual_pred_s = (res_V @ a_res_pred).reshape(Nz, Nx, Nsp)
    return residual_pred_s, a_res_pred, a_res_std