"""
visualize.py — plot raw fields for one or all (Re, mf) cases.

Single case:   python visualize.py 15000 0.10
Sweep all:     python visualize.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import COL_IDX, load_case

CASES_DIR = Path('/home/isacco/DATA/Datasets_Isacco')
OUT_DIR   = Path('plots/cases')

FIELDS = ['T', 'CH4', 'O2', 'CO', 'CO2', 'H2O']

n_cols = 3
n_rows = (len(FIELDS) + n_cols - 1) // n_cols


def compute_global_limits(paths):
    """Return {field: (vmin, vmax)} across all cases."""
    limits = {name: [float('inf'), float('-inf')] for name in FIELDS}
    for p in paths:
        grid, _, _ = load_case(p)
        for name in FIELDS:
            f = grid[:, :, COL_IDX[name]]
            limits[name][0] = min(limits[name][0], f.min())
            limits[name][1] = max(limits[name][1], f.max())
    return {name: tuple(v) for name, v in limits.items()}


def plot_case(path, out_dir, limits):
    grid, x_vals, z_vals = load_case(path)

    stem = path.stem
    mf_str = stem.split('_mfH2_')[1].split('_Re_')[0]
    re_str = stem.split('_Re_')[1]
    RE, MF = int(re_str), float(mf_str)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten()

    for i, name in enumerate(FIELDS):
        ax = axs[i]
        vmin, vmax = limits[name]
        lvls = plt.matplotlib.ticker.MaxNLocator(nbins=20).tick_values(vmin, vmax)
        im = ax.contourf(x_vals, z_vals, grid[:, :, COL_IDX[name]],
                         levels=lvls, cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        plt.colorbar(im, ax=ax)

    for j in range(len(FIELDS), len(axs)):
        axs[j].axis('off')

    fig.suptitle(f'Re={RE}, mf={MF}', fontsize=13)
    plt.tight_layout()
    out_path = out_dir / f'Re{RE}_mf{mf_str.replace(".", "")}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


if len(sys.argv) == 3:
    RE = int(sys.argv[1])
    MF = float(sys.argv[2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(CASES_DIR.glob('*.xy'))
    print(f'Computing global color limits across {len(paths)} cases...')
    limits = compute_global_limits(paths)
    plot_case(next(CASES_DIR.glob(f'*_mfH2_{MF:.2f}_Re_{RE}.xy')), OUT_DIR, limits)
else:
    paths = sorted(CASES_DIR.glob('*.xy'))
    print(f'Computing global color limits across {len(paths)} cases...')
    limits = compute_global_limits(paths)
    print(f'Sweeping {len(paths)} cases → {OUT_DIR}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in paths:
        plot_case(p, OUT_DIR, limits)
    print('Done.')
