"""
plot_max_fields.py
==================
For each field in IMPORTANT_FIELDS, plot a heatmap over the (Re, mf) grid
where each cell shows the maximum value of that field in that simulation.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import COLS, COL_IDX, load_case

# ─────────────────────────────────────────────────────────────────────────────
# configuration
# ─────────────────────────────────────────────────────────────────────────────

CASES_DIR = Path('/home/isacco/DATA/ZOU_GPR/Datasets_Isacco')
PLOT_DIR  = Path('plots/max_fields')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RE_VALS = [11000, 13000, 15000, 17000, 19000]
MF_VALS = [0.04,  0.08,  0.12,  0.16,  0.20]

IMPORTANT_FIELDS = ['T', 'CH4', 'O2', 'CO2', 'H2O']

# ─────────────────────────────────────────────────────────────────────────────
# load max value per field per case
# ─────────────────────────────────────────────────────────────────────────────

n_re, n_mf = len(RE_VALS), len(MF_VALS)
max_vals = {name: np.zeros((n_re, n_mf)) for name in IMPORTANT_FIELDS}

for i, re in enumerate(RE_VALS):
    for j, mf in enumerate(MF_VALS):
        path = next(CASES_DIR.glob(f'*_mfH2_{mf:.2f}_Re_{int(re)}.xy'))
        grid, _, _ = load_case(path)          # (Nz, Nx, Nsp)
        for name in IMPORTANT_FIELDS:
            max_vals[name][i, j] = grid[:, :, COL_IDX[name]].mean()
        print(f'  Re={re}, mf={mf:.2f} done')

# ─────────────────────────────────────────────────────────────────────────────
# plot one heatmap per field
# ─────────────────────────────────────────────────────────────────────────────

for name in IMPORTANT_FIELDS:
    data = max_vals[name]   # (n_re, n_mf)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, origin='lower', aspect='auto', cmap='viridis')

    ax.set_xticks(range(n_mf)); ax.set_xticklabels([f'{v:.2f}' for v in MF_VALS])
    ax.set_yticks(range(n_re)); ax.set_yticklabels(RE_VALS)
    ax.set_xlabel('mf'); ax.set_ylabel('Re')
    ax.set_title(f'Max {name} over (Re, mf) grid')
    plt.colorbar(im, ax=ax, label=f'max({name})')

    # annotate each cell with its value
    for i in range(n_re):
        for j in range(n_mf):
            ax.text(j, i, f'{data[i, j]:.3g}',
                    ha='center', va='center', fontsize=7,
                    color='white' if data[i, j] < 0.6 * data.mean() else 'black')

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f'max_{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plots/max_fields/max_{name}.png')

print('\nDone.')