"""Plot all conditions and seeds from the audited confirmation report."""
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path.cwd() / 'work/reports/completion-confirmation-v1'
    analysis = json.loads((folder / 'analysis.json').read_text())
    if analysis['status'] != 'all nine frozen runs audited':
        raise ValueError('Audit required')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    seeds = sorted({row['seed'] for row in analysis['per_seed']})
    for seed, color in zip(seeds, ('#335c81', '#d97706', '#34856a')):
        rows = {row['rotating_steps']: row for row in analysis['per_seed'] if row['seed'] == seed}
        axes[0].plot([0, 1, 2], [rows[8]['macro_f1_all'], rows[8]['macro_f1_rotating'],
                               rows[32]['macro_f1_rotating']], 'o-', color=color, label=f'Seed {seed}')
        for ax, field in ((axes[1], 'optimizer_fraction'), (axes[2], 'mlx_peak_fraction')):
            ax.plot([1, 2], [rows[8][field], rows[32][field]], 'o-', color=color)
    axes[0].set_ylabel('Reserved-test macro-F1 (240 shared cases)')
    axes[0].set_title('Quality varies by schedule and seed')
    axes[0].set_xticks([0, 1, 2], ['All / 8', 'Rotate / 8', 'Rotate / 32'])
    axes[0].legend(fontsize=8)
    axes[1].set_title('Resident optimizer state')
    axes[2].set_title('Peak MLX allocation')
    for ax in axes[1:]:
        ax.axhline(1, color='#777777', linestyle='--', linewidth=1)
        ax.set_ylabel('Fraction of paired All / 8 run')
        ax.set_xticks([1, 2], ['Rotate / 8', 'Rotate / 32'])
        ax.set_xlim(.8, 2.2)
        ax.set_ylim(0, 1.1)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Original BF16 Qwen3-0.6B on M1 / 16 GiB: frozen local confirmation', fontsize=12)
    fig.text(.5, .025, 'Rotate / 32 matches per-layer updates but consumes 4x the examples. Short single-corpus study; no novelty claim.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(folder / 'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
