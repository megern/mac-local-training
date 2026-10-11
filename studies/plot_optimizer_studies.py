"""Static scientific figure of all paired seed blocks, including unfavorable runs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--kind', choices=['endpoint', 'normalization', 'recomputation'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(args.analysis.read_text())
    normalization = args.kind != 'endpoint'
    candidate = 'recomputed' if args.kind == 'recomputation' else 'normalized'
    elapsed_key = f'elapsed_{candidate}_over_endpoint' if normalization else 'elapsed_endpoint_over_fixed'
    memory_key = f'peak_{candidate}_over_endpoint' if normalization else 'peak_endpoint_over_fixed'
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), layout='constrained')
    for index, comparison in enumerate(data['comparisons']):
        rows = comparison['paired_seed_blocks']
        positions = list(range(len(rows)))
        labels = [str(row['seed']) for row in rows]
        for column, key, label in [(0, elapsed_key, 'Elapsed ratio'), (1, memory_key, 'Peak MLX allocation ratio')]:
            axis = axes[index, column]
            axis.plot(positions, [row[key] for row in rows], 'o-', color='#24788c')
            axis.axhline(1.0, color='#686868', ls='--', lw=1, label='No change')
            axis.set_xticks(positions, labels)
            axis.set_xlabel('Paired seed block')
            axis.set_ylabel(label)
            axis.set_title(f"Rank {comparison['rank']} / {label.lower()}")
            axis.grid(alpha=.2)
            if column == 0:
                stats = comparison['elapsed']
                lo, hi = stats['paired_bootstrap_95_ci']
                axis.text(.02, .97, f"Geometric mean {stats['geometric_mean_ratio']:.3f}\n95% paired bootstrap [{lo:.3f}, {hi:.3f}]", transform=axis.transAxes, va='top', fontsize=9,
                          bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
            if column == 1:
                axis.ticklabel_format(axis='y', style='plain', useOffset=False)
                values = [row[key] for row in rows]
                axis.set_ylim(min(.90, min(values) - .01), max(1.10, max(values) + .01))
            if column == 1 and normalization:
                axis.axhline(.97, color='#c37b30', ls=':', lw=1, label='3% reduction gate')
                axis.legend(fontsize=8, loc='lower left')
    ratio_label = {'normalization': 'group-local normalization / endpoint baseline', 'recomputation': 'deferred recomputation / endpoint baseline', 'endpoint': 'endpoint retention / fixed-order offload'}[args.kind]
    fig.suptitle(f"Original Qwen3 0.6B BF16 on M1 16 GiB\n{ratio_label}; lower ratios are better", fontsize=12)
    fig.supxlabel('Six paired blocks per rank; shared machine; four measured updates after two warmups. No task-quality or 70B fit claim.', fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
