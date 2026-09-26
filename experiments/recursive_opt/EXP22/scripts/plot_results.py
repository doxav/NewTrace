"""Plot saved strict evidence without importing or changing the execution runtime."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'SD-EVOX': '#2369b5', 'SD-FIXED': '#77a9d9', 'TRACE-RECURSIVE': '#b54c28', 'TRACE-FIXED': '#e4a079'}


def main() -> None:
    """Export quality and cumulative compute with policy-switch markers."""
    report = json.loads((ROOT/'artifacts/diagnostic_summary.json').read_text())
    for task in ('prism', 'signal_processing'):
        runs = [row for row in report['attempts'] if row['config']['stage'] == 'strict' and row['config']['task'] == task]
        if not runs:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for row in runs:
            arm = row['config']['arm']
            m = row['metrics']; curve = m['curve']
            if not curve:
                continue
            color = COLORS[arm]
            x = [point['iteration'] for point in curve]
            label = arm + ('' if m['complete'] else ' (partial)')
            axes[0, 0].step([0, *x], [m['initial_score'], *[point['best_score'] for point in curve]], where='post', color=color, label=label)
            for event in m['policy_events']:
                if event['activated']:
                    iteration = event['iteration']
                    score = curve[iteration - 1]['best_score'] if iteration else m['initial_score']
                    axes[0, 0].scatter([iteration], [score], marker='D', color=color, s=30, zorder=3)
            usage = [point['llm_usage'] for point in curve]
            final_attempt = m['usage']['roles'].get('solution', x[-1])
            axes[0, 1].step(x, [u['roles'].get('solution', 0) for u in usage], where='post', color=color, label=arm+' solution')
            axes[0, 1].step(x, [u['roles'].get('meta', 0) for u in usage], where='post', color=color, linestyle='--', label=arm+' meta')
            for role in ('solution', 'meta'):
                axes[0, 1].scatter([final_attempt], [m['usage']['roles'].get(role, 0)], color=color, s=16)
            axes[1, 0].step(x, [(u['input_tokens'] + u['output_tokens']) / 1000 for u in usage], where='post', color=color, label=label)
            axes[1, 1].step(x, [u['reported_cost'] for u in usage], where='post', color=color, label=label)
            # Include any guide/meta work after the final solution boundary.
            axes[1, 0].scatter([final_attempt], [(m['usage']['input_tokens'] + m['usage']['output_tokens']) / 1000], color=color, s=16)
            axes[1, 1].scatter([final_attempt], [m['usage']['reported_cost']], color=color, s=16)
        titles = ('Best native score (diamonds: policy switches)', 'Cumulative calls (solid: solution; dashed: meta)', 'Cumulative input + output tokens', 'Cumulative provider-reported cost')
        labels = ('Combined score', 'HTTP calls', 'Thousands of tokens', 'USD')
        for axis, title, label in zip(axes.flat, titles, labels):
            axis.set_title(title, fontsize=10)
            axis.set_xlabel('Solution attempt')
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            axis.set_xlim(left=0, right=100)
        axes[0, 0].legend(fontsize=8, loc='best')
        fig.suptitle(f"EXP22 · {task.replace('_', ' ').title()} · Novita / low\nSingle stochastic run per arm · {report['status']}", fontsize=13)
        for extension in ('png', 'svg'):
            path = ROOT/f'artifacts/{task}_strict_curves.{extension}'
            fig.savefig(path, dpi=160)
            if extension == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)
    (ROOT/'artifacts/plot_environment.json').write_text(json.dumps({'matplotlib': matplotlib.__version__, 'backend': matplotlib.get_backend()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
