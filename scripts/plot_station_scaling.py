"""
Plot the station-scaling benchmark (scripts/benchmark_station_scaling.py).

Each argument is a run folder written by the benchmark (it contains
results.csv and benchmark_args.yaml). All runs must use the same query mode
(fixed or proportional), which is read from the OAT run and used in the
figure names.

Figures (PNG at 300 dpi and PDF, in --figure-dir), for the step chosen with
--step (train or inference):
  - station_scaling_<query_mode>_<step>: OAT and its KNN search vs GP (exact)
    and GP (iterative).
  - fullattention_vs_sparse_<query_mode>_<step> (only with
    --oat-full-attention): OAT vs OAT with full attention.

Usage (from the repository root):
    python -m scripts.plot_station_scaling \
        --oat benchmarking/scaling_benchmark/<date>_oat_fixed_B1 \
        --gp-exact benchmarking/scaling_benchmark/<date>_gp_oat_fixed_B1_exact \
        --gp-iterative benchmarking/scaling_benchmark/<date>_gp_oat_fixed_B1 \
        --oat-full-attention benchmarking/scaling_benchmark/<date>_oat_full_attention_fixed_B1
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import yaml

RC = {
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

DEFAULT_FIGURE_DIR = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'figures'


def plot_scaling(ax, experiment_path, label, color, step='train', column='mean_s',
                 linestyle='-', marker_face=None, marker='o'):

    table = pd.read_csv(pathlib.Path(experiment_path) / 'results.csv')
    step_rows = table[table['step'] == step]
    rows = step_rows[step_rows['status'] == 'ok']
    ax.plot(rows['n_obs'], rows[column], marker=marker, ms=4, lw=1.2, color=color,
            linestyle=linestyle, markerfacecolor=marker_face or color, label=label)

    # Mark the first N_obs that ran out of memory.
    oom_rows = step_rows[step_rows['status'] == 'oom']
    if len(oom_rows) > 0 and len(rows) > 0:
        n_last = rows['n_obs'].iloc[-1]
        y_last = rows[column].iloc[-1]
        n_oom = oom_rows['n_obs'].iloc[0]
        # Dotted line from the last measured point to the OOM size, at the same height.
        ax.plot([n_last, n_oom], [y_last, y_last], ls=':', lw=1.2, color=color)
        ax.plot(n_oom, y_last, marker='x', ms=6, mew=1.5, ls='none', color=color)
        ax.annotate('OOM', (n_oom, y_last), xytext=(4, 4), textcoords='offset points',
                    fontsize=8.5, color='0.3')


def plot_oat_vs_gp(oat_results_path, gp_exact_results_path, gp_iterative_results_path,
                   query_mode, step, figure_dir):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.5, 2.8))

        column = 'mean_s'
        plot_scaling(ax, oat_results_path, label='OAT', color='#0072B2', step=step, column=column)
        plot_scaling(ax, oat_results_path, label="OAT's KNN search", color='#0072B2',
                     step='knn', column=column, linestyle='--', marker_face='white')
        plot_scaling(ax, gp_exact_results_path, label='GP (exact)', color='#009E73', step=step, column=column)
        plot_scaling(ax, gp_iterative_results_path, label='GP (iterative)', color='#009E73', step=step, column=column, marker='^')

        ax.set_xscale('log', base=2) # Note that we expect the number of stations to be powers of 2.
        ax.set_yscale('log')
        ax.set_xlabel('Number of observed stations')
        ax.set_ylabel(f'Time per {step} step [s]')
        ax.grid(True, which='major', lw=0.5, color='0.85')
        ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22))

        name = f'station_scaling_{query_mode}_{step}'
        fig.savefig(figure_dir / f'{name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(figure_dir / f'{name}.pdf', bbox_inches='tight')

        plt.close(fig)


def plot_full_attention(oat_results_path, oat_full_attention_results_path, query_mode, step,
                        figure_dir):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.5, 2.8))

        column = 'mean_s'
        plot_scaling(ax, oat_results_path, label='OAT', color='#0072B2', step=step, column=column)
        plot_scaling(ax, oat_full_attention_results_path, label='OAT with full attention', color='#0072B2', step=step, column=column, linestyle='--', marker_face='white')

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of observed stations')
        ax.set_ylabel(f'Time per {step} step [s]')
        ax.grid(True, which='major', lw=0.5, color='0.85')
        ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22))

        name = f'fullattention_vs_sparse_{query_mode}_{step}'
        fig.savefig(figure_dir / f'{name}.png', dpi=300, bbox_inches='tight')
        fig.savefig(figure_dir / f'{name}.pdf', bbox_inches='tight')

        plt.close(fig)


def read_query_mode(results_path):
    with open(pathlib.Path(results_path) / 'benchmark_args.yaml') as file:
        return yaml.safe_load(file)['query_mode']


def main():
    parser = argparse.ArgumentParser(description='Plot the station-scaling benchmark.')
    parser.add_argument('--oat', type=pathlib.Path, required=True, help='OAT run folder.')
    parser.add_argument('--gp-exact', type=pathlib.Path, required=True,
                        help='GP run folder with --gp-exact-cholesky.')
    parser.add_argument('--gp-iterative', type=pathlib.Path, required=True,
                        help='GP run folder with the default (iterative) inference.')
    parser.add_argument('--oat-full-attention', type=pathlib.Path, default=None,
                        help='Optional: OAT run folder with full attention, for the second figure.')
    parser.add_argument('--step', choices=['train', 'inference'], default='train',
                        help='Which timed step to plot, in both figures.')
    parser.add_argument('--figure-dir', type=pathlib.Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    # The query mode of the OAT run names the figures; the other runs must match it.
    query_mode = read_query_mode(args.oat)
    other_runs = [args.gp_exact, args.gp_iterative]
    if args.oat_full_attention is not None:
        other_runs.append(args.oat_full_attention)
    for results_path in other_runs:
        if read_query_mode(results_path) != query_mode:
            raise ValueError(
                f'{results_path} uses query mode {read_query_mode(results_path)!r}, '
                f'but the OAT run uses {query_mode!r}.'
            )

    args.figure_dir.mkdir(parents=True, exist_ok=True)

    plot_oat_vs_gp(args.oat, args.gp_exact, args.gp_iterative, query_mode, args.step,
                   args.figure_dir)
    if args.oat_full_attention is not None:
        plot_full_attention(args.oat, args.oat_full_attention, query_mode, args.step,
                            args.figure_dir)


if __name__ == '__main__':
    main()
