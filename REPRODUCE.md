# Reproduce the paper

Run all commands from the repository root, in the project environment (`poetry run bash figures/<script>`). First download the artifacts:

```bash
poetry run python -m scripts.download_artifacts --what all --dest .
```

Each script writes to `figures/out/`. It stops with one message when an input is missing. "MeteoSwiss data" means the Swiss dataset (COSMO-1E, COSMO-E1 and PeakWeather as prepared for the paper). This data is not public. On a cluster, run a script with `sbatch --wrap "poetry run bash figures/<script>"`.

| Paper item | Script | Needs | Approximate run time |
|---|---|---|---|
| Table 1 (`tab:ml_results`, `tab:ci_table`, `tab:att_oat_perf`) | `figures/generate_table1.sh` | Europe data and checkpoints. Swiss rows: the released result files, or MeteoSwiss data. | 4 h on 8 CPU cores (Europe evaluations) |
| Figure 1 (`fig:oat`), panels only | `figures/generate_fig1.sh` | MeteoSwiss data | minutes |
| Figure 2 (`fig:data`) | `figures/generate_fig2.sh` | Europe data and MeteoSwiss data | minutes |
| Figure 3 (`fig:example_field`) | `figures/generate_fig3.sh` | MeteoSwiss data, Swiss OAT checkpoint | minutes |
| Figure 4 (`fig:pit`) | `figures/generate_fig4.sh` | Europe part: Europe data and checkpoints. Swiss part: MeteoSwiss data. | 25 min for each dataset (GPU node) |
| Figure 5 (`fig:error_cosmo`, `fig:error_era5`) | `figures/generate_fig5.sh` | As figure 4. Computes missing `evaluate_test_set --long` dumps. | 2 h for each dataset on CPU, when the dumps are missing |
| Figure 6 (`fig:era5_to_cosmo`) | `figures/generate_fig6.sh` | Europe data and MeteoSwiss COSMO-E1 data | minutes |
| Figure 7 (`fig:extreme_events`) | `figures/generate_fig7.sh` | Europe row: Europe data and checkpoint. Swiss row: MeteoSwiss data. | 10 min (GPU node) |
| Figure A1 (`fig:scaling`) | `figures/generate_figA1.sh` | Swiss checkpoints (configs and GP weights only; the data is fake), a CUDA GPU | 4 h together with figure A2 |
| Figure A2 (`fig:fullattention_vs_sparse`) | `figures/generate_figA2.sh` | The figure A1 runs, a CUDA GPU | see figure A1 |

Notes:

- The run times come from the paper runs (2026). The figure and benchmark times depend on the hardware. The benchmark figures show the times of your hardware, not the paper times.
- `generate_table1.sh` and `generate_fig5.sh` (when it computes missing dumps) write new result files (`<tag>_metrics_test.csv` and `<tag>_metrics_test.summary.json`) into `data/results/`. They replace the released files of the same name. Keep a copy of the released files if you want to compare.
- The result files unpack to `<dest>/data/results`. The scripts `build_results_table` and `evaluate_test_set` always use `<repo>/data/results`. With another `--dest`, give `--results-dir <dest>/data/results` to `build_results_table`. Or download the checkpoints record with `--dest .`.
- To build the table from the released result files only: `poetry run python -m scripts.build_results_table --results-dir data/results`.
- The paper layout of figure 1 was put together by hand from the panels.
- `DRY_RUN=1 bash figures/<script>` prints the commands and does not run them.
