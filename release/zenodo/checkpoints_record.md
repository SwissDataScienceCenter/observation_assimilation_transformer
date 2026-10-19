# Zenodo record: OAT checkpoints and results v1.0

Paste these fields into the Zenodo upload form. Use the same text for the next
version of the record.

## Fields

- **Title:** OAT checkpoints and results v1.0
- **Resource type:** Dataset
- **Creators:** the authors in `CITATION.cff`, in the same order:
  - Donner, Christian (Swiss Data Science Center, ETH Zurich)
  - Cattaneo, Michele (Swiss Data Science Center, ETH Zurich)
  - Samarin, Maxim (Swiss Data Science Center, ETH Zurich)
  - Zanetta, Francesco (Federal Office of Meteorology and Climatology MeteoSwiss)
  - Lloréns Jover, Icíar (Federal Office of Meteorology and Climatology MeteoSwiss)
  - Aznar Siguan, Gabriela (Federal Office of Meteorology and Climatology MeteoSwiss)
  - Volpi, Michele (Swiss Data Science Center, ETH Zurich)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Version:** 1.0
- **Related identifiers:**
  - The paper "Observation Assimilation Transformer: Probabilistic Weather
    Analysis from Sparse Station Observations". Relation: "is supplement to".
    [USER CHECK] The paper has no DOI yet. Add it when it is known.
  - https://github.com/SwissDataScienceCenter/observation_assimilation_transformer.
    Relation: "is supplement to". Resource type: Software.
- **Keywords:** weather, downscaling, data assimilation, transformer,
  probabilistic forecasting

## Description

Trained model checkpoints and per-station result files of the paper
"Observation Assimilation Transformer: Probabilistic Weather Analysis from
Sparse Station Observations". The code is at
https://github.com/SwissDataScienceCenter/observation_assimilation_transformer.

The record has three files:

- `checkpoints.tar`: the 18 model checkpoints of the paper. Each run folder
  `checkpoints/<family>/<run>/` contains the best checkpoint (`.ckpt`) and the
  `config.yaml` of the run. The families are `oat`, `oat_field`,
  `uninformed_oat`, `uninformed_oat_field`, `gp_oat_field`,
  `deep_gp_oat_field` and `ootg` (the Gridded TNP baseline). There are 9
  checkpoints for the Swiss dataset (COSMO-1E and PeakWeather) and 9 for the
  Europe dataset (ERA5 and Meteostat). The file `configs/checkpoints.yaml` in
  the code repository gives the checkpoint for each model in the paper.
- `results.tar`: the result files of the paper table (`data/results/`). For
  each model and dataset there is a per-station metrics file (`.csv`) and a
  summary (`.summary.json`). The interpolation baselines have result files but
  no checkpoint.
- `SHA256SUMS`: the SHA-256 checksum of each file in the two tar files. The
  paths are the paths after the unpack.

Download and check the files with the script of the code repository:

    python -m scripts.download_artifacts --what checkpoints --dest .

The script puts the checkpoints into `checkpoints/` and the result files into
`data/results/`, and checks each file against `SHA256SUMS`.

The Swiss checkpoints need MeteoSwiss data (COSMO-1E, COSMO-E1 and PeakWeather
as prepared for the paper) for new evaluations. This data is not public. The
result files give the Swiss rows of the paper table. The Europe dataset is in
the record "OAT Europe dataset".
