# Zenodo record: OAT Europe dataset

Paste these fields into the Zenodo upload form. Use the same text for the next
version of the record. Resolve each `[USER CHECK]` before publication. The
`[LEGAL]` points stay open until the legal checks of the release are done.

## Fields

- **Title:** OAT Europe dataset
- **Resource type:** Dataset
- **Creators:** the authors in `CITATION.cff`, in the same order:
  - Donner, Christian (Swiss Data Science Center, ETH Zurich)
  - Cattaneo, Michele (Swiss Data Science Center, ETH Zurich)
  - Samarin, Maxim (Swiss Data Science Center, ETH Zurich)
  - Zanetta, Francesco (Federal Office of Meteorology and Climatology MeteoSwiss)
  - Lloréns Jover, Icíar (Federal Office of Meteorology and Climatology MeteoSwiss)
  - Aznar Siguan, Gabriela (Federal Office of Meteorology and Climatology MeteoSwiss)
  - Volpi, Michele (Swiss Data Science Center, ETH Zurich)
- **Licence:** Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0)
- **Version:** 1.0
- **Related identifiers:**
  - The paper "Observation Assimilation Transformer: Probabilistic Weather
    Analysis from Sparse Station Observations". Relation: "is supplement to".
  - https://github.com/SwissDataScienceCenter/observation_assimilation_transformer.
    Relation: "is supplement to". Resource type: Software.
- **Keywords:** weather, ERA5, Meteostat, station observations, Europe,
  downscaling

## Description

The Europe dataset of the paper "Observation Assimilation Transformer:
Probabilistic Weather Analysis from Sparse Station Observations". It contains
ERA5 surface fields, static fields and hourly station observations for Europe.
The code is at
https://github.com/SwissDataScienceCenter/observation_assimilation_transformer.

### Files

- `era5-europe-mars-n320-2019-2024-1h-v1.zarr.tar`: ERA5 surface fields
  (anemoi-datasets zarr format).
- `europe-static.zarr.tar`: elevation and distance to the coast.
- `hourly_2019_2025_temp_prcp_wspd_wpgt_rhum.parquet`: hourly station
  observations from Meteostat.
- `station_quality.parquet`: quality scores for each station.
- `SHA256SUMS`: the SHA-256 checksum of each file. The paths are the paths
  after the unpack.

Download and check the files with the script of the code repository:

    python -m scripts.download_artifacts --what data --dest .

The script unpacks the files into `data/` and checks each file against
`SHA256SUMS`.

### ERA5 surface fields

- Source: ERA5 and ERA5.1 on the reduced Gaussian grid N320, from the ECMWF
  MARS archive (class `ea`, stream `oper`, surface level). `2t`, `10u`, `10v`
  and `tcc` are hourly analyses (type `an`). `tp` comes from the short-range
  forecasts (type `fc`, step 1 h). It is the accumulation over the hour.
- Variables: `2t` (2 m temperature, K), `10u` and `10v` (10 m wind components,
  m/s), `tcc` (total cloud cover, fraction 0 to 1), `tp` (total precipitation,
  m for each hour).
- Area: latitude 34° N to 72° N, longitude 25° W to 45° E. The subset has
  20 933 grid points.
- Time range: 2019-01-01T00:00 to 2024-12-31T23:00 UTC, every hour
  (52 608 time steps). No dates are missing.
- Processing: a spatial and temporal subset of an anemoi-datasets ERA5 dataset
  with `anemoi.datasets.open_dataset`. The values are not changed. The
  statistics (mean, standard deviation, minimum, maximum) are computed over
  2019 to 2024.
- Licence: CC BY 4.0 (Copernicus licence). The source anemoi dataset gives
  the attribution "ECMWF". The data was copied from MARS in 2026. Attribution
  text: "Contains modified Copernicus Climate Change Service information
  [2026]. Neither the European Commission nor ECMWF is responsible for any use
  that may be made of the Copernicus information or data it contains."

### Static fields

- `elevation` (m, NaN over the ocean): from the AWS Terrain Tiles
  (https://registry.opendata.aws/terrain-tiles), skadi format
  (https://s3.amazonaws.com/elevation-tiles-prod/skadi/), 1 arc-second HGT
  tiles. Tilezen joerd (https://github.com/tilezen/joerd) makes these tiles
  from a blend of elevation sources. For this area, the sources are EU-DEM,
  SRTM, GMTED2010, ETOPO1, ArcticDEM (north of 60° N) and national models of
  Austria, Norway and the United Kingdom. The tiles are merged and averaged to
  a coarser grid. The attribute `long_name` in the zarr says "SRTM 90m". This
  is not correct: the source is the blend above and the grid spacing is about
  1 km.
- `distance_to_coast` (m, 0 over the ocean): the land mask comes from the
  Natural Earth 10 m land polygons
  (https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip). The
  distance is a Euclidean distance transform in EPSG:3035. The result is
  projected back to EPSG:4326 with bilinear interpolation.
- Grid: EPSG:4326, latitude 34° N to 72° N, longitude 25° W to 45° E,
  4145 × 7636 points, spacing 33 arc-seconds (0.0091667°).
- Licence of `elevation`: [LEGAL] the licences of the Tilezen sources
  (https://github.com/tilezen/joerd/blob/master/docs/attribution.md). The
  Tilezen page does not give the EU-DEM licence. Attribution:
  - Europe terrain data produced using Copernicus data and information funded
    by the European Union - EU-DEM layers.
  - SRTM data courtesy of the U.S. Geological Survey.
  - GMTED2010 data courtesy of the U.S. Geological Survey.
  - Global ETOPO1 terrain data U.S. National Oceanic and Atmospheric
    Administration.
  - ArcticDEM terrain data DEM(s) were created from DigitalGlobe, Inc.,
    imagery and funded under National Science Foundation awards 1043681,
    1559691, and 1542736.
  - Austria terrain data © offene Daten Österreichs – Digitales Geländemodell
    (DGM) Österreich (CC BY 3.0 AT).
  - Norway terrain data © Kartverket (CC BY 4.0).
  - United Kingdom terrain data © Environment Agency copyright and/or
    database right 2015. All rights reserved (Open Government Licence v3).
- Licence of `distance_to_coast`: Natural Earth (public domain,
  https://www.naturalearthdata.com/about/terms-of-use/).

### Station observations (Meteostat)

- Source: Meteostat hourly data (https://meteostat.net), downloaded with
  `meteostat-cli` 0.3.0 (Python package `meteostat` 2.1.4) on 2026-05-19, for
  all stations in latitude 34° N to 72° N and longitude 25° W to 45° E.
- Time range: 2019-01-01T00:00 UTC to 2025-12-31T00:00 UTC, every hour.
- Columns and units: `station` (Meteostat station ID), `time`, `temp` (°C),
  `rhum` (%), `prcp` (mm/h), `wspd` (km/h), `wpgt` (wind gust, km/h).
- Processing:
  - Remove the stations that are not on land (Natural Earth 10 m land
    polygons).
  - Set `temp` below −40 °C to missing.
  - Set `prcp` above 200 mm/h to missing.
- Size: 184 572 846 rows.
- Licence: Meteostat data, Deutscher Wetterdienst (DWD), CC BY-NC 4.0.
  Attribution: "Data by Meteostat" (https://dev.meteostat.net/license).

### Station quality

- One row for each of the 4210 stations of the download (before the land
  filter).
- Checks for each station and variable:
  - Physical limits: `temp` −80 to 60, `rhum` 0 to 100, `wspd` 0 to 300,
    `wpgt` 0 to 400, `prcp` 0 to 500.
  - Steps between two hours: `temp` more than 15, `rhum` more than 50, `wspd`
    more than 100.
  - Values more than 3 standard deviations from the mean of the station and
    the month. For precipitation, only the wet hours are used.
  - Flat lines: 6 or more equal values in a row.
  - Drift: a Kolmogorov–Smirnov test for each year (p < 0.05, at least 1000
    values in the year), for `temp`, `prcp`, `wspd` and `rhum`.
- Scores: a quality score for each variable and a composite `quality_score`
  (0.9 minus the weighted fraction of flagged values, plus 0.1 times the drift
  score). The column `tier` gives the station type from the ID format (`wmo`,
  `icao` or `other`). The file also gives the country, latitude, longitude
  and elevation of each station.
- Use in the paper: the OAT code does not apply a quality threshold. It only
  removes the stations with a missing latitude, longitude or elevation.

### Units in the OAT code

The OAT code converts the ERA5 fields to the observation units: temperature
`2t − 273.15` (°C), precipitation `tp × 1000` (mm/h), wind speed
`sqrt(10u² + 10v²)` (m/s). It converts the observed `wspd` and `wpgt` from
km/h to m/s.
