# Sphere-projected coordinates for neighbor indexing

**Date:** 2026-08-12
**Status:** Design approved, pending spec review
**Branch:** flat-field-model-interface (or a dedicated feature branch)

## Problem

The OAT spatial attention masks are built by `SpatialNeighborSearch`
(`src/oat/model_module/masking.py`), which runs `torch_cluster.knn` /
`radius` over the batch's spatial coordinates. Those coordinates flow in as
`observation_data["coords"]` and `query_data["coords"]`, shape `[B, N, 2]`.

For the COSMO/PeakWeather dataset the coordinates live on a small domain where
Euclidean distance in the native coordinate values is an acceptable proxy for
real distance. For **ERA5/Meteostat** the coordinates are geographic
**lon/lat in degrees** over a continental domain. Euclidean distance in raw
lon/lat is wrong there:

- longitude degrees shrink with `cos(lat)`, so KNN over-weights longitude at
  high latitude;
- the ±180° seam makes points that are physically adjacent look maximally far
  apart.

**Goal:** run the neighbor search for ERA5 in a space where Euclidean nearest
neighbor is geodesically meaningful, by projecting lon/lat onto the unit
sphere, **without** changing behavior for any other dataset and **without**
mutating the `coords` that other consumers rely on.

## Scope

- In scope: a dataset-owned coordinate projection used *only* for the KNN
  neighbor search, wired through to `SpatialNeighborSearch`, with ERA5 as the
  first (and currently only) dataset that overrides the default.
- ERA5 masking uses `knn` mode. `radius` mode is out of scope (its threshold
  would need re-expressing in unit-sphere chord length; not needed now).
- Out of scope: COSMO/PeakWeather — left exactly as-is, no code or config
  change, no `search_coords` emitted.
- Out of scope: geometry-bias features (`geo`), output heads, plotting — they
  keep consuming lon/lat `coords` unchanged.

## Key constraint discovered during design

`observation_data["coords"]` / `query_data["coords"]` are **not** read only by
the mask generator. They are also consumed as 2D lon/lat by:

- the OOTG baseline — `src/oat/baselines/ootg_predictor.py`,
  `src/oat/baselines/ootg_generator.py`;
- the geometry-bias path and plotting (implicitly, via lon/lat semantics).

Therefore the projection must **not** overwrite `coords`. It is written to a
**separate** `search_coords` key that only the mask generator reads. Everything
else continues to see lon/lat.

## Design

### 1. Dataset-owned projection method

Base class default is identity (the coordinates are already in a Euclidean
space):

```python
# src/oat/data_module/field_dataset.py  (FieldDatasetBase)
def project_index_coords(self, coords: Tensor) -> Tensor:
    """Map spatial coords into the space where Euclidean KNN is meaningful.

    Default: identity — coordinates are already treated as Euclidean.
    Datasets whose coordinates are geographic (lon/lat) override this.

    Args:
        coords: (..., 2) spatial coordinates.
    Returns:
        (..., D) coordinates for neighbor search (D may differ from input).
    """
    return coords
```

ERA5 overrides with the unit-sphere projection:

```python
# src/oat/data_module/field_era5_meteostat_dataset.py
#   (FieldERA5MeteostatDataset)
def project_index_coords(self, coords: Tensor) -> Tensor:
    # coords: (..., 2) = (lon, lat) in degrees
    lon = torch.deg2rad(coords[..., 0])
    lat = torch.deg2rad(coords[..., 1])
    cos_lat = torch.cos(lat)
    x = cos_lat * torch.cos(lon)
    y = cos_lat * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)  # (..., 3) on the unit sphere
```

Notes:
- 2D → 3D is fine: `torch_cluster.knn` accepts arbitrary dimensionality.
- Chordal (straight-line) distance on the unit sphere is a monotone function of
  great-circle distance, so KNN neighbors are the true geodesic neighbors.
- The ±180° seam and `cos(lat)` distortion are both handled by construction.

### 2. Applied to a separate `search_coords` key

`_unwrap_fields_to_model_batch`
(`src/oat/predictor_module/field_batch_utils.py`) assembles the model batch
from collated `Field`s. It gains an injected projection callable and writes an
extra key per group, leaving `coords` untouched:

```python
def _unwrap_fields_to_model_batch(
    batch, roles, requires_observation, project_index_coords=None,
):
    proj = project_index_coords or (lambda c: c)

    def grp(group):
        f = batch[group]["fields"]
        obs = f[roles["target"]]
        coords = obs.spatial_coords            # (B, N, 2) lon/lat, unchanged
        return {
            "coords": coords,
            "search_coords": proj(coords),     # (B, N, D) for search only
            "geo": f[roles["geo_bias"]].values,
            "topo": f[roles["topo"]].values,
            "obs": obs.values,
            "mask_obs": obs.mask,
            "global_tokens": f[roles["global"]].values,
        }
    ...
```

Decision: **emit `search_coords` only when the dataset overrides the base
method**, so COSMO batches are byte-for-byte unchanged. The predictor decides
this with an explicit override check (see §4) and passes
`project_index_coords=None` for datasets that do not override; the helper then
emits no `search_coords` key at all.

### 3. Mask generator consumes `search_coords` with fallback

`model.py` (the three `mask_generator(...)` call sites, ~lines 819-823) read
`search_coords` when present, else `coords`:

```python
def _search(d):
    return d.get("search_coords", d["coords"])

edges_src    = self.mask_generator(coords_1=_search(observation_data))
edges_tgt    = self.mask_generator(coords_1=_search(query_data))
edges_memory = self.mask_generator(
    coords_1=_search(observation_data),
    coords_2=_search(query_data),
)
```

`SpatialNeighborSearch` itself is unchanged — it already accepts arbitrary
coordinate dimensionality.

### 4. Wiring — predictor surfaces the bound method

`_unwrap_fields_to_model_batch` is dataset-blind, so the predictor injects the
projection. Chosen mechanism: the **bound method** from the dataset (most
faithful to "define it in the dataset").

- The predictor obtains `project_index_coords` from the dataset object it
  already reaches (`train_loader.dataset` at setup, co-located with the
  existing `validate_output_head_supports` access) and stores it.
- **Override check:** the predictor stores the bound method only when the
  dataset actually overrides it —
  `type(ds).project_index_coords is not FieldDatasetBase.project_index_coords`;
  otherwise it stores `None`. This is what keeps COSMO batches unchanged (§2).
- `_normalize_batch` passes the stored callable (or `None`) into
  `_unwrap_fields_to_model_batch`.
- Applies to both `OATPredictor` and `GPPredictor` (both call the unwrap
  helper).

**Eval/checkpoint path requirement (known footgun in this repo).** The
projection must be re-established whenever the dataset is rebuilt for testing —
`scripts/evaluate_test_set.py` and `src/oat/predictor_module/checkpoint_utils.py`
— not only during `fit`. Since the bound method comes from the live dataset
object, the requirement is simply that the eval path builds the dataset and
hands it to the predictor before batches are unwrapped, exactly as the training
path does. This must be verified on real eval, not just training.

To keep `TrainDataset`-style wrappers working, the lookup walks to the
underlying dataset the same way the existing `validate_output_head_supports`
code does (`training_dataset.dataset` when present), so a `Subset`/wrapper does
not hide the method.

## Files touched

- `src/oat/data_module/field_dataset.py` — add default `project_index_coords`.
- `src/oat/data_module/field_era5_meteostat_dataset.py` — override with the
  unit-sphere projection.
- `src/oat/predictor_module/field_batch_utils.py` — accept and apply the
  projection, emit `search_coords`.
- `src/oat/predictor_module/oat_predictor.py` — obtain/store the bound method,
  pass it into the unwrap helper.
- `src/oat/predictor_module/gp_predictor.py` — same wiring as OAT.
- `src/oat/model_module/model.py` — read `search_coords` with fallback at the
  three mask-generator call sites.
- Eval path (`scripts/evaluate_test_set.py` /
  `src/oat/predictor_module/checkpoint_utils.py`) — ensure the projection is
  wired when the dataset is rebuilt for testing.

## Testing

Unit:
- `FieldDatasetBase.project_index_coords` is identity (returns input unchanged,
  same shape).
- ERA5 override maps known lon/lat to the correct xyz (spot values, e.g.
  `(0,0)→(1,0,0)`, `(90,0)→(0,1,0)`, `(_,90)→(0,0,1)`), and all outputs have
  unit norm.
- Seam check: two points at lon `+179.9` and `-179.9` at equal latitude are
  near neighbors in projected space but far apart in raw lon/lat.

Integration:
- A batch through `_unwrap_fields_to_model_batch` with the ERA5 projection
  carries `search_coords` of shape `(B, N, 3)`; with the base (identity) /
  COSMO path it carries **no** `search_coords`, and the batch is otherwise
  unchanged.
- Mask generator falls back to `coords` when `search_coords` is absent
  (COSMO regression: masks identical to pre-change).

## Non-goals / deferred

- `radius`-mode threshold conversion to chord length.
- Any COSMO/PeakWeather change.
- A general CRS/coordinate-system abstraction — a single overridable method is
  enough for the one geographic dataset in play.
