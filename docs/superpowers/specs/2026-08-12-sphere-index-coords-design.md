# Sphere-projected coordinates for neighbor indexing and geometry bias

**Date:** 2026-08-12
**Status:** Design approved, pending spec review
**Branch:** flat-field-model-interface (or a dedicated feature branch)

## Problem

Two OAT components treat the batch's spatial coordinates as if they lived in a
flat Euclidean space:

1. **KNN neighbor search** (`SpatialNeighborSearch`,
   `src/oat/model_module/masking.py`) runs `torch_cluster.knn` over
   `observation_data["coords"]` / `query_data["coords"]`, shape `[B, N, 2]`,
   using raw Euclidean distance.
2. **Geometry bias** (`GeometryAwareMHA._dist_function`,
   `src/oat/model_module/attention.py`) computes a plain elementwise difference
   `geo_q - geo_k` over the geo-bias field, whose first two channels are the
   same lon/lat coordinates, and feeds that difference to the bias MLP.

For **COSMO/PeakWeather** the coordinates live on a small domain where flat
Euclidean treatment is an acceptable proxy. For **ERA5/Meteostat** the
coordinates are geographic **lon/lat in degrees** over a continental domain
(lat 34→72, lon −25→45). Both defects appear:

- longitude degrees shrink with `cos(lat)`, over-weighting longitude at high
  latitude;
- the ±180° seam makes physically adjacent points look maximally far apart.

**Goal:** run both the neighbor search and the geometry-bias difference for
ERA5 in a space where Euclidean operations are geographically meaningful, by
projecting lon/lat onto the unit sphere — **without** changing behavior for any
other dataset and **without** mutating the `coords` that other consumers rely
on.

## Scope

- In scope: a single dataset-owned coordinate projection (`project_index_coords`)
  used for **both** the KNN search coordinates and the geometry-bias horizontal
  coordinates, wired through from the dataset, with ERA5 as the first (and
  currently only) overriding dataset.
- ERA5 masking uses `knn` mode. `radius` mode is out of scope (its threshold
  would need re-expressing in unit-sphere chord length; not needed now).
- Out of scope: COSMO/PeakWeather — left exactly as-is, no code or config
  change, no projected coordinates emitted.
- Out of scope: other consumers of `coords` (OOTG baseline, plotting) — they
  keep consuming lon/lat `coords` unchanged.

## Key constraint discovered during design

`observation_data["coords"]` / `query_data["coords"]` are **not** read only by
the mask generator. They are also consumed as 2D lon/lat by the OOTG baseline
(`src/oat/baselines/ootg_predictor.py`, `ootg_generator.py`) and plotting.
Therefore the projection must **not** overwrite `coords`. The projected result
is written to a **separate** `search_coords` key that only the mask generator
reads, and is reused (not recomputed) for the geo-bias vector. Everything else
continues to see lon/lat.

## Design decision: per-coordinate geo bias (rotation invariance)

The geo bias currently feeds the MLP a *vector* difference `geo_q - geo_k`. On
the sphere there is a genuine choice of representation, analyzed and decided
here so the plan does not reopen it:

- **Raw lon/lat vector diff (today):** broken for ERA5 — seam + cos(lat).
- **Per-coordinate vector diff on the projected sphere** `[Δx, Δy, Δz, Δelev]`:
  fixes seam + cos(lat), keeps N/S–E/W direction (anisotropy), requires **no
  model change**. It is *not* SO(3) invariant — the vector depends on the pair's
  absolute orientation in the fixed ECEF frame — so rotation invariance would
  have to be learned rather than guaranteed.
- **Scalar chord distance** `[‖Δxyz‖, Δelev]`: strictly SO(3) invariant but
  **isotropic** (for a point pair the great-circle distance is the *only* SO(3)
  invariant, so all direction is lost), and it needs a pairwise model-side norm
  plus an ERA5-vs-COSMO reduction split.

**Decision: per-coordinate vector diff.** Rationale: it is the simplest
(zero model change, consistent with COSMO's existing directional bias), keeps
direction, and is strictly more expressive than the scalar (the norm is
recoverable from the components, so the model can still learn distance-only
decay). The lost SO(3) invariance is second-order here because the neighbor
*set* is already fixed correctly by the KNN mask; the geo bias only *modulates*
weights within an already-correct neighborhood. Scalar-distance invariance is
recorded as the fallback if location-overfitting is later observed.

## Design

### 1. Dataset-owned projection method

Base class default is identity (coordinates are already Euclidean):

```python
# src/oat/data_module/dataset.py  (DatasetBase)
def project_index_coords(self, coords: Tensor) -> Tensor:
    """Map spatial coords into the space where Euclidean ops are meaningful.

    Default: identity — coordinates are already treated as Euclidean.
    Datasets whose coordinates are geographic (lon/lat) override this.

    Args:
        coords: (..., 2) spatial coordinates.
    Returns:
        (..., D) coordinates for neighbor search / geo-bias (D may differ).
    """
    return coords
```

ERA5 overrides with the unit-sphere projection:

```python
# src/oat/data_module/era5_meteostat_dataset.py
#   (ERA5MeteostatDataset)
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
- 2D → 3D is fine: `torch_cluster.knn` and the bias MLP both accept arbitrary
  dimensionality.
- Chordal (straight-line) distance on the unit sphere is a monotone function of
  great-circle distance, so KNN neighbors are the true geodesic neighbors.
- The ±180° seam and `cos(lat)` distortion are both handled by construction.

### 2. Application site A — KNN search coords (`search_coords`)

`_unwrap_fields_to_model_batch`
(`src/oat/predictor_module/field_batch_utils.py`) gains an injected projection
callable and writes an extra key per group, leaving `coords` untouched:

```python
def _unwrap_fields_to_model_batch(
    batch, roles, requires_observation, project_index_coords=None,
):
    def grp(group):
        f = batch[group]["fields"]
        obs = f[roles["target"]]
        coords = obs.spatial_coords            # (B, N, 2) lon/lat, unchanged
        geo_vals = f[roles["geo_bias"]].values # (B, N, G) standardized
        out = {
            "coords": coords,
            "geo": geo_vals,
            "topo": f[roles["topo"]].values,
            "obs": obs.values,
            "mask_obs": obs.mask,
            "global_tokens": f[roles["global"]].values,
        }
        if project_index_coords is not None:
            proj = project_index_coords(coords)      # (B, N, D)
            out["search_coords"] = proj              # site A
            out["geo"] = torch.cat(                  # site B (see §3)
                [proj, geo_vals[..., 2:]], dim=-1
            )
        return out
    ...
```

Decision: the projected keys are emitted **only when the dataset overrides the
base method**, so COSMO batches are byte-for-byte unchanged. The predictor
decides this with an explicit override check (§5) and passes
`project_index_coords=None` for datasets that do not override.

### 3. Application site B — geometry-bias `geo` vector

The geo-bias field values reaching the model are **standardized**, not raw
degrees: `DatasetBase.compute_field_leaves` maps every non-obs field
(including `geo`) to a plain standardization transform
(`dataset.py:359`, `stats[n].get_standard_transform()`). So the geo
lon/lat channels cannot be projected in place with `deg2rad`/`cos`.

Instead the projection is applied to the **raw** lon/lat, which is exactly the
`coords` tensor — the geo field's first two channels are `coords.x/coords.y`
before standardization. This is the *same* projected tensor computed for
site A, so it is reused, not recomputed:

- ERA5 (overriding): `geo = cat([proj, geo_vals[..., 2:]], dim=-1)` — projected
  xyz (unnormalized, ∈[−1,1]) for the horizontal, plus the geo field's
  already-standardized **elevation** channel(s) untouched. Shape `(B, N, 4)`.
- COSMO (non-overriding): `geo = geo_vals`, unchanged.

The model is **unchanged**: `_dist_function` keeps returning `geo_q - geo_k`,
now a plain Euclidean per-coordinate difference `[Δx, Δy, Δz, Δelev]` in the
projected space (per the decision above).

Consequences:
- **Contract:** any dataset that overrides `project_index_coords` must expose a
  geo field whose first two channels are the spatial coords (lon, lat) aligned
  with `coords`. ERA5 satisfies this
  (`configs/dataset/era5_meteostat/settings.yaml`, `out: [longitude, latitude, elevation]`);
  assert it at setup.
- **Mixed scaling:** unnormalized xyz (∈[−1,1]) sit beside standardized
  elevation (~N(0,1)) — comparable magnitudes, fine for the bias MLP. xyz are
  deliberately *not* standardized: per-axis standardization would warp the
  sphere and break the chord geometry.

### 4. `d_geo_bias` becomes projection-aware

`d_geo_bias` is derived, not configured: `oat_predictor.py:606` sets
`d_geo_bias = len(data_cfg.geo_vars)`. After projection the geo vector width
changes (2 lon/lat channels → `proj_dim` channels), so the derivation must
account for it:

```python
# probe the projection's output width once, on a dummy (1, 2) coord
proj_dim = project_index_coords(torch.zeros(1, 2)).shape[-1]
d_geo_bias = len(data_cfg.geo_vars) - 2 + proj_dim
```

- Identity projection: `proj_dim == 2` → `d_geo_bias == len(geo_vars)`
  (backward-compatible, COSMO unchanged).
- ERA5 sphere: `proj_dim == 3` → `d_geo_bias == len(geo_vars) + 1` (3 → 4).

### 5. Wiring — predictor surfaces the bound method

`_unwrap_fields_to_model_batch` is dataset-blind, so the predictor injects the
projection. Chosen mechanism: the **bound method** from the dataset (most
faithful to "define it in the dataset").

- The predictor obtains `project_index_coords` from the dataset object it
  already reaches (`train_loader.dataset` at setup, co-located with the existing
  `validate_output_head_supports` access) and stores it.
- **Override check:** the predictor stores the bound method only when the
  dataset actually overrides it —
  `type(ds).project_index_coords is not DatasetBase.project_index_coords`;
  otherwise it stores `None`. This is what keeps COSMO batches unchanged (§2).
- The predictor probes `proj_dim` (project a dummy `(1, 2)` coord) to compute
  the projection-aware `d_geo_bias` (§4) before model construction.
- `_normalize_batch` passes the stored callable (or `None`) into
  `_unwrap_fields_to_model_batch`.
- Applies to both `OATPredictor` and `GPPredictor` (both call the unwrap helper
  and both build a model with `d_geo_bias`).
- To keep `TrainDataset`-style wrappers working, the lookup walks to the
  underlying dataset the same way the existing `validate_output_head_supports`
  code does (`training_dataset.dataset` when present).

**Eval/checkpoint path requirement (known footgun in this repo).** The
projection and the projection-aware `d_geo_bias` must be re-established whenever
the dataset is rebuilt for testing — `scripts/evaluate_test_set.py` and
`src/oat/predictor_module/checkpoint_utils.py` — not only during `fit`. Since
the bound method comes from the live dataset object, the requirement is that the
eval path builds the dataset and hands it to the predictor before batches are
unwrapped, exactly as the training path does. Verify on real eval, not just
training.

### 6. Mask generator consumes `search_coords` with fallback

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

## Files touched

- `src/oat/data_module/dataset.py` — add default `project_index_coords`.
- `src/oat/data_module/era5_meteostat_dataset.py` — override with the
  unit-sphere projection.
- `src/oat/predictor_module/field_batch_utils.py` — accept and apply the
  projection; emit `search_coords` and the projected `geo`.
- `src/oat/predictor_module/oat_predictor.py` — obtain/store the bound method,
  compute projection-aware `d_geo_bias`, pass the callable into the unwrap
  helper; assert the geo first-two-channels contract for overriding datasets.
- `src/oat/predictor_module/gp_predictor.py` — same wiring as OAT.
- `src/oat/model_module/model.py` — read `search_coords` with fallback at the
  three mask-generator call sites. **No change to `attention.py`** — the geo
  bias consumes the reshaped `geo` unchanged.
- Eval path (`scripts/evaluate_test_set.py` /
  `src/oat/predictor_module/checkpoint_utils.py`) — ensure the projection and
  `d_geo_bias` are wired when the dataset is rebuilt for testing.

## Testing

Unit:
- `DatasetBase.project_index_coords` is identity (returns input unchanged,
  same shape).
- ERA5 override maps known lon/lat to the correct xyz (spot values, e.g.
  `(0,0)→(1,0,0)`, `(90,0)→(0,1,0)`, `(_,90)→(0,0,1)`), and all outputs have
  unit norm.
- Seam check: two points at lon `+179.9` and `-179.9` at equal latitude are
  near neighbors in projected space but far apart in raw lon/lat.

Integration:
- A batch through `_unwrap_fields_to_model_batch` with the ERA5 projection
  carries `search_coords` of shape `(B, N, 3)` **and** `geo` of shape
  `(B, N, 4)` = `[xyz, standardized elevation]`; with the base (identity) /
  COSMO path it carries **no** `search_coords` and `geo` is unchanged, and the
  batch is otherwise identical.
- `d_geo_bias` derivation: identity → `len(geo_vars)`; ERA5 → `len(geo_vars)+1`.
- Mask generator falls back to `coords` when `search_coords` is absent
  (COSMO regression: masks identical to pre-change).
- COSMO regression: `geo` and masks byte-for-byte identical to pre-change.

## Non-goals / deferred

- `radius`-mode threshold conversion to chord length.
- SO(3)-invariant scalar-distance geo bias (recorded fallback, not built).
- Any COSMO/PeakWeather change.
- A general CRS/coordinate-system abstraction — a single overridable method is
  enough for the one geographic dataset in play.
