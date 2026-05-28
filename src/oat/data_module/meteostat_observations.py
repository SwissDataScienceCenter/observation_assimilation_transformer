from __future__ import annotations

import numpy as np
import pandas as pd


class MeteostatObservations:
    def __init__(self, obs_path: str, station_meta_path: str) -> None:
        meta = pd.read_parquet(station_meta_path)
        if "station" in meta.columns:
            meta = meta.set_index("station")
        self.stations_table: pd.DataFrame = meta.dropna(
            subset=["latitude", "longitude", "elevation"]
        )[["latitude", "longitude", "elevation"]].copy()
        self.stations_table.index = self.stations_table.index.astype(str)

        obs = pd.read_parquet(obs_path)
        obs["time"] = pd.to_datetime(obs["time"], utc=True)
        obs["station"] = obs["station"].astype(str)
        obs = obs[obs["station"].isin(set(self.stations_table.index))]
        self._obs: pd.DataFrame = obs.set_index(["time", "station"]).sort_index()

    @property
    def stations(self) -> np.ndarray:
        obs_ids = set(self._obs.index.get_level_values("station").unique())
        meta_ids = set(self.stations_table.index)
        return np.array(sorted(obs_ids & meta_ids))

    def get_observations(
        self,
        first_date: pd.Timestamp,
        last_date: pd.Timestamp,
        parameters: list[str],
        stations: np.ndarray | list[str],
        return_mask: bool = True,
        as_numpy: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        if not as_numpy:
            raise NotImplementedError("Only as_numpy=True is supported.")

        stations = [str(s) for s in stations]
        times = pd.date_range(
            first_date.floor("h"),
            last_date.ceil("h"),
            freq="1h",
            tz="UTC",
            inclusive="both",
        )
        T, N, V = len(times), len(stations), len(parameters)
        obs_arr = np.full((T, N, V), np.nan, dtype=np.float32)

        try:
            window = self._obs.loc[
                first_date : last_date - pd.Timedelta(seconds=1), parameters
            ]
        except KeyError:
            window = pd.DataFrame(columns=parameters)

        if not window.empty:
            unstacked = window.unstack(level="station")
            for vi, param in enumerate(parameters):
                if param not in unstacked.columns.get_level_values(0):
                    continue
                param_data = unstacked[param].reindex(index=times, columns=stations)
                obs_arr[:, :, vi] = param_data.to_numpy(dtype=np.float32, na_value=np.nan)

        mask = ~np.isnan(obs_arr)
        return (obs_arr, mask) if return_mask else obs_arr
