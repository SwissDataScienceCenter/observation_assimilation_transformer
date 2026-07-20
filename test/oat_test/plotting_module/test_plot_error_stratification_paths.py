import pandas as pd

import scripts.plot_error_stratification as pes


def test_long_parquet_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pes,
        "resolve_checkpoint",
        lambda ds, m: f"/x/checkpoints/gp/gp_{ds}-temp_job1/best.ckpt",
    )
    monkeypatch.setattr(pes, "OUT_DIR", tmp_path)
    p = pes.long_parquet_path("cosmo_peakweather", "gp_temp", "test")
    assert p == tmp_path / "gp_cosmo_peakweather-temp_job1_long_test.parquet"


def test_load_model_frames_concats_with_model_col(monkeypatch, tmp_path):
    monkeypatch.setattr(pes, "OUT_DIR", tmp_path)

    def fake_path(ds, m, split):
        f = tmp_path / f"{m}.parquet"
        pd.DataFrame({"station_idx": [0], "crps_t": [0.5]}).to_parquet(f)
        return f

    monkeypatch.setattr(pes, "long_parquet_path", fake_path)
    df = pes.load_model_frames("cosmo_peakweather", "oat_x", ["gp_a", "gp_b"], "test")
    assert set(df["model"]) == {"oat_x", "gp_a", "gp_b"}
    assert len(df) == 3
