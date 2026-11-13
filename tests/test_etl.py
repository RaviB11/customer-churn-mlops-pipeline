from churn_pipeline import config, etl


def test_raw_shape(raw_df):
    assert raw_df.shape == (7043, 21)


def test_warehouse_round_trip_matches_csv(raw_df, tmp_path):
    db = etl.build_warehouse(raw_df, tmp_path / "wh.db")
    joined = etl.load_from_warehouse(db)
    assert len(joined) == len(raw_df)
    a = raw_df.sort_values(config.ID_COL).reset_index(drop=True)[joined.columns]
    b = joined.reset_index(drop=True)
    for col in joined.columns:
        assert (a[col].astype(str).str.strip() == b[col].astype(str).str.strip()).all(), col


def test_hash_is_stable():
    assert etl.file_sha256(config.RAW_CSV) == etl.file_sha256(config.RAW_CSV)


def test_integrity_check_passes_on_clean_data(raw_df, tmp_path):
    from sqlalchemy import create_engine

    db = etl.build_warehouse(raw_df, tmp_path / "wh.db")
    with create_engine(f"sqlite:///{db}").connect() as conn:
        counts = etl.check_warehouse_integrity(conn)
    assert set(counts.values()) == {len(raw_df)}


def test_integrity_check_catches_orphans_and_duplicates(raw_df, tmp_path):
    import pytest
    from sqlalchemy import create_engine, text

    db = etl.build_warehouse(raw_df, tmp_path / "wh.db")
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM services WHERE customerID = (SELECT MIN(customerID) FROM services)"))
        conn.execute(text("INSERT INTO accounts SELECT * FROM accounts LIMIT 1"))
        conn.execute(text("INSERT INTO churn_labels (customerID, Churn) VALUES ('0000-ORPHN', 'No')"))
    with pytest.raises(etl.WarehouseIntegrityError) as exc:
        etl.load_from_warehouse(db)
    msg = str(exc.value)
    assert "missing from services" in msg and "accounts" in msg and "orphan" in msg


def test_default_paths_follow_config_at_call_time(raw_df, tmp_path, monkeypatch):
    """Repointing config.WAREHOUSE_DB must redirect the default-path calls
    (the old defaults were frozen at import and ignored the change)."""
    target = tmp_path / "redirected.db"
    monkeypatch.setattr(config, "WAREHOUSE_DB", target)
    assert etl.build_warehouse(raw_df) == target and target.exists()
    assert len(etl.load_from_warehouse()) == len(raw_df)


def test_missing_warehouse_raises_clear_error(tmp_path):
    import pytest

    missing = tmp_path / "nope.db"
    with pytest.raises(FileNotFoundError, match="build_warehouse"):
        etl.load_from_warehouse(missing)
    assert not missing.exists()  # SQLite must not have created an empty file
