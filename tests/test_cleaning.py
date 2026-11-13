import pandas as pd
import pytest

from churn_pipeline import config
from churn_pipeline.cleaning import clean_dataset, clean_features


def test_no_nulls_after_cleaning(raw_df):
    clean = clean_dataset(raw_df)
    assert clean.drop(columns=[config.ID_COL]).isna().sum().sum() == 0


def test_total_charges_numeric_and_blank_handling(raw_df):
    blanks = raw_df["TotalCharges"].astype(str).str.strip() == ""
    assert blanks.sum() == 11  # known property of the source file
    clean = clean_dataset(raw_df)
    assert pd.api.types.is_float_dtype(clean["TotalCharges"])
    assert (raw_df.loc[blanks, "tenure"] == 0).all()
    assert (clean.loc[blanks, "TotalCharges"] == 0.0).all()


def test_binary_fields_are_zero_one(raw_df):
    clean = clean_dataset(raw_df)
    for col in config.BINARY_YES_NO_COLS + ["SeniorCitizen", "gender_male", config.TARGET_COL]:
        assert set(clean[col].unique()) <= {0, 1}, col


def test_duplicate_ids_rejected(raw_df):
    dup = pd.concat([raw_df.head(3), raw_df.head(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        clean_dataset(dup)


def test_whitespace_is_stripped(raw_df):
    rec = raw_df.head(1).copy()
    rec["Contract"] = "  Month-to-month "
    rec["Partner"] = "Yes "
    out = clean_features(rec)
    assert out["Contract"].iloc[0] == "Month-to-month"
    assert out["Partner"].iloc[0] == 1
