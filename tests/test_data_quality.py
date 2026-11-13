from churn_pipeline import config
from churn_pipeline.cleaning import clean_dataset


def test_label_distribution_in_expected_range(raw_df):
    # IBM sample churn rate is ~26.5%; a large move means a broken extract
    rate = clean_dataset(raw_df)[config.TARGET_COL].mean()
    assert 0.20 <= rate <= 0.33


def test_customer_ids_unique(raw_df):
    assert raw_df[config.ID_COL].is_unique


def test_categories_match_expected_schema(raw_df):
    for col, allowed in config.EXPECTED_CATEGORIES.items():
        assert set(raw_df[col].unique()) <= set(allowed), col


def test_numeric_ranges(raw_df):
    clean = clean_dataset(raw_df)
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        assert clean[col].between(lo, hi).all(), col
