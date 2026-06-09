"""Unit tests for src/models/*.py"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def rfm_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "CustomerID": [f"C{i}" for i in range(n)],
        "Recency":    np.random.randint(1, 365, n),
        "Frequency":  np.random.randint(1, 50,  n),
        "Monetary":   np.random.uniform(10, 5000, n).round(2),
        "R_Score":    np.random.randint(1, 6, n),
        "F_Score":    np.random.randint(1, 6, n),
        "M_Score":    np.random.randint(1, 6, n),
        "RFM_Total":  np.random.randint(3, 16, n),
        "Segment":    np.random.choice(["Champions", "Loyal", "At-Risk", "Lost"], n),
    })


@pytest.fixture
def clean_df():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2010-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "CustomerID":  [f"C{i % 50}" for i in range(n)],
        "Invoice":     [f"I{i}" for i in range(n)],
        "StockCode":   [f"SKU{i % 20}" for i in range(n)],
        "Description": [f"Product {i % 20}" for i in range(n)],
        "InvoiceDate": dates,
        "Quantity":    np.random.randint(1, 20, n),
        "Price":       np.random.uniform(1, 50, n).round(2),
        "TotalPrice":  np.random.uniform(5, 500, n).round(2),
        "Country":     np.random.choice(["UK", "Germany", "France"], n),
    })


@pytest.fixture
def daily_df():
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", periods=120, freq="D")
    return pd.DataFrame({
        "ds":     dates,
        "y":      np.random.uniform(5000, 50000, 120).round(2),
        "orders": np.random.randint(10, 100, 120),
    })


# ── Segmentation tests ──────────────────────────────────────────────────────

class TestSegmentation:
    def test_prepare_features_shape(self, rfm_df):
        from src.models.segmentation import prepare_features
        X, scaler = prepare_features(rfm_df)
        assert X.shape == (len(rfm_df), 3)

    def test_prepare_features_scaled(self, rfm_df):
        from src.models.segmentation import prepare_features
        X, _ = prepare_features(rfm_df)
        assert abs(X.mean()) < 1.0  # approximately zero-centered

    def test_label_clusters(self, rfm_df):
        from src.models.segmentation import label_clusters
        labels = np.random.randint(0, 3, len(rfm_df))
        result = label_clusters(rfm_df, labels)
        assert "Cluster" in result.columns
        assert "ClusterLabel" in result.columns
        assert result["Cluster"].nunique() <= 3


# ── Churn tests ─────────────────────────────────────────────────────────────

class TestChurn:
    def test_label_churn_binary(self, rfm_df):
        from src.models.churn import label_churn
        result = label_churn(rfm_df)
        assert "Churned" in result.columns
        assert set(result["Churned"].unique()).issubset({0, 1})

    def test_label_churn_threshold(self, rfm_df):
        from src.models.churn import label_churn
        from src.utils.config import CHURN_THRESHOLD_DAYS
        result = label_churn(rfm_df)
        assert (result[result["Recency"] >= CHURN_THRESHOLD_DAYS]["Churned"] == 1).all()
        assert (result[result["Recency"] < CHURN_THRESHOLD_DAYS]["Churned"] == 0).all()

    def test_build_features_shape(self, rfm_df):
        from src.models.churn import label_churn, build_features
        rfm_df = label_churn(rfm_df)
        X, y = build_features(rfm_df)
        assert len(X) == len(rfm_df)
        assert len(y) == len(rfm_df)
        assert y.isin([0, 1]).all()


# ── Inventory tests ─────────────────────────────────────────────────────────

class TestInventory:
    def test_compute_z_score(self):
        from src.models.inventory import compute_z_score
        z_95 = compute_z_score(0.95)
        z_99 = compute_z_score(0.99)
        assert abs(z_95 - 1.645) < 0.01
        assert z_99 > z_95

    def test_sku_demand_stats(self, clean_df):
        from src.models.inventory import compute_sku_demand_stats
        stats = compute_sku_demand_stats(clean_df)
        assert "avg_daily_demand" in stats.columns
        assert "std_daily_demand" in stats.columns
        assert (stats["avg_daily_demand"] >= 0).all()

    def test_reorder_metrics(self, clean_df):
        from src.models.inventory import compute_sku_demand_stats, compute_reorder_metrics
        from src.utils.config import LEAD_TIME_DAYS
        stats = compute_sku_demand_stats(clean_df)
        metrics = compute_reorder_metrics(stats, LEAD_TIME_DAYS, z=1.645)
        assert "reorder_point" in metrics.columns
        assert "safety_stock" in metrics.columns
        assert "risk_tier" in metrics.columns
        assert (metrics["reorder_point"] >= 0).all()

    def test_risk_tiers_valid(self, clean_df):
        from src.models.inventory import compute_sku_demand_stats, compute_reorder_metrics
        stats = compute_sku_demand_stats(clean_df)
        metrics = compute_reorder_metrics(stats, 7, z=1.645)
        assert set(metrics["risk_tier"].unique()).issubset({"High", "Medium", "Low"})


# ── Forecasting tests ───────────────────────────────────────────────────────

class TestForecasting:
    def test_compute_mape(self):
        from src.models.forecasting import compute_mape
        actual    = pd.Series([100.0, 200.0, 300.0])
        predicted = pd.Series([110.0, 190.0, 300.0])
        mape = compute_mape(actual, predicted)
        assert 0 < mape < 20

    def test_mape_perfect_prediction(self):
        from src.models.forecasting import compute_mape
        s = pd.Series([100.0, 200.0, 300.0])
        assert compute_mape(s, s) == pytest.approx(0.0)

    def test_build_forecast_df(self, daily_df):
        from src.models.forecasting import build_forecast_df
        # Mock result dict
        horizon = 30
        mock_result = {
            "forecast":  pd.Series(np.random.uniform(10000, 40000, horizon)),
            "lower_ci":  pd.Series(np.random.uniform(5000,  30000, horizon)),
            "upper_ci":  pd.Series(np.random.uniform(30000, 50000, horizon)),
        }
        combined = build_forecast_df(daily_df, mock_result, horizon)
        assert "ds" in combined.columns
        assert "type" in combined.columns
        assert "forecast" in combined["type"].values
        assert "actual"   in combined["type"].values
        assert len(combined) == len(daily_df) + horizon