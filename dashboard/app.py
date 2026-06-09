"""RetailPulse — Streamlit Dashboard Home (Safe Version)."""

import streamlit as st
import sys
from pathlib import Path

# =============================
# FIX IMPORT PATH (CRITICAL)
# =============================
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.utils import load_sales, load_rfm, load_forecast, load_inventory


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="RetailPulse",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛒 RetailPulse")
st.markdown("**ML-Powered Retail Analytics Platform** — UCI Online Retail II (2009–2011)")
st.divider()


# =============================
# SAFE DATA LOADING
# =============================
st.subheader("📂 Data Status")

try:
    sales = load_sales()
    rfm = load_rfm()
    forecast = load_forecast()
    inventory = load_inventory()
except Exception as e:
    st.error("❌ Data loading failed")
    st.exception(e)
    st.stop()


# =============================
# STATUS METRICS
# =============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Sales Data", "✅ Ready" if sales is not None else "❌ Missing")
col2.metric("RFM / Churn", "✅ Ready" if rfm is not None else "❌ Missing")
col3.metric("Forecast", "✅ Ready" if forecast is not None else "❌ Missing")
col4.metric("Inventory", "✅ Ready" if inventory is not None else "❌ Missing")

st.divider()


# =============================
# KPI SECTION (SAFE)
# =============================
if sales is not None and not sales.empty:

    st.subheader("📊 Key Metrics")

    # numeric safety
    if "TotalPrice" in sales.columns:
        sales["TotalPrice"] = sales["TotalPrice"].fillna(0)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Revenue", f"£{sales['TotalPrice'].sum():,.0f}")
    c2.metric("Total Orders", f"{sales['Invoice'].nunique():,}" if "Invoice" in sales else "N/A")
    c3.metric("Customers", f"{sales['CustomerID'].nunique():,}" if "CustomerID" in sales else "N/A")
    c4.metric("SKUs", f"{sales['StockCode'].nunique():,}" if "StockCode" in sales else "N/A")
    c5.metric("Countries", f"{sales['Country'].nunique():,}" if "Country" in sales else "N/A")

st.divider()


# =============================
# NAVIGATION GUIDE
# =============================
st.subheader("🧭 Navigate")

st.markdown("""
| Page                      | What you'll find |
|--------------------------|------------------|
| 📊 Sales Dashboard      | Revenue trends, top products, country breakdown |
| 👥 Customer Dashboard    | RFM segments, churn risk, CLV analysis |
| 📈 Forecast Dashboard    | Demand forecasting with ML predictions |
| 📦 Inventory Dashboard   | Stock risk, reorder alerts, safety stock |

👉 Use the **sidebar (left)** to switch pages.
""")


# =============================
# DEBUG HELPER (optional but useful)
# =============================
with st.expander("🔧 Debug Info (click to expand)"):
    st.write("Sales shape:", None if sales is None else sales.shape)
    st.write("RFM shape:", None if rfm is None else rfm.shape)
    st.write("Forecast shape:", None if forecast is None else forecast.shape)
    st.write("Inventory shape:", None if inventory is None else inventory.shape)