"""Page 4 — Inventory Dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard.utils import load_inventory, not_found_msg

st.set_page_config(page_title="Inventory Dashboard", page_icon="📦", layout="wide")
st.title("📦 Inventory Dashboard")
st.divider()

inv = load_inventory()
if inv is None:
    not_found_msg("src.models.inventory")
    st.stop()

with st.sidebar:
    st.header("Filters")
    tiers = ["All"] + sorted(inv["risk_tier"].dropna().unique().tolist())
    sel_tier = st.selectbox("Risk Tier", tiers)
    top_n = st.slider("Show top N SKUs", 10, 100, 25)

filtered = inv.copy()
if sel_tier != "All":
    filtered = filtered[filtered["risk_tier"] == sel_tier]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total SKUs",        f"{len(inv):,}")
c2.metric("High Risk SKUs",    f"{(inv['risk_tier']=='High').sum():,}")
c3.metric("Avg Reorder Point", f"{inv['reorder_point'].mean():,.0f} units")
c4.metric("Avg Safety Stock",  f"{inv['safety_stock'].mean():,.0f} units")
st.divider()

color_map = {"High":"#EF4444","Medium":"#F59E0B","Low":"#10B981"}
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Tier Distribution")
    tc = inv["risk_tier"].value_counts().reset_index()
    tc.columns = ["Risk Tier","Count"]
    fig = px.pie(tc, names="Risk Tier", values="Count", color="Risk Tier",
                 color_discrete_map=color_map, hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Reorder Point vs Avg Daily Demand")
    p = inv[inv["avg_daily_demand"] < inv["avg_daily_demand"].quantile(0.95)].copy()
    hover = ["StockCode","Description"] if "Description" in p.columns else ["StockCode"]
    fig2 = px.scatter(p, x="avg_daily_demand", y="reorder_point",
                      color="risk_tier", color_discrete_map=color_map,
                      size="total_sold", size_max=15, opacity=0.6,
                      hover_data=hover,
                      labels={"avg_daily_demand":"Avg Daily Demand","reorder_point":"Reorder Point"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader(f"🚨 Top {min(top_n,20)} High-Risk SKUs — Reorder Point")
hr = inv[inv["risk_tier"]=="High"].sort_values("avg_daily_demand",ascending=False).head(20)
st.plotly_chart(px.bar(hr.sort_values("reorder_point"), x="reorder_point", y="StockCode",
                       orientation="h", color="avg_daily_demand",
                       color_continuous_scale="Reds",
                       labels={"reorder_point":"Reorder Point","StockCode":"SKU"},
                       height=500),
               use_container_width=True)

st.divider()
st.subheader("SKU Inventory Table")
cols = ["StockCode","avg_daily_demand","safety_stock","reorder_point","total_sold","risk_tier"]
if "Description" in filtered.columns:
    cols.insert(1, "Description")
tbl = filtered[cols].sort_values("avg_daily_demand",ascending=False).head(200).reset_index(drop=True)
st.dataframe(tbl, use_container_width=True)