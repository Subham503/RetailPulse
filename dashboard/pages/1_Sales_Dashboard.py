"""Page 1 — Sales Dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dashboard.utils import load_sales, not_found_msg

st.set_page_config(page_title="Sales Dashboard", page_icon="📊", layout="wide")
st.title("📊 Sales Dashboard")
st.divider()

df = load_sales()
if df is None:
    not_found_msg("src.data.clean")
    st.stop()

with st.sidebar:
    st.header("Filters")
    years = sorted(df["Year"].dropna().unique().astype(int).tolist())
    sel_years = st.multiselect("Year", years, default=years)
    countries = ["All"] + sorted(df["Country"].unique().tolist())
    sel_country = st.selectbox("Country", countries)

filtered = df[df["Year"].isin(sel_years)]
if sel_country != "All":
    filtered = filtered[filtered["Country"] == sel_country]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue",    f"£{filtered['TotalPrice'].sum():,.0f}")
c2.metric("Total Orders",     f"{filtered['Invoice'].nunique():,}")
c3.metric("Avg Order Value",  f"£{filtered.groupby('Invoice')['TotalPrice'].sum().mean():,.2f}")
c4.metric("Total Items Sold", f"{filtered['Quantity'].sum():,.0f}")
st.divider()

st.subheader("Monthly Revenue Trend")
monthly = (filtered.groupby("YearMonth")["TotalPrice"].sum().reset_index()
           .rename(columns={"TotalPrice":"Revenue"}).sort_values("YearMonth"))
fig = px.line(monthly, x="YearMonth", y="Revenue", markers=True,
              color_discrete_sequence=["#4F46E5"],
              labels={"YearMonth":"Month","Revenue":"Revenue (£)"})
fig.update_layout(xaxis_tickangle=-45, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Products by Revenue")
    top = (filtered.groupby("Description")["TotalPrice"].sum().nlargest(10)
           .reset_index().sort_values("TotalPrice").rename(columns={"TotalPrice":"Revenue"}))
    st.plotly_chart(px.bar(top, x="Revenue", y="Description", orientation="h",
                           color="Revenue", color_continuous_scale="Blues",
                           labels={"Description":""}),
                   use_container_width=True)

with col2:
    st.subheader("Revenue by Country (Top 15)")
    top_c = (filtered.groupby("Country")["TotalPrice"].sum().nlargest(15)
             .reset_index().sort_values("TotalPrice").rename(columns={"TotalPrice":"Revenue"}))
    st.plotly_chart(px.bar(top_c, x="Revenue", y="Country", orientation="h",
                           color="Revenue", color_continuous_scale="Purples"),
                   use_container_width=True)

st.divider()
col3, col4 = st.columns(2)
with col3:
    st.subheader("Revenue by Day of Week")
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    dow = (filtered.groupby("DayOfWeek")["TotalPrice"].sum().reset_index()
           .rename(columns={"TotalPrice":"Revenue"}))
    dow["Day"] = dow["DayOfWeek"].map(dow_map)
    st.plotly_chart(px.bar(dow, x="Day", y="Revenue", color="Revenue",
                           color_continuous_scale="Greens"), use_container_width=True)

with col4:
    st.subheader("Seasonality Heatmap (Year × Month)")
    heat = (filtered.groupby(["Year","Month"])["TotalPrice"].sum().reset_index()
            .rename(columns={"TotalPrice":"Revenue"}))
    pivot = heat.pivot(index="Year", columns="Month", values="Revenue").fillna(0)
    st.plotly_chart(px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto"),
                   use_container_width=True)

with st.expander("🔍 Raw Data Sample"):
    st.dataframe(filtered.head(500), use_container_width=True)