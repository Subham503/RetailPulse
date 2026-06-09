"""Page 2 — Customer Dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.express as px
import pandas as pd
from dashboard.utils import load_rfm, not_found_msg

st.set_page_config(page_title="Customer Dashboard", page_icon="👥", layout="wide")
st.title("👥 Customer Dashboard")
st.divider()

rfm = load_rfm()
if rfm is None:
    not_found_msg("src.features.rfm")
    st.stop()

with st.sidebar:
    st.header("Filters")
    if "Segment" in rfm.columns:
        segs = ["All"] + sorted(rfm["Segment"].dropna().unique().tolist())
        sel_seg = st.selectbox("Segment", segs)
    else:
        sel_seg = "All"

filtered = rfm.copy()
if sel_seg != "All":
    filtered = filtered[filtered["Segment"] == sel_seg]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers",     f"{len(filtered):,}")
c2.metric("Avg Monetary",  f"£{filtered['Monetary'].mean():,.0f}")
c3.metric("Avg Recency",   f"{filtered['Recency'].mean():.0f} days")
if "ChurnProb" in filtered.columns:
    high = (filtered["ChurnProb"] >= 0.7).sum()
    c4.metric("High Churn Risk", f"{high:,}")
else:
    c4.metric("Avg Frequency", f"{filtered['Frequency'].mean():.1f}")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Segments")
    if "Segment" in rfm.columns:
        seg_c = rfm["Segment"].value_counts().reset_index()
        seg_c.columns = ["Segment","Count"]
        fig = px.pie(seg_c, names="Segment", values="Count", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Recency vs Monetary")
    color_col = "Segment" if "Segment" in rfm.columns else "R_Score"
    fig2 = px.scatter(rfm.sample(min(2000,len(rfm)), random_state=42),
                      x="Recency", y="Monetary", color=color_col,
                      size="Frequency", size_max=15, opacity=0.6,
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={"Recency":"Recency (days)","Monetary":"Monetary (£)"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
col3, col4 = st.columns(2)
with col3:
    if "ChurnProb" in rfm.columns:
        st.subheader("Churn Probability Distribution")
        fig3 = px.histogram(rfm, x="ChurnProb", nbins=30,
                            color_discrete_sequence=["#EF4444"],
                            labels={"ChurnProb":"Churn Probability"})
        fig3.add_vline(x=0.5, line_dash="dash", annotation_text="Threshold 0.5")
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    if "RFM_Total" in rfm.columns:
        st.subheader("RFM Total Score Distribution")
        fig4 = px.histogram(rfm, x="RFM_Total", nbins=13,
                            color_discrete_sequence=["#10B981"],
                            labels={"RFM_Total":"RFM Total (3–15)"})
        st.plotly_chart(fig4, use_container_width=True)

st.divider()
if "ChurnProb" in rfm.columns:
    st.subheader("🚨 Top 20 High Churn Risk Customers")
    at_risk = (rfm[["CustomerID","Recency","Frequency","Monetary","Segment","ChurnProb"]]
               .sort_values("ChurnProb", ascending=False).head(20).reset_index(drop=True))
    at_risk["Monetary"]  = at_risk["Monetary"].map("£{:,.2f}".format)
    at_risk["ChurnProb"] = at_risk["ChurnProb"].map("{:.1%}".format)
    st.dataframe(at_risk, use_container_width=True)

if "Segment" in rfm.columns:
    st.subheader("Segment Summary")
    summary = (rfm.groupby("Segment")
               .agg(Customers=("CustomerID","count"), Avg_Recency=("Recency","mean"),
                    Avg_Frequency=("Frequency","mean"), Avg_Monetary=("Monetary","mean"))
               .round(1).sort_values("Customers", ascending=False).reset_index())
    st.dataframe(summary, use_container_width=True)