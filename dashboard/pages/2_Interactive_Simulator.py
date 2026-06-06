import streamlit as st
import pandas as pd
import plotly.express as px
from portal_utils import inject_custom_css, safe_api_get

# Page configuration
st.set_page_config(page_title="Interactive Simulator - Credit Portal", page_icon="🛡️", layout="wide")
inject_custom_css()

st.title("🎛️ Scorecard Interactive Simulator")
st.markdown("<p style='color: #cbd5e1; font-size: 1.1rem; margin-top: -10px; font-weight: 700;'>Select a feature to explore its Weight-of-Evidence (WoE) intervals and scaled credit score points.</p>", unsafe_allow_html=True)
st.markdown("---")

# Fetch selected features list from API
importance_data = safe_api_get("/feature_importance")
features = importance_data.get("features", [])

if not features:
    # Fallback list if API is offline
    features = [
        "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED", 
        "AMT_CREDIT", "AMT_ANNUITY", "DAYS_LAST_PHONE_CHANGE", 
        "BUREAU_DEBT_CREDIT_RATIO", "DPD_MAX_INSTALL", "DPD_MAX_CC", "DPD_MAX_POS"
    ]

# Feature Selector
selected_feature = st.selectbox(
    "Select Feature to inspect:",
    features,
    index=0
)

# Fetch points mapping from API simulator endpoint
with st.spinner("Fetching scorecard rules..."):
    simulator_data = safe_api_get(f"/run_score_simulator/{selected_feature}")

if simulator_data:
    ftype = simulator_data.get("type", "numerical")
    points_dict = simulator_data.get("points", {})
    
    # Organize data into a DataFrame
    sim_rows = []
    for bin_label, pts in points_dict.items():
        sim_rows.append({
            "Bin/Interval": bin_label,
            "Points": pts
        })
    df_sim = pd.DataFrame(sim_rows)
    
    # Custom display layout
    col_table, col_chart = st.columns([2, 3])
    
    with col_table:
        st.subheader("📋 Scorecard Point Rules")
        st.markdown(f"**Feature:** `{selected_feature}`  \n**Type:** {ftype.capitalize()}")
        
        # Display table
        st.dataframe(
            df_sim,
            use_container_width=True,
            hide_index=True
        )
        
        # Additional contextual note based on feature type
        if ftype == "numerical":
            st.info("💡 Note: For numerical inputs, the value is evaluated in bins [start, end) where 'edges' represent boundaries. A missing/NaN value maps to the 'Missing' category.")
        else:
            st.info("💡 Note: Categorical inputs are grouped internally. Unrecognized categories fallback to 'Other' or 'Missing'.")
            
    with col_chart:
        st.subheader("📊 Points Distribution Visual")
        
        # Create horizontal bar chart
        # Color coding: Green for positive points, Red for negative points
        df_sim["Color"] = df_sim["Points"].apply(lambda x: "Adds Points (Low Risk)" if x >= 0 else "Subtracts Points (High Risk)")
        
        fig = px.bar(
            df_sim,
            x="Points",
            y="Bin/Interval",
            color="Color",
            orientation="h",
            color_discrete_map={
                "Adds Points (Low Risk)": "#10b981",
                "Subtracts Points (High Risk)": "#f43f5e"
            },
            labels={"Points": "Score Points", "Bin/Interval": "Bin Boundary"},
            category_orders={"Bin/Interval": df_sim["Bin/Interval"].tolist()}
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Inter, sans-serif', size=13),
            xaxis=dict(showgrid=True, gridcolor='#1e2d4a'),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Unable to load simulator data for the selected feature. Make sure the FastAPI backend is running.")
