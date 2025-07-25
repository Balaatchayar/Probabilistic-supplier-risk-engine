import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(layout="wide")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("vendor_delay_data.csv")
    df['Delay_Days'] = df['Delay_Days'].astype(int)
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'])
    return df

df = load_data()

# -------------------- TITLE --------------------
st.title("Probabilistic Supplier Risk Engine")
st.header("Know your vendors. Trust your timelines.")

# -------------------- FILTERS --------------------
st.markdown("---")
st.header("🔍 Filter Your Data")

col1, col2, col3 = st.columns(3)

with col1:
    material = st.selectbox("📦 Select Material", sorted(df['Material_ID'].unique()))

with col2:
    location = st.selectbox("🌍 Filter by Location", ['All'] + sorted(df['Location'].unique()))

with col3:
    months = st.slider("🗓️ Analyze Last N Months", 1, 6, 3)

# Apply filters
filtered_df = df[df['Material_ID'] == material]
if location != 'All':
    filtered_df = filtered_df[filtered_df['Location'] == location]

latest_date = filtered_df['Delivery_Date'].max()
min_date = latest_date - pd.DateOffset(months=months)
filtered_df = filtered_df[filtered_df['Delivery_Date'] >= min_date]

# -------------------- SUMMARY TABLE --------------------
st.markdown("### 📊 Vendor Summary Statistics")

vendor_stats = filtered_df.groupby('Vendor_ID').agg(
    avg_delay=('Delay_Days', 'mean'),
    late_pct=('Delay_Days', lambda x: (x > 0).mean()),
    blank_reason_pct=('Reason', lambda x: (x == '').mean())
).reset_index()

vendor_stats['risk_score'] = vendor_stats['avg_delay'] + 2 * vendor_stats['late_pct']
vendor_stats = vendor_stats.sort_values(by='risk_score', ascending=False)

st.dataframe(
    vendor_stats.style.format({
        'avg_delay': '{:.2f}',
        'late_pct': '{:.1%}',
        'blank_reason_pct': '{:.1%}',
        'risk_score': '{:.2f}'
    }),
    use_container_width=True
)

# -------------------- PLOTS --------------------
st.markdown("### 🔬 Vendor Drill-down: PDF & PMF")

selected_vendors = st.multiselect("📌 Select Vendor(s) to Analyze", vendor_stats['Vendor_ID'])

for vendor in selected_vendors:
    vendor_data = filtered_df[filtered_df['Vendor_ID'] == vendor]
    avg_delay = vendor_data['Delay_Days'].mean()
    late_pct = (vendor_data['Delay_Days'] > 0).mean() * 100
    zero_pct = (vendor_data['Delay_Days'] == 0).mean() * 100

    st.markdown(f"#### 🔍 Vendor `{vendor}`  | Avg Delay: `{avg_delay:.2f}` days | % Late: `{late_pct:.1f}%` | % On Time: `{zero_pct:.1f}%`")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.kdeplot(vendor_data['Delay_Days'], fill=True, color='skyblue', ax=ax)
        ax.axvline(0, color='red', linestyle='--', label='Zero Delay')
        ax.set_title(f"PDF - Vendor {vendor}")
        ax.set_xlabel("Delay Days")
        ax.set_ylabel("Probability Density")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        pmf = vendor_data['Delay_Days'].value_counts(normalize=True).sort_index()
        ax2.bar(pmf.index, pmf.values, color='salmon')
        ax2.set_title(f"PMF - Vendor {vendor}")
        ax2.set_xlabel("Exact Delay Days")
        ax2.set_ylabel("Probability")
        st.pyplot(fig2)
