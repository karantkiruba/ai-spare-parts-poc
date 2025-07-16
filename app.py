import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

# Dummy Data Generator
def generate_dummy_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
    parts = [f"P{str(i).zfill(3)}" for i in range(1, 21)]
    dealers = [f"DLR{str(i).zfill(2)}" for i in range(1, 6)]

    data = []
    for part in parts:
        for dealer in dealers:
            qty_base = np.random.randint(30, 150)
            seasonal_factor = np.random.uniform(0.8, 1.2, size=len(dates))
            for i, date in enumerate(dates):
                qty = int(qty_base * seasonal_factor[i] + np.random.normal(0, 10))
                data.append([date, part, dealer, max(qty, 0)])
    return pd.DataFrame(data, columns=['Date', 'PartID', 'DealerID', 'Quantity'])

# Load Dummy Data
data = generate_dummy_data()

# Sidebar Filters
st.sidebar.title("Filter Options")
selected_part = st.sidebar.selectbox("Select Part ID", sorted(data['PartID'].unique()))
selected_dealer = st.sidebar.selectbox("Select Dealer ID", sorted(data['DealerID'].unique()))

# Filtered Data
df_filtered = data[(data['PartID'] == selected_part) & (data['DealerID'] == selected_dealer)]

# Forecast using Rolling Mean
df_filtered = df_filtered.sort_values('Date')
df_filtered['RollingForecast'] = df_filtered['Quantity'].rolling(window=3).mean().fillna(method='bfill')

# Safety Stock (std dev method)
std_dev = df_filtered['Quantity'].std()
safety_stock = round(1.65 * std_dev)  # 95% service level

# Anomaly Detection
model = IsolationForest(contamination=0.1)
df_filtered['Anomaly'] = model.fit_predict(df_filtered[['Quantity']])
df_filtered['AnomalyFlag'] = df_filtered['Anomaly'].apply(lambda x: 'Abnormal' if x == -1 else 'Normal')

# Dashboard Layout
st.title("🧠 AI-Driven Spare Parts Planning Dashboard")
st.subheader(f"Dealer: {selected_dealer} | Part: {selected_part}")

col1, col2, col3 = st.columns(3)
col1.metric("Average Monthly Demand", round(df_filtered['Quantity'].mean(), 2))
col2.metric("Forecasted Qty (Next Month)", round(df_filtered['RollingForecast'].iloc[-1], 2))
col3.metric("Suggested Safety Stock", safety_stock)

# Time Series Chart
fig = px.line(df_filtered, x='Date', y='Quantity', title='Actual vs Forecasted Demand')
fig.add_scatter(x=df_filtered['Date'], y=df_filtered['RollingForecast'], mode='lines', name='Forecast')
st.plotly_chart(fig, use_container_width=True)

# Anomaly Table
st.markdown("### 📋 Anomaly Detection")
st.dataframe(df_filtered[['Date', 'Quantity', 'RollingForecast', 'AnomalyFlag']].tail(10), use_container_width=True)

# Final Planning Suggestion
suggested_order = round(df_filtered['RollingForecast'].iloc[-1] + safety_stock)
st.markdown("### ✅ Final Order Recommendation")
st.success(f"Recommended Order Quantity: **{suggested_order} units**")
