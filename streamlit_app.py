# streamlit_app.py
# A Streamlit application for body composition analysis and reporting with goals, milestones, and PDF export

import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import pytz
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io

# --- Helper Functions ---

def load_data(files, timezone):
    dfs = []
    tz = pytz.timezone(timezone)
    for file in files:
        name = file.name.lower()
        if name.endswith('.csv'):
            df = pd.read_csv(file)
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif name.endswith('.json'):
            df = pd.read_json(file)
        else:
            continue
        dfs.append(df)
    if not dfs:
        st.error("No valid data files uploaded.")
        return None
    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_localize(None).dt.tz_localize(tz)
    df.sort_values('timestamp', inplace=True)
    return df

def clean_validate(df):
    ranges = {
        'weight_kg': (30, 300),
        'body_fat_percent': (3, 75),
        'bmi': (10, 60),
        'muscle_mass_kg': (10, 200),
        'visceral_fat': (1, 50),
        'bone_mass_kg': (1, 20),
        'water_percent': (20, 80),
        'bmr_kcal': (800, 5000)
    }
    for col, (low, high) in ranges.items():
        if col in df.columns:
            mask = ~df[col].between(low, high)
            if mask.any():
                st.warning(f"Out-of-range values detected in '{col}' (expected {low}-{high}):")
                st.write(df.loc[mask, ['timestamp', col]])
    return df

def feature_engineer(df, short_window, long_window):
    metrics = ['weight_kg', 'body_fat_percent', 'bmi', 'muscle_mass_kg',
               'visceral_fat', 'bone_mass_kg', 'water_percent', 'bmr_kcal']
    for col in metrics:
        if col in df.columns:
            df[f'{col}_roll_mean_{short_window}d'] = df[col].rolling(window=short_window).mean()
            df[f'{col}_roll_mean_{long_window}d'] = df[col].rolling(window=long_window).mean()
            df[f'{col}_ema_{short_window}d'] = df[col].ewm(span=short_window).mean()
            df[f'{col}_ema_{long_window}d'] = df[col].ewm(span=long_window).mean()
            df[f'{col}_weekly_delta'] = df[col].diff(periods=7)
            df[f'{col}_weekly_pct'] = df[col].pct_change(periods=7) * 100
    return df

def generate_insights(df, goal_weight, goal_bfp):
    insights = []
    if 'timestamp' not in df.columns:
        return "No timestamp column found."
    now = df['timestamp'].max()
    recent_30 = df[df['timestamp'] >= now - pd.Timedelta(days=30)]
    if not recent_30.empty and 'weight_kg' in df.columns:
        delta_30 = recent_30['weight_kg'].iloc[-1] - recent_30['weight_kg'].iloc[0]
        insights.append(f"**30-day weight change:** {delta_30:.2f} kg")
    if 'weight_kg_weekly_delta' in df.columns:
        fastest = df['weight_kg_weekly_delta'].min()
        insights.append(f"**Fastest weekly loss:** {fastest:.2f} kg")
    plateaus = (df['weight_kg_weekly_delta'] == 0).sum() if 'weight_kg_weekly_delta' in df.columns else 0
    insights.append(f"**Weeks with no weight change:** {plateaus}")
    if 'weight_kg' in df.columns:
        milestones = []
        for m in range(int(df['weight_kg'].max()), int(df['weight_kg'].min()) - 1, -1):
            if (df['weight_kg'] <= m).any():
                milestones.append(f"Reached ≤{m} kg")
        insights.append("**Milestones:** " + ", ".join(milestones))
    if goal_weight and df['weight_kg'].iloc[-1] <= goal_weight:
        insights.append(f"🎯 **Goal weight achieved!** Current: {df['weight_kg'].iloc[-1]:.1f} kg")
    if goal_bfp and 'body_fat_percent' in df.columns and df['body_fat_percent'].iloc[-1] <= goal_bfp:
        insights.append(f"🎯 **Goal body fat % achieved!** Current: {df['body_fat_percent'].iloc[-1]:.1f}%")
    return "\n\n".join(insights)

def download_report(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

def generate_template():
    cols = ["timestamp", "weight_kg", "body_fat_percent", "bmi", "muscle_mass_kg",
            "visceral_fat", "bone_mass_kg", "water_percent", "bmr_kcal"]
    example = pd.DataFrame({col: ["2025-06-01"] if col == "timestamp" else [None] for col in cols})
    csv = io.StringIO()
    example.to_csv(csv, index=False)
    return csv.getvalue()

# --- Streamlit App Layout ---

def main():
    st.title("Body Composition Tracker")
    st.sidebar.header("Settings")
    short_window = st.sidebar.number_input("Short rolling window (days)", min_value=1, max_value=30, value=7)
    long_window = st.sidebar.number_input("Long rolling window (days)", min_value=short_window+1, max_value=60, value=14)
    timezone = st.sidebar.text_input("Timezone", "Australia/Adelaide")
    goal_weight = st.sidebar.number_input("Target Weight (kg)", min_value=30.0, max_value=200.0, value=75.0)
    goal_bfp = st.sidebar.number_input("Target Body Fat %", min_value=5.0, max_value=50.0, value=15.0)

    st.markdown("Download a sample template to get started:")
    template_csv = generate_template()
    st.download_button("Download CSV Template", template_csv, file_name="body_composition_template.csv", mime="text/csv")

    files = st.file_uploader("Upload data files", type=['csv','xlsx','xls','json'], accept_multiple_files=True)

    if files:
        df = load_data(files, timezone)
        if df is None:
            return
        df = clean_validate(df)
        df = feature_engineer(df, short_window, long_window)

        st.subheader("Raw Data Preview")
        st.dataframe(df)

        for metric in ['weight_kg', 'body_fat_percent']:
            if metric in df.columns:
                st.subheader(f"Time Series: {metric.replace('_', ' ').title()}")
                fig = px.line(df, x='timestamp', y=[metric,
                                                   f"{metric}_ema_{short_window}d",
                                                   f"{metric}_ema_{long_window}d"],
                              labels={'value': metric, 'timestamp': 'Date'})
                st.plotly_chart(fig, use_container_width=True)

        if 'weight_kg' in df.columns and 'body_fat_percent' in df.columns:
            st.subheader("Weight vs Body Fat %")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['weight_kg'], name='Weight (kg)', yaxis='y1'))
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['body_fat_percent'], name='Body Fat (%)', yaxis='y2'))
            fig.update_layout(
                yaxis=dict(title='Weight (kg)'),
                yaxis2=dict(title='Body Fat (%)', overlaying='y', side='right')
            )
            st.plotly_chart(fig, use_container_width=True)

        if 'timestamp' in df.columns:
            df['interval_hours'] = df['timestamp'].diff().dt.total_seconds() / 3600.0
            st.subheader("Weigh-in Interval Histogram (hours)")
            fig = px.histogram(df.dropna(subset=['interval_hours']), x='interval_hours', nbins=20,
                               labels={'interval_hours': 'Hours'})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Insights")
        insights = generate_insights(df, goal_weight, goal_bfp)
        st.markdown(insights)

        csv_data = download_report(df)
        st.download_button("Download CSV Report", csv_data, file_name="body_composition_report.csv", mime="text/csv")

if __name__ == '__main__':
    main()
