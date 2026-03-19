import streamlit as st
import pandas as pd
import numpy as np
import pybaseball as pyb
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt, patches
from baseball_field_viz import transform_coords, draw_field, spraychart

# Page Configuration
st.set_page_config(page_title="Baseball Visualization - Anthony Volpe", layout="wide")

# Main Title
st.title("⚾ Advanced Baseball Visualization")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Visualization Controls")

# Load Data
@st.cache_data
def load_data():
    # Fetching data for Anthony Volpe (ID: 683011)
    #WE CAN SET ANY DATE
    return pyb.statcast_batter('2025-03-01', '2025-11-01', player_id=683011)

with st.spinner('Loading data...'):
    data = load_data()

# Filter for batting events
batting_data = data[data['events'].notnull()].copy()

# Show basic info in sidebar
st.sidebar.metric("Total Pitches", len(data))
st.sidebar.metric("Batting Events", len(batting_data))

# Visualization Type Selector
viz_type = st.sidebar.selectbox(
    "Select Visualization Type",
    ["Strike Zone", "Field Map", "Trends", "Heatmaps", "Comparative Analysis"]
)

# 1. STRIKE ZONE VISUALIZATION
if viz_type == "Strike Zone":
    st.header("🎯 Strike Zone Analysis")
    st.markdown("This visualization shows pitch locations for batted balls.")

    col1, col2 = st.columns(2)

    with col1:
        # Strike Zone Graph
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.add_patch(
            patches.Rectangle((-0.83, 1.5), 1.66, 2, fill=False, edgecolor='black', lw=2, label='Strike Zone'))

        # Slider to filter by velocity
        min_speed = st.slider("Minimum Velocity (mph)",
                              min_value=int(batting_data['release_speed'].min()),
                              max_value=int(batting_data['release_speed'].max()),
                              value=int(batting_data['release_speed'].min()))

        filtered_data = batting_data[batting_data['release_speed'] >= min_speed]

        sns.scatterplot(data=filtered_data, x='plate_x', y='plate_z',
                        hue='events', alpha=0.7, ax=ax)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 5)
        ax.set_title(f"Strike Zones (Velocity >= {min_speed} mph)")
        ax.set_xlabel("Horizontal Position")
        ax.set_ylabel("Vertical Position")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # Strike Zone Heatmap
        st.subheader("Heatmap - Strike Zone")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))

        # Create 2D Heatmap
        heatmap_data = pd.crosstab(
            pd.cut(batting_data['plate_z'], bins=10),
            pd.cut(batting_data['plate_x'], bins=10)
        )
        sns.heatmap(heatmap_data, ax=ax_heat, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Frequency'})
        ax_heat.set_title("Pitch Density")
        ax_heat.set_xlabel("Horizontal Position")
        ax_heat.set_ylabel("Vertical Position")
        st.pyplot(fig_heat)
        plt.close(fig_heat)

# 2. INTERACTIVE FIELD MAP
elif viz_type == "Field Map":
    st.header("🏟️ Batting Map")
    st.markdown("Visualization of where batted balls land on the field.")

    # Color selector
    color_by = st.radio("Color by:", ['events', 'launch_speed', 'launch_angle'])

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    spraychart(ax2, batting_data, color_by=color_by,
               title=f'Batting Distribution - Colored by {color_by}')
    st.pyplot(fig2)
    plt.close(fig2)

# 3. TREND GRAPHS
elif viz_type == "Trends":
    st.header("📈 Trends and Statistics")
    st.markdown("Analysis of trends in hitter performance.")

    st.subheader("Exit Velocity Evolution")

    # Group by date
    batting_data['game_date'] = pd.to_datetime(batting_data['game_date'])
    daily_stats = batting_data.groupby('game_date').agg({
        'launch_speed': 'mean',
        'launch_angle': 'mean',
        'events': 'count'
    }).reset_index()

    fig_line = px.line(daily_stats, x='game_date', y='launch_speed',
                       title='Average Exit Velocity per Game',
                       labels={'launch_speed': 'Velocity (mph)', 'game_date': 'Date'})
    st.plotly_chart(fig_line, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Exit Velocity Distribution")
        fig_hist = px.histogram(batting_data, x='launch_speed', nbins=30,
                                title='Exit Velocity Histogram',
                                labels={'launch_speed': 'Velocity (mph)'})
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Velocity vs. Launch Angle")
        fig_scatter = px.scatter(batting_data, x='launch_speed', y='launch_angle',
                                 color='events', hover_data=['events'],
                                 title='Velocity-Angle Relationship',
                                 labels={'launch_speed': 'Velocity (mph)',
                                         'launch_angle': 'Angle (degrees)'})
        st.plotly_chart(fig_scatter, use_container_width=True)

# 4. ADVANCED HEATMAPS
elif viz_type == "Heatmaps":
    st.header("🔥 Advanced Heatmaps")
    st.markdown("Density visualization for different metrics.")

    metric = st.selectbox("Select metric for heatmap",
                          ['release_speed', 'launch_speed', 'launch_angle'])

    col1, col2 = st.columns(2)

    with col1:
        # Pitch Density Heatmap
        fig_h1, ax_h1 = plt.subplots(figsize=(8, 6))
        sns.kdeplot(data=data, x='plate_x', y='plate_z',
                    cmap='viridis', fill=True, ax=ax_h1, alpha=0.6)
        ax_h1.add_patch(patches.Rectangle((-0.83, 1.5), 1.66, 2,
                                          fill=False, edgecolor='red', lw=2))
        ax_h1.set_title(f'Pitch Density - {metric}')
        ax_h1.set_xlim(-2, 2)
        ax_h1.set_ylim(0, 5)
        st.pyplot(fig_h1)
        plt.close(fig_h1)

    with col2:
        # Effectiveness Heatmap
        fig_h2, ax_h2 = plt.subplots(figsize=(8, 6))

        if metric in batting_data.columns:
            pivot_table = batting_data.pivot_table(
                values=metric,
                index=pd.cut(batting_data['plate_z'], bins=8),
                columns=pd.cut(batting_data['plate_x'], bins=8),
                aggfunc='mean'
            )
            sns.heatmap(pivot_table, ax=ax_h2, cmap='RdYlBu_r',
                        annot=False, cbar_kws={'label': metric})
            ax_h2.set_title(f'Average {metric} per Zone')
        st.pyplot(fig_h2)
        plt.close(fig_h2)

# 5. COMPARATIVE ANALYSIS
else:
    st.header("📊 Comparative Analysis")
    st.markdown("Comparison of different metrics and pitch types.")

    st.subheader("Performance by Pitch Type")

    pitch_types = batting_data['pitch_type'].value_counts().head(10).index
    pitch_data = batting_data[batting_data['pitch_type'].isin(pitch_types)]

    fig_bar = px.bar(pitch_data.groupby('pitch_type').agg({
        'launch_speed': 'mean',
        'launch_angle': 'mean'
    }).reset_index(),
                     x='pitch_type', y=['launch_speed', 'launch_angle'],
                     barmode='group',
                     title='Averages by Pitch Type',
                     labels={'value': 'Value', 'pitch_type': 'Pitch Type'})
    st.plotly_chart(fig_bar, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Velocity Distribution by Type")
        fig_box1 = px.box(pitch_data, x='pitch_type', y='launch_speed',
                          title='Exit Velocity by Pitch Type')
        st.plotly_chart(fig_box1, use_container_width=True)

    with col2:
        st.subheader("Angle Distribution by Type")
        fig_box2 = px.box(pitch_data, x='pitch_type', y='launch_angle',
                          title='Launch Angle by Pitch Type')
        st.plotly_chart(fig_box2, use_container_width=True)

# Expandable Dataframe
with st.expander("View Full Data"):
    st.dataframe(batting_data)

# Summary Stats
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Stats")
if len(batting_data) > 0:
    st.sidebar.metric("Avg Exit Velocity",
                      f"{batting_data['launch_speed'].mean():.1f} mph")
    st.sidebar.metric("Avg Launch Angle",
                      f"{batting_data['launch_angle'].mean():.1f}°")
    st.sidebar.metric("Most Common Event",
                      batting_data['events'].mode().iloc[0] if len(batting_data) > 0 else "N/A")

