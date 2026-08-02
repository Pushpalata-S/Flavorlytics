import pathlib
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_raw_data, clean_data_for_eda

st.set_page_config(page_title="EDA Dashboards — Flavorlytics", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")
st.caption("Interactive visual exploration of the Zomato Bangalore Dataset")

@st.cache_data
def get_eda_data():
    data_path = PROJECT_ROOT / "data" / "zomato.csv"
    raw_df = load_raw_data(data_path)
    cleaned_df = clean_data_for_eda(raw_df)
    return cleaned_df

df = get_eda_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⭐ Ratings & Votes",
    "💰 Cost Analysis",
    "🍴 Cuisines & Types",
    "📱 Online & Booking",
    "🏙️ City Breakdown"
])

# ---------------------------------------------------------
# TAB 1: RATINGS & VOTES
# ---------------------------------------------------------
with tab1:
    st.subheader("⭐ Rating Distribution & Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rate_hist = px.histogram(
            df.dropna(subset=['rate']),
            x="rate",
            nbins=30,
            title="Distribution of Restaurant Ratings",
            color_discrete_sequence=['#FF4B4B'],
            marginal="box"
        )
        fig_rate_hist.update_layout(xaxis_title="Rating (out of 5)", yaxis_title="Count")
        st.plotly_chart(fig_rate_hist, use_container_width=True)
        
    with col2:
        fig_votes_rate = px.scatter(
            df.dropna(subset=['rate', 'votes']),
            x="votes",
            y="rate",
            color="rate",
            size="votes",
            title="Votes vs Rating",
            color_continuous_scale="cividis"
        )
        fig_votes_rate.update_layout(xaxis_title="Number of Votes", yaxis_title="Rating")
        st.plotly_chart(fig_votes_rate, use_container_width=True)
        
    st.markdown("#### 📈 Cost vs Rating Scatter Analysis")
    fig_cost_rate = px.scatter(
        df.dropna(subset=['cost', 'rate']),
        x="cost",
        y="rate",
        color="rate",
        title="Approx Cost (for two) vs Rating",
        color_continuous_scale="viridis",
        hover_data=['name', 'city']
    )
    fig_cost_rate.update_layout(xaxis_title="Approx Cost for Two (₹)", yaxis_title="Rating")
    st.plotly_chart(fig_cost_rate, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: COST ANALYSIS
# ---------------------------------------------------------
with tab2:
    st.subheader("💰 Restaurant Cost Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cost_hist = px.histogram(
            df,
            x="cost",
            nbins=40,
            title="Distribution of Approx Cost for Two",
            color_discrete_sequence=['#FF8C00'],
            marginal="violin"
        )
        fig_cost_hist.update_layout(xaxis_title="Cost for Two (₹)", yaxis_title="Count")
        st.plotly_chart(fig_cost_hist, use_container_width=True)
        
    with col2:
        fig_cost_box = px.box(
            df,
            y="cost",
            x="online_order",
            color="online_order",
            title="Cost Distribution by Online Ordering Availability",
            color_discrete_sequence=['#36A2EB', '#FF6384']
        )
        fig_cost_box.update_layout(xaxis_title="Online Order Available", yaxis_title="Cost for Two (₹)")
        st.plotly_chart(fig_cost_box, use_container_width=True)

    st.markdown("#### 🍱 North Indian vs. South Indian Cost Comparison")
    north_indian = df[df['cuisines'] == "North Indian"]["cost"].dropna()
    south_indian = df[df['cuisines'] == "South Indian"]["cost"].dropna()
    
    plot_df = pd.DataFrame({
        "Cuisine": ["North Indian"] * len(north_indian) + ["South Indian"] * len(south_indian),
        "Cost": pd.concat([north_indian, south_indian], ignore_index=True)
    })
    
    fig_cuisine_cost = px.box(
        plot_df,
        x="Cuisine",
        y="Cost",
        color="Cuisine",
        title="Avg Cost Comparison: North Indian vs South Indian Outlets",
        color_discrete_sequence=['#E74C3C', '#2ECC71']
    )
    st.plotly_chart(fig_cuisine_cost, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: CUISINES & TYPES
# ---------------------------------------------------------
with tab3:
    st.subheader("🍴 Popular Cuisines & Restaurant Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_chains = df['name'].value_counts().head(20).reset_index()
        top_chains.columns = ['Restaurant Name', 'Outlet Count']
        fig_chains = px.bar(
            top_chains,
            x="Outlet Count",
            y="Restaurant Name",
            orientation="h",
            title="Top 20 Restaurant Chains by Outlet Count",
            color="Outlet Count",
            color_continuous_scale="cividis"
        )
        fig_chains.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_chains, use_container_width=True)
        
    with col2:
        top_cuisines = df['cuisines'].value_counts().head(20).reset_index()
        top_cuisines.columns = ['Cuisine Type', 'Count']
        fig_cuisines = px.bar(
            top_cuisines,
            x="Count",
            y="Cuisine Type",
            orientation="h",
            title="Top 20 Most Popular Cuisines",
            color="Count",
            color_continuous_scale="plasma"
        )
        fig_cuisines.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cuisines, use_container_width=True)

    st.markdown("#### 🗺️ City Heatmaps by Restaurant & Listed Type")
    
    city_rest = df.groupby(['city', 'rest_type']).size().reset_index(name='count')
    frequent_types = ["Quick Bites", "Casual Dining", "Cafe"]
    city_rest_filt = city_rest[city_rest['rest_type'].isin(frequent_types)]
    
    city_pivot = city_rest_filt.pivot(index='city', columns='rest_type', values='count').fillna(0).reset_index()
    city_pivot_melted = city_pivot.melt(id_vars='city', var_name='rest_type', value_name='count')
    
    fig_heatmap = px.density_heatmap(
        city_pivot_melted,
        x='rest_type',
        y='city',
        z='count',
        color_continuous_scale='cividis',
        title='Density Heatmap of Restaurant Types by City'
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: ONLINE ORDER & BOOKING
# ---------------------------------------------------------
with tab4:
    st.subheader("📱 Online Ordering & Table Booking Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        online_counts = df['online_order'].value_counts().reset_index()
        online_counts.columns = ['Online Order', 'Count']
        fig_online_pie = px.pie(
            online_counts,
            names='Online Order',
            values='Count',
            title='Online Ordering Availability Share',
            color='Online Order',
            color_discrete_map={'Yes': '#2ECC71', 'No': '#E74C3C'}
        )
        st.plotly_chart(fig_online_pie, use_container_width=True)
        
    with col2:
        booking_counts = df['book_table'].value_counts().reset_index()
        booking_counts.columns = ['Table Booking', 'Count']
        fig_booking_pie = px.pie(
            booking_counts,
            names='Table Booking',
            values='Count',
            title='Table Booking Availability Share',
            color='Table Booking',
            color_discrete_map={'Yes': '#3498DB', 'No': '#E67E22'}
        )
        st.plotly_chart(fig_booking_pie, use_container_width=True)

    st.markdown("#### 🏙️ Online Order & Table Booking Ratios by City")
    
    total_per_city = df.groupby("city").size().reset_index(name="total_count")
    online_per_city = df[df['online_order'] == 'Yes'].groupby("city").size().reset_index(name="online_count")
    booking_per_city = df[df['book_table'] == 'Yes'].groupby("city").size().reset_index(name="booking_count")
    
    merged_ratios = total_per_city.merge(online_per_city, on='city', how='left').merge(booking_per_city, on='city', how='left').fillna(0)
    merged_ratios['online_ratio'] = round((merged_ratios['online_count'] / merged_ratios['total_count']) * 100, 1)
    merged_ratios['booking_ratio'] = round((merged_ratios['booking_count'] / merged_ratios['total_count']) * 100, 1)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_online_city = px.bar(
            merged_ratios.sort_values('online_ratio', ascending=True),
            x='online_ratio',
            y='city',
            orientation='h',
            title='Online Order Ratio by City (%)',
            color='online_ratio',
            color_continuous_scale='cividis'
        )
        fig_online_city.update_layout(height=650)
        st.plotly_chart(fig_online_city, use_container_width=True)
        
    with col_b:
        fig_booking_city = px.bar(
            merged_ratios.sort_values('booking_ratio', ascending=True),
            x='booking_ratio',
            y='city',
            orientation='h',
            title='Table Booking Ratio by City (%)',
            color='booking_ratio',
            color_continuous_scale='cividis'
        )
        fig_booking_city.update_layout(height=650)
        st.plotly_chart(fig_booking_city, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: CITY BREAKDOWN
# ---------------------------------------------------------
with tab5:
    st.subheader("🏙️ City-Wise Breakdown & High Performance Ratios")
    
    city_counts = df.groupby('city')['name'].count().reset_index(name='count').sort_values('count', ascending=True)
    
    fig_city_counts = px.bar(
        city_counts,
        x='count',
        y='city',
        orientation='h',
        title='Total Restaurants Count by Listed City',
        color='count',
        color_continuous_scale='cividis'
    )
    fig_city_counts.update_layout(height=650)
    st.plotly_chart(fig_city_counts, use_container_width=True)
    
    st.markdown("#### 🌟 High-Rated (≥ 4.5) & Costly (≥ ₹1300) Outlets Proportion")
    
    high_rated_city = df[df['rate'] >= 4.5].groupby('city')['name'].count().reset_index(name='high_rated_count')
    high_cost_city = df[df['cost'] >= 1300].groupby('city')['name'].count().reset_index(name='high_cost_count')
    
    city_perf = city_counts.merge(high_rated_city, on='city', how='left').merge(high_cost_city, on='city', how='left').fillna(0)
    city_perf['high_rated_pct'] = round((city_perf['high_rated_count'] / city_perf['count']) * 100, 2)
    city_perf['high_cost_pct'] = round((city_perf['high_cost_count'] / city_perf['count']) * 100, 2)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        fig_hr_city = px.bar(
            city_perf.sort_values('high_rated_pct', ascending=True),
            x='high_rated_pct',
            y='city',
            orientation='h',
            title='High Rated Restaurants (≥4.5) Share by City (%)',
            color='high_rated_pct',
            color_continuous_scale='cividis'
        )
        fig_hr_city.update_layout(height=600)
        st.plotly_chart(fig_hr_city, use_container_width=True)
        
    with col_c2:
        fig_hc_city = px.bar(
            city_perf.sort_values('high_cost_pct', ascending=True),
            x='high_cost_pct',
            y='city',
            orientation='h',
            title='Costly Restaurants (≥₹1300) Share by City (%)',
            color='high_cost_pct',
            color_continuous_scale='cividis'
        )
        fig_hc_city.update_layout(height=600)
        st.plotly_chart(fig_hc_city, use_container_width=True)
