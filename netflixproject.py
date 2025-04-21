#!/usr/bin/env python
# coding: utf-8



# ================================
# Data Cleaning and Preprocessing
# ================================

# Filled missing values in 'director', 'cast', 'country', 'rating', and 'duration' with 'Unknown'
# Converted 'date_added' to datetime format and extracted 'year_added' and 'month_added'
# Removed rows with missing 'date_added' entries
# Extracted numeric duration and duration type from 'duration' column (e.g., "90 min", "2 seasons")
# Normalized duration type values (e.g., converted 'seasons' to 'season', 'mins' to 'min')
# Converted TV show durations from seasons to approximate minutes (1 season ≈ 450 mins)
# Stripped whitespace from string columns (e.g., 'title', 'director', 'cast', 'country', etc.)
# Removed duplicate rows

# =================================
# Feature Engineering
# =================================

# Created 'duration_num' for numeric duration values
# Created 'duration_type' for type of duration (e.g., min, season)
# Created 'standard_duration_mins' to standardize movie and TV show durations in minutes
# Extracted 'year_added' and 'month_added' from 'date_added' for temporal analysis
# Extracted most frequent actors using the 'cast' column
# Dropped rows where 'country' is 'Unknown' or missing
# Prepared data for visualization and grouped data for analysis (e.g., by year, type, country)

# ================================
# Data Export
# ================================

# Saved cleaned dataset to 'netflix_cleaned.csv'
#get_ipython().system('pip install tabulate')
#get_ipython().system('pip install streamlit')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st


st.set_page_config(layout="wide", page_title="Netflix Dashboard", page_icon="🍿")

st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; color: #333; }
    .css-1d391kg, .css-18e3th9 { background-color: #ffffff !important; }
    h1, h2, h3, h4, h5, h6, .stMetricValue, .stMetricLabel { color: #333; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stSelectbox, .stMultiSelect, .stSlider { background-color: #ffffff; color: #333; border: 1px solid #ccc; }
    </style>
""", unsafe_allow_html=True)

st.title("Netflix Titles Analysis Dashboard")


@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['director'].fillna('Unknown', inplace=True)
    df['country'].fillna('Unknown', inplace=True)
    df['cast'].fillna('Unknown', inplace=True)
    df['rating'].fillna('Unknown', inplace=True)
    df['duration'].fillna('Unknown', inplace=True)

    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)
    df['duration_type'] = df['duration'].str.extract('([a-zA-Z]+)').astype(str)
    df['duration_type'] = df['duration_type'].str.lower().str.strip()
    df['duration_type'] = df['duration_type'].replace({'seasons': 'season', 'mins': 'min'})

    df['standard_duration_mins'] = df.apply(
        lambda row: row['duration_num'] * 450 if row['duration_type'] == 'season' else row['duration_num'],
        axis=1
    )

    for col in ['title', 'director', 'cast', 'country', 'rating', 'listed_in', 'description']:
        df[col] = df[col].str.strip()

    df.drop_duplicates(inplace=True)
    mode_value = df['standard_duration_mins'].mode()[0]
    df['standard_duration_mins'].fillna(mode_value, inplace=True)

    return df

df = load_data()


st.sidebar.header("Filters")
type_filter = st.sidebar.multiselect("Select Type", options=df['type'].unique(), default=df['type'].unique())
years_available = sorted(df['year_added'].dropna().unique())
year_filter = st.sidebar.multiselect("Select Year(s)", options=years_available, default=years_available[-5:])

filtered_df = df[(df['type'].isin(type_filter)) & (df['year_added'] >= year_filter[0]) & (df['year_added'] <= year_filter[1])]


if st.checkbox("📋 Show Raw Data"):
    st.dataframe(filtered_df)


col1, col2, col3 = st.columns(3)
col1.metric("Total Titles", len(filtered_df))
col2.metric("Movies", len(filtered_df[filtered_df['type'] == 'Movie']))
col3.metric("TV Shows", len(filtered_df[filtered_df['type'] == 'TV Show']))


st.subheader("🎬 Type Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x='type', ax=ax1)
ax1.set_title('Number of Movies vs TV Shows')
st.pyplot(fig1)

st.subheader("📊 Pie Chart: Type Share")
type_counts = filtered_df['type'].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
ax2.set_title('Distribution of Movies and TV Shows')
st.pyplot(fig2)

st.subheader("📈 Average Duration Over Years")
fig3, ax3 = plt.subplots()
avg_duration = filtered_df.groupby('year_added')['standard_duration_mins'].mean()
ax3.plot(avg_duration.index, avg_duration.values, marker='o', color='green')
ax3.set_title('Average Duration Over Years')
ax3.set_xlabel('Year')
ax3.set_ylabel('Duration (mins)')
st.pyplot(fig3)

st.subheader("🔢 Ratings Distribution")
fig4, ax4 = plt.subplots()
rating_counts = filtered_df['rating'].value_counts()
ax4.bar(rating_counts.index, rating_counts.values, color='orange')
ax4.set_title('Distribution of Ratings')
ax4.set_xlabel('Rating')
ax4.set_ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(fig4)

st.subheader("📦 Duration Type by Content Type")
fig5, ax5 = plt.subplots()
duration_type_counts = filtered_df.groupby(['type', 'duration_type']).size().unstack()
duration_type_counts.plot(kind='bar', stacked=True, ax=ax5, colormap='viridis')
ax5.set_title('Type vs Duration Type')
st.pyplot(fig5)

# Country stats
st.subheader("🌍 Top 5 Countries by Total Titles")
df_country = filtered_df[filtered_df['country'] != 'Unknown']
country_type_counts = df_country.groupby(['country', 'type']).size().unstack().fillna(0)
top_countries = country_type_counts.sum(axis=1).sort_values(ascending=False).head(5)
fig6, ax6 = plt.subplots()
country_type_counts.loc[top_countries.index].plot(kind='barh', stacked=True, colormap='coolwarm', ax=ax6)
ax6.set_title('Movies & TV Shows by Country')
st.pyplot(fig6)

# Radar plot
st.subheader("📡 Rating Distribution Radar (United States)")
rating_dist = filtered_df.groupby(['country', 'rating']).size().unstack().fillna(0)
if 'United States' in rating_dist.index:
    us_data = rating_dist.loc['United States']
    labels = us_data.index
    values = us_data.values
    values = np.concatenate((values, [values[0]]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig7, ax7 = plt.subplots(subplot_kw={'polar': True})
    ax7.fill(angles, values, color='blue', alpha=0.6)
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(labels)
    ax7.set_title('US Rating Radar')
    ax7.set_yticklabels([])
    st.pyplot(fig7)

# Directors
st.subheader("🎥 Top 10 Directors by Count")
df_directors = filtered_df[filtered_df['director'] != 'Unknown']
top_directors = df_directors['director'].value_counts().head(10)
fig8, ax8 = plt.subplots()
top_directors.plot(kind='barh', color='skyblue', ax=ax8)
ax8.set_title('Top 10 Directors')
ax8.invert_yaxis()
st.pyplot(fig8)

# Country duration
st.subheader("🕒 Top Countries by Total Watch Time")
duration_by_country = filtered_df.groupby('country')['standard_duration_mins'].sum().sort_values(ascending=False).head(5)
fig9, ax9 = plt.subplots()
duration_by_country.plot(kind='barh', color='skyblue', ax=ax9)
ax9.set_title('Total Watch Time by Country')
ax9.set_xlabel('Minutes')
ax9.invert_yaxis()
st.pyplot(fig9)
