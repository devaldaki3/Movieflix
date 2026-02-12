"""
EDA Generator for Analytics Dashboard
Generates visualizations and insights for user behavior and content consumption analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import io
import base64
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.facecolor'] = '#181818'
plt.rcParams['axes.facecolor'] = '#2a2a2a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['grid.color'] = '#444444'


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#181818', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def load_data():
    """Load movie dataset"""
    try:
        df = pd.read_csv('./Artifacts/movies.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def generate_rating_distribution(df):
    """Generate rating distribution visualization"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ratings = df['vote_average'].dropna()
    ax.hist(ratings, bins=30, color='#e50914', edgecolor='#FAFA33', alpha=0.8)
    ax.axvline(ratings.mean(), color='#FAFA33', linestyle='--', linewidth=2, 
               label=f'Mean: {ratings.mean():.2f}')
    ax.set_title('User Rating Distribution', fontsize=14, fontweight='bold', color='#FAFA33')
    ax.set_xlabel('Average Rating', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig_to_base64(fig)


def generate_genre_analysis(df):
    """Generate top genres bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract genres
    all_genres = []
    for genres in df['genres'].dropna():
        genre_list = genres.split()
        all_genres.extend(genre_list)
    
    genre_counts = Counter(all_genres)
    top_genres = dict(genre_counts.most_common(10))
    
    bars = ax.barh(list(top_genres.keys()), list(top_genres.values()), 
                   color='#e50914', edgecolor='#FAFA33')
    ax.set_title('Top 10 Most Popular Genres', fontsize=14, fontweight='bold', color='#FAFA33')
    ax.set_xlabel('Number of Movies', fontsize=11)
    ax.set_ylabel('Genre', fontsize=11)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (genre, count) in enumerate(top_genres.items()):
        ax.text(count + 20, i, f'{count:,}', va='center', fontsize=9, color='white')
    
    ax.grid(alpha=0.3, axis='x')
    
    return fig_to_base64(fig)


def generate_release_trends(df):
    """Generate release year trends"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    year_counts = df['release_year'].value_counts().sort_index()
    year_counts = year_counts[year_counts.index >= 1980]
    
    ax.plot(year_counts.index, year_counts.values, color='#e50914', linewidth=2, marker='o', markersize=3)
    ax.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='#e50914')
    ax.set_title('Movie Release Trends (1980-Present)', fontsize=14, fontweight='bold', color='#FAFA33')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of Movies', fontsize=11)
    ax.grid(alpha=0.3)
    
    return fig_to_base64(fig)


def generate_budget_revenue(df):
    """Generate budget vs revenue scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    financial_df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    sample = financial_df.sample(min(1000, len(financial_df)))
    
    ax.scatter(sample['budget'], sample['revenue'], alpha=0.5, c='#e50914', s=30, edgecolors='#FAFA33', linewidth=0.5)
    ax.plot([0, sample['budget'].max()], [0, sample['budget'].max()], 'y--', alpha=0.5, label='Break-even', linewidth=2)
    ax.set_title('Budget vs Revenue Analysis', fontsize=14, fontweight='bold', color='#FAFA33')
    ax.set_xlabel('Budget ($)', fontsize=11)
    ax.set_ylabel('Revenue ($)', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig_to_base64(fig)


def generate_popularity_analysis(df):
    """Generate popularity distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    popularity = df['popularity'].dropna()
    popularity = popularity[popularity < popularity.quantile(0.95)]
    
    ax.hist(popularity, bins=50, color='#17a2b8', edgecolor='#FAFA33', alpha=0.8)
    ax.set_title('Movie Popularity Distribution', fontsize=14, fontweight='bold', color='#FAFA33')
    ax.set_xlabel('Popularity Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.grid(alpha=0.3)
    
    return fig_to_base64(fig)


def generate_runtime_analysis(df):
    """Generate runtime distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    runtime = df['runtime'].dropna()
    runtime = runtime[(runtime > 0) & (runtime < 300)]
    
    ax.hist(runtime, bins=40, color='#6f42c1', edgecolor='#FAFA33', alpha=0.8)
    ax.axvline(runtime.mean(), color='#FAFA33', linestyle='--', linewidth=2, 
               label=f'Mean: {runtime.mean():.0f} min')
    ax.axvline(runtime.median(), color='#e50914', linestyle='--', linewidth=2, 
               label=f'Median: {runtime.median():.0f} min')
    ax.set_title('Movie Runtime Distribution', fontsize=14, fontweight='bold', color='#FAFA33')
    ax.set_xlabel('Runtime (minutes)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig_to_base64(fig)


def generate_correlation_heatmap(df):
    """Generate correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    numeric_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
    corr_data = df[numeric_cols].dropna()
    correlation_matrix = corr_data.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={'color': 'white', 'fontsize': 10})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='#FAFA33', pad=20)
    
    # Set tick colors
    ax.tick_params(colors='white')
    
    return fig_to_base64(fig)


def get_key_statistics(df):
    """Get key statistics for display"""
    stats = {}
    
    # Basic stats
    stats['total_movies'] = len(df)
    stats['avg_rating'] = df['vote_average'].mean()
    stats['median_rating'] = df['vote_average'].median()
    stats['avg_runtime'] = df['runtime'].dropna().mean()
    
    # Financial stats
    financial_df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    stats['avg_budget'] = financial_df['budget'].mean()
    stats['avg_revenue'] = financial_df['revenue'].mean()
    
    financial_df['roi'] = (financial_df['revenue'] - financial_df['budget']) / financial_df['budget'] * 100
    roi_filtered = financial_df[financial_df['roi'] < 1000]['roi']
    stats['median_roi'] = roi_filtered.median()
    
    # Genre stats
    all_genres = []
    for genres in df['genres'].dropna():
        all_genres.extend(genres.split())
    genre_counts = Counter(all_genres)
    stats['top_genre'] = genre_counts.most_common(1)[0] if genre_counts else ('Unknown', 0)
    stats['total_genres'] = len(genre_counts)
    
    # Year stats
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    year_counts = df['release_year'].value_counts()
    stats['peak_year'] = year_counts.idxmax() if len(year_counts) > 0 else 'N/A'
    stats['peak_year_count'] = year_counts.max() if len(year_counts) > 0 else 0
    
    return stats


def generate_all_visualizations():
    """Generate all EDA visualizations"""
    df = load_data()
    if df is None:
        return None
    
    visualizations = {
        'rating_dist': generate_rating_distribution(df),
        'genre_analysis': generate_genre_analysis(df),
        'release_trends': generate_release_trends(df),
        'budget_revenue': generate_budget_revenue(df),
        'popularity': generate_popularity_analysis(df),
        'runtime': generate_runtime_analysis(df),
        'correlation': generate_correlation_heatmap(df),
        'statistics': get_key_statistics(df)
    }
    
    return visualizations


if __name__ == '__main__':
    print("Generating EDA visualizations...")
    viz = generate_all_visualizations()
    if viz:
        print("✅ Visualizations generated successfully!")
        print(f"   Total visualizations: {len(viz) - 1}")  # -1 for statistics
    else:
        print("❌ Failed to generate visualizations")
