"""
Simplified EDA Generator - Returns statistics without heavy visualization libraries
"""

import pandas as pd
import numpy as np
from collections import Counter
import json


def load_data():
    """Load movie dataset"""
    try:
        df = pd.read_csv('./Artifacts/movies.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def get_key_statistics(df):
    """Get key statistics for display"""
    stats = {}
    
    # Basic stats
    stats['total_movies'] = int(len(df))
    stats['avg_rating'] = float(df['vote_average'].mean())
    stats['median_rating'] = float(df['vote_average'].median())
    stats['avg_runtime'] = float(df['runtime'].dropna().mean())
    
    # Financial stats
    financial_df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    stats['avg_budget'] = float(financial_df['budget'].mean())
    stats['avg_revenue'] = float(financial_df['revenue'].mean())
    
    financial_df['roi'] = (financial_df['revenue'] - financial_df['budget']) / financial_df['budget'] * 100
    roi_filtered = financial_df[financial_df['roi'] < 1000]['roi']
    stats['median_roi'] = float(roi_filtered.median())
    
    # Genre stats
    all_genres = []
    for genres in df['genres'].dropna():
        all_genres.extend(genres.split())
    genre_counts = Counter(all_genres)
    top_genre = genre_counts.most_common(1)[0] if genre_counts else ('Unknown', 0)
    stats['top_genre'] = [top_genre[0], int(top_genre[1])]
    stats['total_genres'] = len(genre_counts)
    
    # Get top 10 genres for chart
    top_10_genres = dict(genre_counts.most_common(10))
    stats['genre_distribution'] = {k: int(v) for k, v in top_10_genres.items()}
    
    # Year stats
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    year_counts = df['release_year'].value_counts()
    stats['peak_year'] = int(year_counts.idxmax()) if len(year_counts) > 0 else 0
    stats['peak_year_count'] = int(year_counts.max()) if len(year_counts) > 0 else 0
    
    # Rating distribution
    rating_bins = [0, 2, 4, 6, 8, 10]
    rating_labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
    df['rating_bin'] = pd.cut(df['vote_average'], bins=rating_bins, labels=rating_labels, include_lowest=True)
    rating_dist = df['rating_bin'].value_counts().sort_index()
    stats['rating_distribution'] = {str(k): int(v) for k, v in rating_dist.items()}
    
    # Popularity stats
    stats['avg_popularity'] = float(df['popularity'].mean())
    stats['max_popularity'] = float(df['popularity'].max())
    
    # Top 10 popular movies
    top_movies = df.nlargest(10, 'popularity')[['title', 'popularity', 'vote_average']]
    stats['top_popular_movies'] = [
        {
            'title': row['title'],
            'popularity': float(row['popularity']),
            'rating': float(row['vote_average'])
        }
        for _, row in top_movies.iterrows()
    ]
    
    # Year-wise release trends (last 20 years)
    recent_years = df[df['release_year'] >= 2004]['release_year'].value_counts().sort_index()
    stats['yearly_releases'] = {int(k): int(v) for k, v in recent_years.items()}
    
    return stats


def generate_all_statistics():
    """Generate all EDA statistics"""
    df = load_data()
    if df is None:
        return None
    
    statistics = get_key_statistics(df)
    
    return {
        'statistics': statistics,
        'success': True
    }


if __name__ == '__main__':
    print("Generating EDA statistics...")
    stats = generate_all_statistics()
    if stats:
        print("✅ Statistics generated successfully!")
        print(json.dumps(stats['statistics'], indent=2))
    else:
        print("❌ Failed to generate statistics")
