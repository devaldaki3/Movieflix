"""
Sentiment Analysis Module for Movie Reviews and Descriptions
Analyzes sentiment from movie overviews and generates insights
"""

import pandas as pd
import numpy as np
from collections import Counter
import re


class SentimentAnalyzer:
    """Sentiment analyzer using lexicon-based approach"""
    
    def __init__(self):
        # Positive words lexicon
        self.positive_words = {
            'love', 'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'best',
            'brilliant', 'perfect', 'beautiful', 'awesome', 'incredible', 'outstanding',
            'superb', 'magnificent', 'spectacular', 'thrilling', 'exciting', 'entertaining',
            'masterpiece', 'triumph', 'success', 'winner', 'epic', 'legendary', 'classic',
            'beloved', 'charming', 'delightful', 'enjoyable', 'fun', 'good', 'happy',
            'impressive', 'inspiring', 'interesting', 'powerful', 'remarkable', 'stunning',
            'terrific', 'touching', 'unforgettable', 'unique', 'uplifting', 'witty',
            'adventure', 'hero', 'victory', 'hope', 'joy', 'peace', 'friendship'
        }
        
        # Negative words lexicon
        self.negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst', 'poor', 'boring',
            'disappointing', 'waste', 'dull', 'weak', 'mediocre', 'failure', 'disaster',
            'mess', 'trash', 'garbage', 'pathetic', 'ridiculous', 'stupid', 'lame',
            'annoying', 'confusing', 'pointless', 'predictable', 'cliche', 'overrated',
            'underwhelming', 'forgettable', 'bland', 'tedious', 'slow', 'dragging',
            'villain', 'evil', 'dark', 'death', 'murder', 'crime', 'danger', 'fear',
            'threat', 'violence', 'war', 'destruction', 'chaos', 'tragedy', 'loss'
        }
        
        # Intensifiers
        self.intensifiers = {'very', 'extremely', 'absolutely', 'totally', 'completely', 'really'}
        
    def preprocess_text(self, text):
        """Preprocess text for sentiment analysis"""
        if pd.isna(text):
            return []
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        return words
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text
        Returns: sentiment score (-1 to 1), label (Positive/Negative/Neutral)
        """
        words = self.preprocess_text(text)
        
        if not words:
            return 0.0, 'Neutral'
        
        positive_count = 0
        negative_count = 0
        intensifier_multiplier = 1.0
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.intensifiers:
                intensifier_multiplier = 1.5
                continue
            
            # Count positive words
            if word in self.positive_words:
                positive_count += intensifier_multiplier
                intensifier_multiplier = 1.0
            
            # Count negative words
            elif word in self.negative_words:
                negative_count += intensifier_multiplier
                intensifier_multiplier = 1.0
            
            else:
                intensifier_multiplier = 1.0
        
        # Calculate sentiment score
        total = positive_count + negative_count
        if total == 0:
            score = 0.0
            label = 'Neutral'
        else:
            score = (positive_count - negative_count) / total
            
            if score > 0.2:
                label = 'Positive'
            elif score < -0.2:
                label = 'Negative'
            else:
                label = 'Neutral'
        
        return round(score, 3), label
    
    def analyze_batch(self, texts):
        """Analyze sentiment for multiple texts"""
        results = []
        for text in texts:
            score, label = self.analyze_sentiment(text)
            results.append({'score': score, 'label': label})
        return results


def analyze_movie_sentiments():
    """Analyze sentiments from movie overviews"""
    try:
        # Load data
        df = pd.read_csv('./Artifacts/movies.csv')
        
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        # Analyze overviews
        print("Analyzing movie overviews...")
        sentiments = []
        
        for overview in df['overview'].fillna(''):
            score, label = analyzer.analyze_sentiment(overview)
            sentiments.append({'score': score, 'label': label})
        
        # Create sentiment dataframe
        sentiment_df = pd.DataFrame(sentiments)
        
        # Calculate statistics
        stats = {
            'total_analyzed': len(sentiment_df),
            'positive_count': int((sentiment_df['label'] == 'Positive').sum()),
            'negative_count': int((sentiment_df['label'] == 'Negative').sum()),
            'neutral_count': int((sentiment_df['label'] == 'Neutral').sum()),
            'avg_sentiment_score': float(sentiment_df['score'].mean()),
            'positive_percentage': float((sentiment_df['label'] == 'Positive').sum() / len(sentiment_df) * 100),
            'negative_percentage': float((sentiment_df['label'] == 'Negative').sum() / len(sentiment_df) * 100),
            'neutral_percentage': float((sentiment_df['label'] == 'Neutral').sum() / len(sentiment_df) * 100)
        }
        
        # Get sentiment distribution by genre
        df['sentiment_label'] = sentiment_df['label']
        df['sentiment_score'] = sentiment_df['score']
        
        # Genre-wise sentiment
        genre_sentiments = {}
        for idx, row in df.iterrows():
            if pd.notna(row['genres']):
                genres = row['genres'].split()
                for genre in genres:
                    if genre not in genre_sentiments:
                        genre_sentiments[genre] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
                    
                    genre_sentiments[genre][row['sentiment_label'].lower()] += 1
                    genre_sentiments[genre]['total'] += 1
        
        # Calculate percentages for top genres
        top_genres = sorted(genre_sentiments.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
        genre_sentiment_data = {}
        
        for genre, counts in top_genres:
            total = counts['total']
            genre_sentiment_data[genre] = {
                'positive': round(counts['positive'] / total * 100, 1),
                'negative': round(counts['negative'] / total * 100, 1),
                'neutral': round(counts['neutral'] / total * 100, 1)
            }
        
        # Get most positive and negative movies
        df_with_titles = df[['title', 'sentiment_score', 'sentiment_label', 'vote_average']].copy()
        
        most_positive = df_with_titles.nlargest(10, 'sentiment_score')[['title', 'sentiment_score', 'vote_average']]
        most_negative = df_with_titles.nsmallest(10, 'sentiment_score')[['title', 'sentiment_score', 'vote_average']]
        
        stats['most_positive_movies'] = [
            {
                'title': row['title'],
                'sentiment_score': float(row['sentiment_score']),
                'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0.0
            }
            for _, row in most_positive.iterrows()
        ]
        
        stats['most_negative_movies'] = [
            {
                'title': row['title'],
                'sentiment_score': float(row['sentiment_score']),
                'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0.0
            }
            for _, row in most_negative.iterrows()
        ]
        
        stats['genre_sentiments'] = genre_sentiment_data
        
        # Sentiment distribution for chart
        stats['sentiment_distribution'] = {
            'Positive': stats['positive_count'],
            'Neutral': stats['neutral_count'],
            'Negative': stats['negative_count']
        }
        
        return stats
        
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("🎭 Running Sentiment Analysis...")
    print("=" * 60)
    
    results = analyze_movie_sentiments()
    
    if results:
        print(f"\n✅ Analysis Complete!")
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Movies Analyzed: {results['total_analyzed']:,}")
        print(f"   Positive: {results['positive_count']:,} ({results['positive_percentage']:.1f}%)")
        print(f"   Neutral: {results['neutral_count']:,} ({results['neutral_percentage']:.1f}%)")
        print(f"   Negative: {results['negative_count']:,} ({results['negative_percentage']:.1f}%)")
        print(f"   Average Sentiment Score: {results['avg_sentiment_score']:.3f}")
        
        print(f"\n🎬 Top 3 Most Positive Movies:")
        for i, movie in enumerate(results['most_positive_movies'][:3], 1):
            print(f"   {i}. {movie['title']} (Score: {movie['sentiment_score']:.3f})")
        
        print(f"\n😞 Top 3 Most Negative Movies:")
        for i, movie in enumerate(results['most_negative_movies'][:3], 1):
            print(f"   {i}. {movie['title']} (Score: {movie['sentiment_score']:.3f})")
    else:
        print("❌ Analysis failed")
