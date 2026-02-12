"""
Analytics Engine for Entertainment & Media Platform
Provides advanced analytics including:
- User Behavior Analysis
- Content Popularity Prediction
- Fake Engagement Detection
- Sentiment Trend Analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import pickle
import os


class UserBehaviorAnalyzer:
    """Analyze user behavior and content consumption patterns"""
    
    def __init__(self):
        self.user_sessions = []
        self.viewing_patterns = {}
        
    def track_user_activity(self, user_id, movie_id, action, timestamp=None):
        """Track user interactions"""
        if timestamp is None:
            timestamp = datetime.now()
            
        activity = {
            'user_id': user_id,
            'movie_id': movie_id,
            'action': action,  # 'view', 'search', 'recommend_click', 'review'
            'timestamp': timestamp
        }
        self.user_sessions.append(activity)
        
    def get_user_preferences(self, user_id):
        """Analyze user preferences based on viewing history"""
        user_activities = [a for a in self.user_sessions if a['user_id'] == user_id]
        
        if not user_activities:
            return None
            
        return {
            'total_views': len([a for a in user_activities if a['action'] == 'view']),
            'total_searches': len([a for a in user_activities if a['action'] == 'search']),
            'engagement_score': len(user_activities) * 10,
            'last_active': max([a['timestamp'] for a in user_activities])
        }
    
    def get_consumption_patterns(self):
        """Get overall consumption patterns"""
        if not self.user_sessions:
            return {
                'total_sessions': 0,
                'unique_users': 0,
                'avg_session_length': 0,
                'peak_hours': []
            }
            
        df = pd.DataFrame(self.user_sessions)
        
        return {
            'total_sessions': len(df),
            'unique_users': df['user_id'].nunique(),
            'most_viewed_movies': df[df['action'] == 'view']['movie_id'].value_counts().head(10).to_dict(),
            'action_distribution': df['action'].value_counts().to_dict()
        }


class PopularityPredictor:
    """Predict content popularity using ML"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, movie_data):
        """Extract features for popularity prediction"""
        features = []
        
        for _, movie in movie_data.iterrows():
            feature_vector = [
                movie.get('vote_count', 0),
                movie.get('vote_average', 0),
                len(str(movie.get('genres', '')).split(',')),
                len(str(movie.get('cast', '')).split(',')),
                movie.get('runtime', 0) if pd.notna(movie.get('runtime')) else 0,
                1 if movie.get('release_year', 2000) >= 2020 else 0,  # Recent release
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def train(self, movie_data, popularity_scores):
        """Train the popularity prediction model"""
        X = self.prepare_features(movie_data)
        y = popularity_scores
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
    def predict_popularity(self, movie_features):
        """Predict popularity score for a movie"""
        if not self.is_trained:
            return 50  # Default score
            
        X = self.scaler.transform([movie_features])
        return self.model.predict(X)[0]


class FakeEngagementDetector:
    """Detect fake engagement and bot activity"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def extract_engagement_features(self, engagement_data):
        """Extract features from engagement data"""
        features = []
        
        for engagement in engagement_data:
            feature_vector = [
                engagement.get('view_count', 0),
                engagement.get('like_count', 0),
                engagement.get('comment_count', 0),
                engagement.get('share_count', 0),
                engagement.get('view_duration', 0),
                engagement.get('unique_viewers', 0),
                engagement.get('repeat_viewers', 0),
                engagement.get('engagement_rate', 0),
                engagement.get('velocity', 0),  # Growth rate
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def detect_anomalies(self, engagement_metrics):
        """Detect suspicious engagement patterns"""
        # Simple rule-based detection
        flags = []
        
        # High views but low engagement
        if engagement_metrics.get('view_count', 0) > 1000:
            engagement_rate = (
                engagement_metrics.get('like_count', 0) + 
                engagement_metrics.get('comment_count', 0)
            ) / engagement_metrics.get('view_count', 1)
            
            if engagement_rate < 0.01:
                flags.append('Low engagement rate')
        
        # Suspicious view duration
        if engagement_metrics.get('avg_view_duration', 0) < 10:
            flags.append('Very short view duration')
            
        # Unnatural growth pattern
        if engagement_metrics.get('velocity', 0) > 1000:
            flags.append('Unnatural growth velocity')
            
        return {
            'is_suspicious': len(flags) > 0,
            'confidence': min(len(flags) * 33, 100),
            'flags': flags
        }


class SentimentTrendAnalyzer:
    """Analyze sentiment trends over time"""
    
    def __init__(self):
        self.sentiment_history = []
        
    def add_sentiment(self, movie_id, sentiment, score, timestamp=None):
        """Add sentiment data point"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.sentiment_history.append({
            'movie_id': movie_id,
            'sentiment': sentiment,
            'score': score,
            'timestamp': timestamp
        })
    
    def get_sentiment_trend(self, movie_id, days=30):
        """Get sentiment trend for a movie"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        movie_sentiments = [
            s for s in self.sentiment_history 
            if s['movie_id'] == movie_id and s['timestamp'] >= cutoff_date
        ]
        
        if not movie_sentiments:
            return None
            
        df = pd.DataFrame(movie_sentiments)
        
        positive_count = len(df[df['sentiment'] == 'Good'])
        total_count = len(df)
        
        return {
            'positive_ratio': positive_count / total_count if total_count > 0 else 0,
            'total_reviews': total_count,
            'avg_score': df['score'].mean() if 'score' in df.columns else 0,
            'trend': 'improving' if positive_count / total_count > 0.6 else 'declining'
        }
    
    def get_trending_movies(self, min_reviews=5):
        """Get movies with positive sentiment trends"""
        if not self.sentiment_history:
            return []
            
        df = pd.DataFrame(self.sentiment_history)
        recent_date = datetime.now() - timedelta(days=7)
        recent_df = df[df['timestamp'] >= recent_date]
        
        movie_stats = recent_df.groupby('movie_id').agg({
            'sentiment': lambda x: (x == 'Good').sum() / len(x),
            'movie_id': 'count'
        }).rename(columns={'movie_id': 'count', 'sentiment': 'positive_ratio'})
        
        trending = movie_stats[movie_stats['count'] >= min_reviews].sort_values(
            'positive_ratio', ascending=False
        )
        
        return trending.head(10).to_dict()


class ContentAnalytics:
    """Main analytics controller"""
    
    def __init__(self):
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.popularity_predictor = PopularityPredictor()
        self.fake_detector = FakeEngagementDetector()
        self.sentiment_analyzer = SentimentTrendAnalyzer()
        
    def get_dashboard_metrics(self):
        """Get comprehensive dashboard metrics"""
        consumption_patterns = self.behavior_analyzer.get_consumption_patterns()
        
        return {
            'user_metrics': consumption_patterns,
            'content_health': {
                'total_content': consumption_patterns.get('total_sessions', 0),
                'engagement_rate': 0.75,  # Placeholder
                'avg_sentiment': 0.65  # Placeholder
            },
            'trending_content': self.sentiment_analyzer.get_trending_movies()
        }
    
    def analyze_movie_performance(self, movie_id, engagement_data):
        """Comprehensive movie performance analysis"""
        # Check for fake engagement
        fake_check = self.fake_detector.detect_anomalies(engagement_data)
        
        # Get sentiment trend
        sentiment_trend = self.sentiment_analyzer.get_sentiment_trend(movie_id)
        
        return {
            'movie_id': movie_id,
            'engagement_quality': fake_check,
            'sentiment_trend': sentiment_trend,
            'overall_health': 'good' if not fake_check['is_suspicious'] else 'suspicious'
        }


# Global analytics instance
analytics_engine = ContentAnalytics()


def get_analytics_engine():
    """Get the global analytics engine instance"""
    return analytics_engine
