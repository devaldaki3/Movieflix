"""
ML-based Popularity Prediction Module
Predicts movie popularity using machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class PopularityPredictor:
    """Predicts movie popularity using ML models"""
    
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare features for ML model"""
        features = pd.DataFrame()
        
        # Numeric features
        features['budget'] = df['budget'].fillna(0)
        features['runtime'] = df['runtime'].fillna(df['runtime'].mean())
        features['vote_average'] = df['vote_average'].fillna(0)
        features['vote_count'] = df['vote_count'].fillna(0)
        
        # Extract year from release_date
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        features['release_year'] = df['release_year'].fillna(2000)
        features['years_since_release'] = 2024 - features['release_year']
        
        # Genre features (one-hot encoding for top genres)
        top_genres = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Adventure', 'Crime', 'Science', 'Horror', 'Fantasy']
        for genre in top_genres:
            features[f'genre_{genre}'] = df['genres'].fillna('').str.contains(genre).astype(int)
        
        # Revenue feature (if available)
        if 'revenue' in df.columns:
            features['revenue'] = df['revenue'].fillna(0)
            features['has_revenue'] = (features['revenue'] > 0).astype(int)
        
        # Derived features
        features['budget_per_minute'] = features['budget'] / (features['runtime'] + 1)
        features['rating_votes_ratio'] = features['vote_average'] * np.log1p(features['vote_count'])
        
        return features
    
    def train_models(self, df):
        """Train both regression and classification models"""
        print("🔧 Preparing features...")
        X = self.prepare_features(df)
        self.feature_names = X.columns.tolist()
        
        # Target variable
        y_regression = df['popularity'].fillna(0)
        
        # Create popularity categories for classification
        # Low: 0-10, Medium: 10-50, High: 50+
        y_classification = pd.cut(y_regression, 
                                  bins=[0, 10, 50, float('inf')], 
                                  labels=['Low', 'Medium', 'High'])
        
        # Remove NaN values from classification
        valid_idx = ~y_classification.isna()
        X_valid = X[valid_idx]
        y_reg_valid = y_regression[valid_idx]
        y_clf_valid = y_classification[valid_idx]
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X_valid, y_reg_valid, y_clf_valid, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("🤖 Training Regression Model (Popularity Score Prediction)...")
        # Regression model for exact popularity score
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.regression_model.fit(X_train_scaled, y_reg_train)
        
        # Evaluate regression
        y_reg_pred = self.regression_model.predict(X_test_scaled)
        reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
        reg_r2 = r2_score(y_reg_test, y_reg_pred)
        
        print(f"   ✅ Regression MSE: {reg_mse:.2f}")
        print(f"   ✅ Regression R²: {reg_r2:.3f}")
        
        print("\n🎯 Training Classification Model (Popularity Category)...")
        # Classification model for popularity category
        self.classification_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.classification_model.fit(X_train_scaled, y_clf_train)
        
        # Evaluate classification
        y_clf_pred = self.classification_model.predict(X_test_scaled)
        clf_accuracy = accuracy_score(y_clf_test, y_clf_pred)
        
        print(f"   ✅ Classification Accuracy: {clf_accuracy:.3f}")
        
        self.is_trained = True
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.regression_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'regression_mse': float(reg_mse),
            'regression_r2': float(reg_r2),
            'classification_accuracy': float(clf_accuracy),
            'feature_importance': feature_importance.head(10).to_dict('records'),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def predict_popularity(self, movie_data):
        """Predict popularity for new movie data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        features = self.prepare_features(movie_data)
        features_scaled = self.scaler.transform(features)
        
        # Regression prediction
        popularity_score = self.regression_model.predict(features_scaled)
        
        # Classification prediction
        popularity_category = self.classification_model.predict(features_scaled)
        popularity_proba = self.classification_model.predict_proba(features_scaled)
        
        return {
            'popularity_score': float(popularity_score[0]),
            'popularity_category': str(popularity_category[0]),
            'category_probabilities': {
                'Low': float(popularity_proba[0][0]),
                'Medium': float(popularity_proba[0][1]),
                'High': float(popularity_proba[0][2])
            }
        }


def train_and_evaluate_models():
    """Train models and return evaluation metrics"""
    try:
        print("📊 Loading dataset...")
        df = pd.read_csv('./Artifacts/movies.csv')
        print(f"   Loaded {len(df):,} movies")
        
        # Initialize predictor
        predictor = PopularityPredictor()
        
        # Train models
        metrics = predictor.train_models(df)
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETE!")
        print("="*60)
        
        # Make sample predictions
        print("\n🎬 Sample Predictions:")
        sample_movies = df.sample(5)
        
        for idx, row in sample_movies.iterrows():
            movie_df = pd.DataFrame([row])
            prediction = predictor.predict_popularity(movie_df)
            
            print(f"\n   Movie: {row['title']}")
            print(f"   Actual Popularity: {row['popularity']:.2f}")
            print(f"   Predicted Score: {prediction['popularity_score']:.2f}")
            print(f"   Predicted Category: {prediction['popularity_category']}")
        
        # Calculate overall statistics
        print("\n📈 Generating Predictions for All Movies...")
        all_features = predictor.prepare_features(df)
        all_features_scaled = predictor.scaler.transform(all_features)
        
        all_predictions = predictor.regression_model.predict(all_features_scaled)
        all_categories = predictor.classification_model.predict(all_features_scaled)
        
        # Category distribution
        category_counts = pd.Series(all_categories).value_counts()
        
        results = {
            'model_metrics': metrics,
            'predictions_summary': {
                'total_predictions': len(all_predictions),
                'avg_predicted_popularity': float(np.mean(all_predictions)),
                'median_predicted_popularity': float(np.median(all_predictions)),
                'category_distribution': {
                    'Low': int(category_counts.get('Low', 0)),
                    'Medium': int(category_counts.get('Medium', 0)),
                    'High': int(category_counts.get('High', 0))
                }
            },
            'top_predicted_movies': [],
            'feature_importance': metrics['feature_importance']
        }
        
        # Get top predicted popular movies
        df['predicted_popularity'] = all_predictions
        top_predicted = df.nlargest(10, 'predicted_popularity')[['title', 'popularity', 'predicted_popularity', 'vote_average']]
        
        results['top_predicted_movies'] = [
            {
                'title': row['title'],
                'actual_popularity': float(row['popularity']),
                'predicted_popularity': float(row['predicted_popularity']),
                'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0.0
            }
            for _, row in top_predicted.iterrows()
        ]
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("🚀 ML Popularity Prediction System")
    print("="*60)
    
    results = train_and_evaluate_models()
    
    if results:
        print("\n" + "="*60)
        print("📊 FINAL RESULTS")
        print("="*60)
        print(f"\nModel Performance:")
        print(f"   R² Score: {results['model_metrics']['regression_r2']:.3f}")
        print(f"   Classification Accuracy: {results['model_metrics']['classification_accuracy']:.3f}")
        
        print(f"\nPredictions Summary:")
        print(f"   Average Predicted Popularity: {results['predictions_summary']['avg_predicted_popularity']:.2f}")
        print(f"   Category Distribution:")
        for cat, count in results['predictions_summary']['category_distribution'].items():
            print(f"      {cat}: {count:,} movies")
        
        print(f"\nTop 5 Most Important Features:")
        for i, feat in enumerate(results['feature_importance'][:5], 1):
            print(f"   {i}. {feat['feature']}: {feat['importance']:.4f}")
