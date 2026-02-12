# Entertainment Analytics Demo
# This notebook demonstrates the enhanced analytics features

## Setup and Imports

```python
import sys
sys.path.append('..')

from analytics_engine import (
    UserBehaviorAnalyzer,
    PopularityPredictor,
    FakeEngagementDetector,
    SentimentTrendAnalyzer,
    ContentAnalytics
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

## 1. User Behavior Analysis

### Track User Activities

```python
# Initialize analyzer
behavior_analyzer = UserBehaviorAnalyzer()

# Simulate user activities
users = ['user_001', 'user_002', 'user_003', 'user_004', 'user_005']
movies = ['Inception', 'The Dark Knight', 'Interstellar', 'The Matrix', 'Avatar']
actions = ['view', 'search', 'recommend_click', 'review']

# Generate sample data
for i in range(100):
    user = np.random.choice(users)
    movie = np.random.choice(movies)
    action = np.random.choice(actions, p=[0.4, 0.3, 0.2, 0.1])
    timestamp = datetime.now() - timedelta(hours=np.random.randint(0, 168))
    
    behavior_analyzer.track_user_activity(user, movie, action, timestamp)

print("✅ Tracked 100 user activities")
```

### Analyze Consumption Patterns

```python
patterns = behavior_analyzer.get_consumption_patterns()

print("\n📊 Consumption Patterns:")
print(f"Total Sessions: {patterns['total_sessions']}")
print(f"Unique Users: {patterns['unique_users']}")
print(f"\nAction Distribution:")
for action, count in patterns['action_distribution'].items():
    print(f"  {action}: {count}")
print(f"\nMost Viewed Movies:")
for movie, views in list(patterns['most_viewed_movies'].items())[:5]:
    print(f"  {movie}: {views} views")
```

### Visualize User Behavior

```python
# Action distribution pie chart
action_dist = patterns['action_distribution']
plt.figure(figsize=(10, 6))
plt.pie(action_dist.values(), labels=action_dist.keys(), autopct='%1.1f%%', startangle=90)
plt.title('User Action Distribution', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.show()

# Most viewed movies bar chart
most_viewed = patterns['most_viewed_movies']
plt.figure(figsize=(12, 6))
plt.bar(most_viewed.keys(), most_viewed.values(), color='skyblue')
plt.xlabel('Movie Title', fontsize=12)
plt.ylabel('View Count', fontsize=12)
plt.title('Most Viewed Movies', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## 2. Popularity Prediction

### Prepare Movie Data

```python
# Load movie dataset
movie_data = pd.read_csv('../Artifacts/main_data.csv')

# Display sample
print("📽️ Sample Movie Data:")
print(movie_data.head())
print(f"\nTotal Movies: {len(movie_data)}")
```

### Train Popularity Predictor

```python
# Initialize predictor
popularity_predictor = PopularityPredictor()

# Prepare sample data for training
sample_movies = movie_data.sample(min(1000, len(movie_data)))

# Create synthetic popularity scores (in real scenario, use actual metrics)
popularity_scores = np.random.randint(20, 100, size=len(sample_movies))

# Extract features
features = popularity_predictor.prepare_features(sample_movies)

print(f"✅ Prepared {len(features)} feature vectors")
print(f"Feature dimensions: {features.shape}")

# Train model
popularity_predictor.train(sample_movies, popularity_scores)
print("✅ Popularity prediction model trained!")
```

### Predict Popularity

```python
# Predict popularity for sample movies
test_movies = movie_data.sample(5)

print("\n🎯 Popularity Predictions:")
for idx, movie in test_movies.iterrows():
    features = [
        movie.get('vote_count', 0),
        movie.get('vote_average', 0),
        3,  # genre count
        5,  # cast count
        120,  # runtime
        1  # recent release
    ]
    
    popularity = popularity_predictor.predict_popularity(features)
    print(f"\n{movie['movie_title']}:")
    print(f"  Predicted Popularity: {popularity:.1f}/100")
    print(f"  Vote Average: {movie.get('vote_average', 'N/A')}")
```

## 3. Fake Engagement Detection

### Analyze Engagement Metrics

```python
# Initialize detector
fake_detector = FakeEngagementDetector()

# Sample engagement data
engagement_scenarios = [
    {
        'name': 'Legitimate High Engagement',
        'metrics': {
            'view_count': 10000,
            'like_count': 800,
            'comment_count': 250,
            'share_count': 150,
            'view_duration': 120,
            'unique_viewers': 9500,
            'repeat_viewers': 500,
            'engagement_rate': 0.12,
            'velocity': 100,
            'avg_view_duration': 120
        }
    },
    {
        'name': 'Suspicious Low Engagement',
        'metrics': {
            'view_count': 50000,
            'like_count': 50,
            'comment_count': 10,
            'share_count': 5,
            'view_duration': 5,
            'unique_viewers': 45000,
            'repeat_viewers': 5000,
            'engagement_rate': 0.001,
            'velocity': 5000,
            'avg_view_duration': 5
        }
    },
    {
        'name': 'Bot-like Activity',
        'metrics': {
            'view_count': 100000,
            'like_count': 100,
            'comment_count': 5,
            'share_count': 2,
            'view_duration': 2,
            'unique_viewers': 95000,
            'repeat_viewers': 5000,
            'engagement_rate': 0.001,
            'velocity': 10000,
            'avg_view_duration': 2
        }
    }
]

print("🔍 Fake Engagement Detection Results:\n")
for scenario in engagement_scenarios:
    result = fake_detector.detect_anomalies(scenario['metrics'])
    
    print(f"Scenario: {scenario['name']}")
    print(f"  Suspicious: {'⚠️ YES' if result['is_suspicious'] else '✅ NO'}")
    print(f"  Confidence: {result['confidence']}%")
    if result['flags']:
        print(f"  Flags: {', '.join(result['flags'])}")
    print()
```

### Visualize Engagement Quality

```python
# Create comparison visualization
scenarios_names = [s['name'] for s in engagement_scenarios]
engagement_rates = [s['metrics']['engagement_rate'] * 100 for s in engagement_scenarios]
suspicion_scores = [fake_detector.detect_anomalies(s['metrics'])['confidence'] 
                   for s in engagement_scenarios]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Engagement rates
ax1.bar(range(len(scenarios_names)), engagement_rates, color=['green', 'orange', 'red'])
ax1.set_xlabel('Scenario', fontsize=12)
ax1.set_ylabel('Engagement Rate (%)', fontsize=12)
ax1.set_title('Engagement Rate Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(scenarios_names)))
ax1.set_xticklabels(scenarios_names, rotation=15, ha='right')

# Suspicion scores
ax2.bar(range(len(scenarios_names)), suspicion_scores, color=['green', 'orange', 'red'])
ax2.set_xlabel('Scenario', fontsize=12)
ax2.set_ylabel('Suspicion Score (%)', fontsize=12)
ax2.set_title('Fake Engagement Detection Score', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(scenarios_names)))
ax2.set_xticklabels(scenarios_names, rotation=15, ha='right')
ax2.axhline(y=50, color='r', linestyle='--', label='Threshold')
ax2.legend()

plt.tight_layout()
plt.show()
```

## 4. Sentiment Trend Analysis

### Track Sentiment Over Time

```python
# Initialize analyzer
sentiment_analyzer = SentimentTrendAnalyzer()

# Simulate sentiment data over 30 days
movies_for_sentiment = ['Inception', 'The Dark Knight', 'Interstellar']

for movie in movies_for_sentiment:
    for day in range(30):
        # Simulate varying sentiment
        num_reviews = np.random.randint(5, 20)
        for _ in range(num_reviews):
            sentiment = np.random.choice(['Good', 'Bad'], p=[0.7, 0.3])
            score = np.random.random()
            timestamp = datetime.now() - timedelta(days=30-day)
            
            sentiment_analyzer.add_sentiment(movie, sentiment, score, timestamp)

print("✅ Generated sentiment data for 30 days")
```

### Analyze Sentiment Trends

```python
print("\n📈 Sentiment Trend Analysis:\n")

for movie in movies_for_sentiment:
    trend = sentiment_analyzer.get_sentiment_trend(movie, days=30)
    
    if trend:
        print(f"{movie}:")
        print(f"  Positive Ratio: {trend['positive_ratio']*100:.1f}%")
        print(f"  Total Reviews: {trend['total_reviews']}")
        print(f"  Average Score: {trend['avg_score']:.2f}")
        print(f"  Trend: {trend['trend'].upper()}")
        print()
```

### Visualize Sentiment Trends

```python
# Create sentiment trend visualization
trends_data = []
for movie in movies_for_sentiment:
    trend = sentiment_analyzer.get_sentiment_trend(movie, days=30)
    if trend:
        trends_data.append({
            'Movie': movie,
            'Positive %': trend['positive_ratio'] * 100,
            'Reviews': trend['total_reviews']
        })

df_trends = pd.DataFrame(trends_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Positive sentiment percentage
ax1.barh(df_trends['Movie'], df_trends['Positive %'], color='skyblue')
ax1.set_xlabel('Positive Sentiment (%)', fontsize=12)
ax1.set_title('Positive Sentiment by Movie', fontsize=14, fontweight='bold')
ax1.axvline(x=50, color='r', linestyle='--', label='50% Threshold')
ax1.legend()

# Total reviews
ax2.barh(df_trends['Movie'], df_trends['Reviews'], color='lightcoral')
ax2.set_xlabel('Total Reviews', fontsize=12)
ax2.set_title('Review Volume by Movie', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

### Get Trending Movies

```python
trending = sentiment_analyzer.get_trending_movies(min_reviews=10)

print("\n🔥 Trending Movies (Last 7 Days):")
if trending:
    for movie_id, stats in list(trending.items())[:5]:
        print(f"  {movie_id}")
else:
    print("  No trending movies found (need more data)")
```

## 5. Comprehensive Analytics Dashboard

### Initialize Complete Analytics Engine

```python
# Create complete analytics engine
analytics = ContentAnalytics()

# Add sample data
for i in range(50):
    user = np.random.choice(users)
    movie = np.random.choice(movies)
    action = np.random.choice(actions)
    analytics.behavior_analyzer.track_user_activity(user, movie, action)

print("✅ Analytics engine initialized with sample data")
```

### Get Dashboard Metrics

```python
dashboard_metrics = analytics.get_dashboard_metrics()

print("\n📊 Dashboard Metrics:")
print(f"\nUser Metrics:")
print(f"  Total Sessions: {dashboard_metrics['user_metrics']['total_sessions']}")
print(f"  Unique Users: {dashboard_metrics['user_metrics']['unique_users']}")

print(f"\nContent Health:")
print(f"  Total Content: {dashboard_metrics['content_health']['total_content']}")
print(f"  Engagement Rate: {dashboard_metrics['content_health']['engagement_rate']*100:.1f}%")
print(f"  Avg Sentiment: {dashboard_metrics['content_health']['avg_sentiment']*100:.1f}%")
```

### Analyze Movie Performance

```python
# Analyze specific movie
movie_id = 'Inception'
engagement_data = {
    'view_count': 15000,
    'like_count': 1200,
    'comment_count': 450,
    'share_count': 300,
    'view_duration': 95,
    'unique_viewers': 14000,
    'repeat_viewers': 1000,
    'engagement_rate': 0.13,
    'velocity': 200,
    'avg_view_duration': 95
}

performance = analytics.analyze_movie_performance(movie_id, engagement_data)

print(f"\n🎬 Movie Performance Analysis: {movie_id}")
print(f"\nEngagement Quality:")
print(f"  Suspicious: {performance['engagement_quality']['is_suspicious']}")
print(f"  Confidence: {performance['engagement_quality']['confidence']}%")
if performance['engagement_quality']['flags']:
    print(f"  Flags: {', '.join(performance['engagement_quality']['flags'])}")

print(f"\nOverall Health: {performance['overall_health'].upper()}")
```

## 6. Key Insights & Recommendations

### Summary Statistics

```python
print("\n📈 Analytics Summary:\n")
print("=" * 60)
print("USER BEHAVIOR INSIGHTS")
print("=" * 60)

patterns = analytics.behavior_analyzer.get_consumption_patterns()
print(f"Total User Sessions: {patterns['total_sessions']}")
print(f"Unique Active Users: {patterns['unique_users']}")
print(f"Average Sessions per User: {patterns['total_sessions'] / max(patterns['unique_users'], 1):.1f}")

print("\n" + "=" * 60)
print("CONTENT PERFORMANCE")
print("=" * 60)

if patterns.get('most_viewed_movies'):
    top_movie = list(patterns['most_viewed_movies'].items())[0]
    print(f"Most Popular Movie: {top_movie[0]} ({top_movie[1]} views)")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("✓ Focus on content similar to top-performing movies")
print("✓ Monitor engagement quality for suspicious patterns")
print("✓ Track sentiment trends to identify declining content")
print("✓ Optimize user experience based on behavior patterns")
print("✓ Implement A/B testing for recommendation algorithms")
```

## Conclusion

This notebook demonstrated:
1. ✅ User behavior tracking and analysis
2. ✅ Content popularity prediction using ML
3. ✅ Fake engagement detection
4. ✅ Sentiment trend analysis
5. ✅ Comprehensive analytics dashboard

These analytics capabilities help address key entertainment industry challenges:
- Content overload → Smart recommendations
- Audience preferences → Behavior analysis
- Revenue optimization → Engagement metrics
- Fake engagement → Anomaly detection
- Sentiment volatility → Trend tracking
