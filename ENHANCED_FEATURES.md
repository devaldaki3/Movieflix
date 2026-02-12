# 🎬 Enhanced Entertainment & Media Analytics Platform

## 📋 Project Overview

This is an **End-to-End Movie Recommendation System** enhanced with advanced **Entertainment & Media Analytics** capabilities. The platform addresses key challenges in the entertainment industry including content overload, audience preference prediction, revenue optimization, fake engagement detection, and sentiment volatility.

---

## 🎯 Objectives Achieved

### ✅ Core Features (Original)
- **Movie Recommendation System**: Content-based filtering using cosine similarity
- **Sentiment Analysis**: NLP-based review classification (Good/Bad)
- **Web Scraping**: Real-time IMDB review extraction
- **Movie Metadata**: Cast, ratings, runtime, genres display

### ✅ Enhanced Features (New)
1. **User Behavior Analysis**
   - Real-time activity tracking
   - Session analytics
   - Consumption pattern analysis
   - User preference profiling

2. **Content Popularity Prediction**
   - ML-based popularity forecasting
   - Gradient Boosting Regressor model
   - Feature engineering from movie metadata

3. **Fake Engagement Detection**
   - Anomaly detection algorithms
   - Bot activity identification
   - Suspicious pattern flagging
   - Engagement quality scoring

4. **Sentiment Trend Analysis**
   - Time-series sentiment tracking
   - Trending content identification
   - Sentiment volatility monitoring
   - Review trend visualization

5. **Analytics Dashboard**
   - Real-time metrics display
   - User behavior insights
   - Content performance tracking
   - Interactive visualizations

---

## 🏗️ Architecture

### Project Structure
```
End-to-End-Movie-Recommendation-System-main/
├── app.py                          # Original Flask application
├── app_enhanced.py                 # Enhanced app with analytics
├── analytics_engine.py             # Analytics & ML modules
├── requirements.txt                # Python dependencies
├── Artifacts/                      # Models & datasets
│   ├── nlp_model.pkl              # Sentiment analysis model
│   ├── tranform.pkl               # Text vectorizer
│   ├── main_data.csv              # Movie dataset
│   └── movies.csv                 # Extended movie data
├── templates/
│   ├── home.html                  # Landing page
│   ├── recommend.html             # Recommendation results
│   └── analytics.html             # Analytics dashboard
├── static/                        # CSS, JS, images
└── NoteBook_Experiments/          # Jupyter notebooks
```

### Technology Stack
```
Backend:        Flask, Python 3.x
ML/AI:          scikit-learn, numpy, pandas
NLP:            nltk, CountVectorizer
Web Scraping:   BeautifulSoup4, urllib3
Analytics:      Custom analytics engine
Visualization:  Chart.js (frontend)
Storage:        Pickle, CSV
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone/Download Project
```bash
cd c:\Users\dakid\Downloads\Projects\End-to-End-Movie-Recommendation-System-main
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Run the Application

**Option A: Original Version**
```powershell
python app.py
```

**Option B: Enhanced Version with Analytics**
```powershell
python app_enhanced.py
```

### Step 6: Access the Application
- **Main App**: http://localhost:5000
- **Analytics Dashboard**: http://localhost:5000/analytics

---

## 📊 Analytics Engine Components

### 1. UserBehaviorAnalyzer
**Purpose**: Track and analyze user interactions

**Features**:
- Activity tracking (views, searches, clicks)
- User preference profiling
- Session analytics
- Consumption pattern analysis

**Methods**:
```python
track_user_activity(user_id, movie_id, action, timestamp)
get_user_preferences(user_id)
get_consumption_patterns()
```

### 2. PopularityPredictor
**Purpose**: Predict content popularity using ML

**Features**:
- Gradient Boosting Regressor model
- Feature extraction from movie metadata
- Popularity score prediction

**Methods**:
```python
prepare_features(movie_data)
train(movie_data, popularity_scores)
predict_popularity(movie_features)
```

### 3. FakeEngagementDetector
**Purpose**: Detect fake engagement and bot activity

**Features**:
- Anomaly detection
- Suspicious pattern identification
- Engagement quality scoring
- Rule-based and ML-based detection

**Methods**:
```python
extract_engagement_features(engagement_data)
detect_anomalies(engagement_metrics)
```

**Detection Rules**:
- Low engagement rate (views vs. likes/comments)
- Suspicious view duration
- Unnatural growth velocity
- Bot-like behavior patterns

### 4. SentimentTrendAnalyzer
**Purpose**: Analyze sentiment trends over time

**Features**:
- Time-series sentiment tracking
- Trend identification
- Trending content discovery
- Sentiment volatility monitoring

**Methods**:
```python
add_sentiment(movie_id, sentiment, score, timestamp)
get_sentiment_trend(movie_id, days)
get_trending_movies(min_reviews)
```

---

## 🔌 API Endpoints

### Original Endpoints
```
GET  /                          # Home page
GET  /home                      # Home page (alias)
POST /similarity                # Get similar movies
POST /recommend                 # Get movie recommendations
```

### New Analytics Endpoints
```
GET  /analytics                           # Analytics dashboard
GET  /api/analytics/dashboard             # Dashboard metrics (JSON)
GET  /api/analytics/user-behavior         # User behavior data (JSON)
GET  /api/analytics/trending              # Trending movies (JSON)
GET  /api/analytics/movie/<movie_id>      # Movie-specific analytics (JSON)
POST /api/analytics/fake-detection        # Fake engagement detection (JSON)
```

---

## 📈 Use Cases

### 1. Content Discovery
- Users search for movies
- System recommends similar content
- Tracks user preferences for personalization

### 2. Sentiment Analysis
- Scrapes IMDB reviews in real-time
- Classifies sentiment (Good/Bad)
- Tracks sentiment trends over time

### 3. Business Intelligence
- Monitor user engagement metrics
- Identify trending content
- Detect fake engagement
- Optimize content strategy

### 4. Fraud Detection
- Identify suspicious engagement patterns
- Flag bot activity
- Ensure authentic metrics

### 5. Trend Analysis
- Track sentiment volatility
- Identify emerging trends
- Predict content popularity

---

## 🎨 Dashboard Features

### Metrics Display
- **Total Sessions**: Number of user interactions
- **Unique Users**: Active viewer count
- **Engagement Rate**: Overall platform engagement
- **Average Sentiment**: Positive review ratio

### Visualizations
- Action distribution charts
- Progress bars for metrics
- Top movies table
- Trend indicators

### Real-time Updates
- Auto-refresh every 30 seconds
- Manual refresh option
- Live data streaming

---

## 🧪 Machine Learning Models

### 1. Sentiment Analysis Model
- **Type**: Classification (NLP)
- **Algorithm**: Pre-trained (stored in nlp_model.pkl)
- **Input**: Movie review text
- **Output**: Good/Bad sentiment

### 2. Popularity Prediction Model
- **Type**: Regression
- **Algorithm**: Gradient Boosting Regressor
- **Features**: vote_count, vote_average, genres, cast, runtime, release_year
- **Output**: Popularity score (0-100)

### 3. Fake Engagement Detection
- **Type**: Anomaly Detection
- **Algorithm**: Random Forest Classifier + Rule-based
- **Features**: view_count, like_count, comment_count, engagement_rate, velocity
- **Output**: Suspicious/Legitimate + confidence score

### 4. Content Similarity
- **Type**: Content-based Filtering
- **Algorithm**: Cosine Similarity
- **Features**: TF-IDF vectors from movie metadata
- **Output**: Top 10 similar movies

---

## 📊 Sample Analytics Output

### User Behavior Metrics
```json
{
  "total_sessions": 150,
  "unique_users": 45,
  "most_viewed_movies": {
    "Inception": 25,
    "The Dark Knight": 20,
    "Interstellar": 18
  },
  "action_distribution": {
    "view": 80,
    "search": 50,
    "home_visit": 20
  }
}
```

### Fake Engagement Detection
```json
{
  "is_suspicious": true,
  "confidence": 66,
  "flags": [
    "Low engagement rate",
    "Unnatural growth velocity"
  ]
}
```

### Sentiment Trend
```json
{
  "positive_ratio": 0.75,
  "total_reviews": 120,
  "avg_score": 0.72,
  "trend": "improving"
}
```

---

## 🔧 Configuration

### Environment Variables
```python
# Flask Configuration
DEBUG = True
SECRET_KEY = 'entertainment_analytics_secret_key_2024'
HOST = '0.0.0.0'
PORT = 5000

# Analytics Configuration
ANALYTICS_ENABLED = True
AUTO_REFRESH_INTERVAL = 30000  # milliseconds
```

### Model Paths
```python
NLP_MODEL_PATH = "./Artifacts/nlp_model.pkl"
VECTORIZER_PATH = "./Artifacts/tranform.pkl"
MOVIE_DATA_PATH = "./Artifacts/main_data.csv"
```

---

## 🎯 Entertainment Industry Challenges Addressed

### ✅ Content Overload
**Solution**: Intelligent recommendation system with content-based filtering

### ✅ Audience Preference Prediction
**Solution**: User behavior tracking and ML-based popularity prediction

### ✅ Revenue Optimization
**Solution**: Analytics dashboard with engagement metrics and trend analysis

### ✅ Fake Engagement
**Solution**: Anomaly detection and fake engagement detector

### ✅ Sentiment Volatility
**Solution**: Real-time sentiment tracking and trend analysis

---

## 📚 Learning Objectives Covered

### ✅ EDA (Exploratory Data Analysis)
- User behavior pattern analysis
- Content consumption analysis
- Statistical insights

### ✅ Recommendation Systems
- Content-based filtering
- Cosine similarity
- Feature engineering

### ✅ Sentiment Analysis
- NLP model implementation
- Review classification
- Trend analysis

### ✅ ML Model Deployment
- Popularity prediction
- Fake engagement detection
- Model persistence (pickle)

### ✅ Deep Learning Concepts
- Feature extraction
- Classification models
- Anomaly detection

---

## 🚀 Future Enhancements

### Planned Features
1. **Collaborative Filtering**: User-based recommendations
2. **Deep Learning**: Neural networks for content classification
3. **Real-time Streaming**: WebSocket integration
4. **A/B Testing**: Experiment framework
5. **User Churn Prediction**: ML model for retention
6. **Content Thumbnail Analysis**: Computer vision
7. **Caption Generation**: NLP for metadata
8. **Multi-language Support**: Internationalization
9. **Mobile App**: React Native/Flutter
10. **Cloud Deployment**: AWS/Azure/GCP

---

## 📖 Documentation

### Code Documentation
- All functions have docstrings
- Type hints for better code clarity
- Inline comments for complex logic

### API Documentation
- RESTful API design
- JSON response format
- Error handling

---

## 🤝 Contributing

This project is designed for educational purposes and demonstrates:
- Full-stack web development
- Machine learning integration
- Analytics implementation
- Real-world problem solving

---

## 📄 License

Distributed under the GNU General Public License v3.0

---

## 🎓 Educational Value

This project demonstrates:
1. **End-to-end ML pipeline**: Data → Model → Deployment
2. **Real-world problem solving**: Entertainment industry challenges
3. **Full-stack development**: Backend + Frontend + ML
4. **Analytics implementation**: Tracking, analysis, visualization
5. **Production-ready code**: Error handling, logging, scalability

---

## 📞 Support

For issues or questions:
1. Check the documentation
2. Review the code comments
3. Test with sample data
4. Verify dependencies

---

## 🎉 Quick Start Summary

```powershell
# 1. Navigate to project
cd c:\Users\dakid\Downloads\Projects\End-to-End-Movie-Recommendation-System-main

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Run enhanced version
python app_enhanced.py

# 4. Access application
# Main App: http://localhost:5000
# Analytics: http://localhost:5000/analytics
```

---

**🎬 Enjoy your Enhanced Entertainment Analytics Platform!**
