# 🚀 Quick Start Guide - Enhanced Movie Recommendation System

## 📋 Table of Contents
1. [Running the Application](#running-the-application)
2. [Accessing Features](#accessing-features)
3. [API Usage](#api-usage)
4. [Analytics Features](#analytics-features)
5. [Troubleshooting](#troubleshooting)

---

## 🏃 Running the Application

### Option 1: Original Version (Basic Recommendations)
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run original app
python app.py

# Access at: http://localhost:5000
```

### Option 2: Enhanced Version (With Analytics)
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run enhanced app
python app_enhanced.py

# Main App: http://localhost:5000
# Analytics: http://localhost:5000/analytics
```

---

## 🎯 Accessing Features

### 1. Movie Recommendations
**URL**: `http://localhost:5000`

**How to use**:
1. Type movie name in search box
2. Select from auto-complete suggestions
3. Click "Recommend" button
4. View 10 similar movies + reviews

### 2. Analytics Dashboard
**URL**: `http://localhost:5000/analytics`

**Features**:
- Real-time user metrics
- Behavior patterns
- Top movies
- Engagement stats
- Auto-refresh every 30s

### 3. Movie Details
**How to access**:
1. Search for a movie
2. Click on recommended movie
3. View cast, reviews, ratings
4. See sentiment analysis results

---

## 🔌 API Usage

### Get Dashboard Metrics
```javascript
fetch('/api/analytics/dashboard')
  .then(response => response.json())
  .then(data => console.log(data));
```

**Response**:
```json
{
  "user_metrics": {
    "total_sessions": 150,
    "unique_users": 45
  },
  "content_health": {
    "engagement_rate": 0.75,
    "avg_sentiment": 0.65
  }
}
```

### Get User Behavior
```javascript
fetch('/api/analytics/user-behavior')
  .then(response => response.json())
  .then(data => console.log(data));
```

**Response**:
```json
{
  "total_sessions": 150,
  "unique_users": 45,
  "action_distribution": {
    "view": 80,
    "search": 50,
    "home_visit": 20
  },
  "most_viewed_movies": {
    "Inception": 25,
    "The Dark Knight": 20
  }
}
```

### Detect Fake Engagement
```javascript
fetch('/api/analytics/fake-detection', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    view_count: 10000,
    like_count: 50,
    comment_count: 10,
    engagement_rate: 0.006,
    velocity: 5000,
    avg_view_duration: 5
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

**Response**:
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

### Get Movie Analytics
```javascript
fetch('/api/analytics/movie/Inception')
  .then(response => response.json())
  .then(data => console.log(data));
```

---

## 📊 Analytics Features

### 1. User Behavior Tracking
**Automatically tracks**:
- Page visits
- Movie searches
- Movie views
- Recommendation clicks

**Access data**:
```python
from analytics_engine import get_analytics_engine
analytics = get_analytics_engine()

# Get user preferences
prefs = analytics.behavior_analyzer.get_user_preferences('user_123')
print(prefs)
```

### 2. Popularity Prediction
**Predict movie popularity**:
```python
from analytics_engine import PopularityPredictor

predictor = PopularityPredictor()
# After training...
popularity = predictor.predict_popularity([
    1000,  # vote_count
    8.5,   # vote_average
    3,     # genre_count
    5,     # cast_count
    120,   # runtime
    1      # is_recent
])
print(f"Predicted popularity: {popularity}/100")
```

### 3. Fake Engagement Detection
**Check engagement quality**:
```python
from analytics_engine import FakeEngagementDetector

detector = FakeEngagementDetector()
result = detector.detect_anomalies({
    'view_count': 50000,
    'like_count': 50,
    'engagement_rate': 0.001,
    'velocity': 5000,
    'avg_view_duration': 5
})

if result['is_suspicious']:
    print(f"⚠️ Suspicious activity detected!")
    print(f"Confidence: {result['confidence']}%")
    print(f"Flags: {result['flags']}")
```

### 4. Sentiment Trend Analysis
**Track sentiment over time**:
```python
from analytics_engine import SentimentTrendAnalyzer

analyzer = SentimentTrendAnalyzer()

# Add sentiment
analyzer.add_sentiment('Inception', 'Good', 0.85)

# Get trend
trend = analyzer.get_sentiment_trend('Inception', days=30)
print(f"Positive ratio: {trend['positive_ratio']*100:.1f}%")
print(f"Trend: {trend['trend']}")
```

---

## 🛠️ Troubleshooting

### Issue: "Error in loading Artifacts"
**Solution**: This is normal on startup. Models load lazily when first needed.

### Issue: Analytics not working
**Check**:
1. Using `app_enhanced.py` (not `app.py`)
2. `analytics_engine.py` exists in project root
3. No import errors in terminal

### Issue: No data in analytics dashboard
**Solution**: 
1. Use the app first (search movies, view recommendations)
2. Data accumulates as you use the app
3. Refresh dashboard to see updates

### Issue: Port 5000 already in use
**Solution**:
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Or change port in app
# Edit app_enhanced.py line: app.run(port=5001)
```

### Issue: Virtual environment not activating
**Solution**:
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.\venv\Scripts\Activate.ps1
```

### Issue: Module not found errors
**Solution**:
```powershell
# Ensure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

---

## 📈 Usage Examples

### Example 1: Basic Movie Search
1. Go to `http://localhost:5000`
2. Type "Inception" in search box
3. Click "Recommend"
4. View similar movies and reviews

### Example 2: View Analytics
1. Use the app (search a few movies)
2. Go to `http://localhost:5000/analytics`
3. See real-time metrics
4. Check user behavior patterns

### Example 3: API Integration
```html
<!DOCTYPE html>
<html>
<body>
  <div id="metrics"></div>
  
  <script>
    async function loadMetrics() {
      const response = await fetch('/api/analytics/dashboard');
      const data = await response.json();
      
      document.getElementById('metrics').innerHTML = `
        <h2>Dashboard Metrics</h2>
        <p>Total Sessions: ${data.user_metrics.total_sessions}</p>
        <p>Unique Users: ${data.user_metrics.unique_users}</p>
      `;
    }
    
    loadMetrics();
  </script>
</body>
</html>
```

---

## 🎓 Learning Path

### Beginner
1. Run the basic app (`app.py`)
2. Search for movies
3. Understand recommendations
4. View sentiment analysis

### Intermediate
1. Run enhanced app (`app_enhanced.py`)
2. Explore analytics dashboard
3. Use API endpoints
4. Understand user tracking

### Advanced
1. Study `analytics_engine.py`
2. Modify ML models
3. Add custom features
4. Deploy to production

---

## 📚 Additional Resources

### Files to Explore
- `app.py` - Original Flask app
- `app_enhanced.py` - Enhanced version with analytics
- `analytics_engine.py` - Analytics & ML modules
- `ENHANCED_FEATURES.md` - Complete documentation
- `NoteBook_Experiments/Analytics_Demo.md` - Demo notebook

### Key Concepts
- **Content-based filtering**: Recommends based on movie features
- **Cosine similarity**: Measures similarity between movies
- **Sentiment analysis**: Classifies reviews as Good/Bad
- **Anomaly detection**: Identifies fake engagement
- **Behavior tracking**: Monitors user interactions

---

## 🎯 Quick Commands Cheat Sheet

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run original app
python app.py

# Run enhanced app
python app_enhanced.py

# Install dependencies
pip install -r requirements.txt

# Deactivate environment
deactivate

# Check Python version
python --version

# List installed packages
pip list

# Stop running app
# Press Ctrl+C in terminal
```

---

## ✅ Checklist for First Run

- [ ] Virtual environment created
- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] No error messages
- [ ] App running on port 5000
- [ ] Can access http://localhost:5000
- [ ] Can search for movies
- [ ] Recommendations working
- [ ] Analytics dashboard accessible (enhanced version)
- [ ] API endpoints responding

---

## 🎉 Success Indicators

You know it's working when:
- ✅ Flask server starts without errors
- ✅ Home page loads with search box
- ✅ Auto-complete shows movie suggestions
- ✅ Recommendations display correctly
- ✅ Reviews show sentiment (Good/Bad)
- ✅ Analytics dashboard shows metrics (enhanced version)
- ✅ API endpoints return JSON data

---

**Need Help?** Check the full documentation in `ENHANCED_FEATURES.md`
