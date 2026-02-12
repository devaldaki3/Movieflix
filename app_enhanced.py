"""
Enhanced Movie Recommendation System with Analytics
Includes: Analytics Dashboard, User Behavior Tracking, Popularity Prediction
"""

import json
import pickle
import requests
import bs4 as bs
import numpy as np
import pandas as pd
import urllib.request
from flask import Flask, render_template, request, jsonify, session
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import uuid

# Import analytics engine
try:
    from analytics_engine import get_analytics_engine
    analytics = get_analytics_engine()
    ANALYTICS_ENABLED = True
except ImportError:
    ANALYTICS_ENABLED = False
    print("Analytics engine not available")

# loading the dataset and the trained model
try:
    clf = pickle.load(open("./Artifacts/nlp_model.pkl", 'rb'))
    vectorizer = pickle.load(open("./Artifacts/tranform.pkl",'rb'))
except:
    print("Error in loading Artifacts")

# creating a similarity matrix using count vectorizer and cosine similarity
def create_similarity():
    try:
        data = pd.read_csv("./Artifacts/main_data.csv")
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data['comb']) 
        similarity = cosine_similarity(count_matrix)
        return data,similarity
    except Exception as e:
        print(e)

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape 
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "[\"abc\",\"def\"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('./Artifacts/main_data.csv')
    return list(data['movie_title'].str.capitalize())

def get_user_id():
    """Get or create user session ID"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

app = Flask(__name__)
app.secret_key = 'entertainment_analytics_secret_key_2024'

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    
    # Track user activity
    if ANALYTICS_ENABLED:
        user_id = get_user_id()
        analytics.behavior_analyzer.track_user_activity(user_id, None, 'home_visit')
    
    return render_template('home.html', suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    
    # Track search activity
    if ANALYTICS_ENABLED:
        user_id = get_user_id()
        analytics.behavior_analyzer.track_user_activity(user_id, movie, 'search')
    
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Track movie view
    if ANALYTICS_ENABLED:
        user_id = get_user_id()
        analytics.behavior_analyzer.track_user_activity(user_id, title, 'view')

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            sentiment = 'Good' if pred else 'Bad'
            reviews_status.append(sentiment)
            
            # Track sentiment for analytics
            if ANALYTICS_ENABLED:
                analytics.sentiment_analyzer.add_sentiment(title, sentiment, pred[0])

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)


# ==================== NEW ANALYTICS ENDPOINTS ====================

@app.route("/analytics")
def analytics_dashboard():
    """Analytics dashboard page"""
    if not ANALYTICS_ENABLED:
        return "Analytics not available", 503
    
    return render_template('analytics.html')

@app.route("/api/analytics/eda")
def get_eda_visualizations():
    """Get EDA statistics for client-side visualization"""
    try:
        from eda_stats import generate_all_statistics
        
        print("Generating EDA statistics...")
        result = generate_all_statistics()
        
        if result and result.get('success'):
            print("✅ EDA statistics generated successfully")
            return jsonify(result['statistics'])
        else:
            print("❌ Failed to generate statistics")
            return jsonify({'error': 'Failed to generate statistics'}), 500
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        print(f"❌ EDA Error: {error_msg}")
        print(f"Traceback: {traceback_msg}")
        return jsonify({'error': error_msg}), 500

@app.route("/api/analytics/sentiment")
def get_sentiment_analysis():
    """Get sentiment analysis results"""
    try:
        from sentiment_analyzer import analyze_movie_sentiments
        
        print("Generating sentiment analysis...")
        results = analyze_movie_sentiments()
        
        if results:
            print("✅ Sentiment analysis completed successfully")
            return jsonify(results)
        else:
            print("❌ Failed to generate sentiment analysis")
            return jsonify({'error': 'Failed to generate sentiment analysis'}), 500
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        print(f"❌ Sentiment Analysis Error: {error_msg}")
        print(f"Traceback: {traceback_msg}")
        return jsonify({'error': error_msg}), 500

@app.route("/api/analytics/ml-predictions")
def get_ml_predictions():
    """Get ML-based popularity predictions"""
    try:
        from ml_predictor import train_and_evaluate_models
        
        print("Generating ML predictions...")
        results = train_and_evaluate_models()
        
        if results:
            print("✅ ML predictions completed successfully")
            return jsonify(results)
        else:
            print("❌ Failed to generate ML predictions")
            return jsonify({'error': 'Failed to generate ML predictions'}), 500
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        print(f"❌ ML Prediction Error: {error_msg}")
        print(f"Traceback: {traceback_msg}")
        return jsonify({'error': error_msg}), 500

@app.route("/api/analytics/dashboard")
def get_dashboard_data():
    """Get dashboard metrics"""
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    metrics = analytics.get_dashboard_metrics()
    return jsonify(metrics)

@app.route("/api/analytics/user-behavior")
def get_user_behavior():
    """Get user behavior analytics"""
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    patterns = analytics.behavior_analyzer.get_consumption_patterns()
    return jsonify(patterns)

@app.route("/api/analytics/trending")
def get_trending_movies():
    """Get trending movies based on sentiment"""
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    trending = analytics.sentiment_analyzer.get_trending_movies()
    return jsonify(trending)

@app.route("/api/analytics/movie/<movie_id>")
def get_movie_analytics(movie_id):
    """Get analytics for a specific movie"""
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    # Sample engagement data (in real app, this would come from database)
    engagement_data = {
        'view_count': 1500,
        'like_count': 120,
        'comment_count': 45,
        'share_count': 30,
        'view_duration': 85,
        'unique_viewers': 1200,
        'repeat_viewers': 300,
        'engagement_rate': 0.13,
        'velocity': 150,
        'avg_view_duration': 85
    }
    
    analysis = analytics.analyze_movie_performance(movie_id, engagement_data)
    return jsonify(analysis)

@app.route("/api/analytics/fake-detection", methods=["POST"])
def detect_fake_engagement():
    """Detect fake engagement in provided metrics"""
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    data = request.get_json()
    result = analytics.fake_detector.detect_anomalies(data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
