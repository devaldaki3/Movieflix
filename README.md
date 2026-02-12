# Movie-Recommendation-System

# About this Project

Movie recommendation systems are designed to provide personalized movie suggestions to users, enhancing their entertainment experience by helping them discover movies tailored to their preferences. This project showcases the development of an End-to-End Movie Recommendation System with advanced analytics capabilities using various machine-learning models and techniques. 

The primary goal of this project is to build a robust movie recommendation system that can analyze user preferences and viewing history to make accurate movie suggestions. It utilizes popular machine-learning algorithms to classify movies, generate personalized recommendations, and provide comprehensive analytics on user behavior, content popularity, sentiment trends, and engagement patterns.

## Key Features

- **Movie Recommendations**: Content-based filtering using cosine similarity
- **Sentiment Analysis**: NLP-based review classification from IMDB
- **Analytics Dashboard**: Real-time user behavior and content performance tracking
- **Popularity Prediction**: ML-based content popularity forecasting
- **Fake Engagement Detection**: Anomaly detection for suspicious activity
- **Sentiment Trend Analysis**: Time-series sentiment tracking
- **User Behavior Tracking**: Session analytics and consumption patterns
- **EDA Capabilities**: Comprehensive exploratory data analysis

## Built-With

 - Flask
 - Numpy
 - Scipy
 - Nltk
 - Scikit-learn==1.2.2
 - Pandas
 - Beautifulsoup4
 - tmdbv3api
 - DVC

 Anyways you can install all the libraries mentioned above at a glance by executing the following command:
 
  ```sh
  pip install -r requirements.txt
  ```

## Getting Started

This will help you understand how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Installation Steps

### Option 1: Installation from GitHub

Follow these steps to install and set up the project directly from the GitHub repository:

1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to install the project.
   - Run the following command to clone the GitHub repository:
     ```
     git clone https://github.com/KalyanMurapaka45/End-to-End-Movie-Recommendation-System.git
     ```

2. **Create a Virtual Environment** (Optional but recommended)
   - It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```
     conda create -p <Environment_Name> python==<python version> -y
     ```

3. **Activate the Virtual Environment** (Optional)
   - Activate the virtual environment based on your operating system:
       ```
       conda activate <Environment_Name>/
       ```

4. **Install Dependencies**
   - Navigate to the project directory:
     ```
     cd [project_directory]
     ```
   - Run the following command to install project dependencies:
     ```
     pip install -r requirements.txt
     ```

5. **Run the Project**
   - Start the project by running the appropriate command.
   
   **Option A: Basic Version (Recommendations Only)**
     ```
     python app.py
     ```
   
   **Option B: Enhanced Version (With Analytics Dashboard)**
     ```
     python app_enhanced.py
     ```

6. **Access the Project**
   - **Main Application**: Open http://localhost:5000 in your web browser
   - **Analytics Dashboard**: Open http://localhost:5000/analytics (enhanced version only)
  
<br><br>
### Option 2: Installation from DockerHub

If you prefer to use Docker, you can install and run the project using a Docker container from DockerHub:

1. **Pull the Docker Image**
   - Open your terminal or command prompt.
   - Run the following command to pull the Docker image from DockerHub:
     ```
     docker pull kalyan45/movierecommend-app
     ```

2. **Run the Docker Container**
   - Start the Docker container by running the following command, and mapping any necessary ports:
     ```
     docker run -p 5000:5000 kalyan45/movierecommend-app
     ```

3. **Access the Project**
   - **Main Application**: Open http://localhost:5000 in your web browser
   - **Analytics Dashboard**: Open http://localhost:5000/analytics



   
# Features & Capabilities

## Core Features
- **Movie Search**: Auto-complete search functionality for easy movie discovery
- **Smart Recommendations**: Get 10 similar movies based on content-based filtering
- **Sentiment Analysis**: Real-time IMDB review scraping and sentiment classification
- **Movie Details**: View cast, ratings, runtime, genres, and overview
- **User Reviews**: See aggregated reviews with sentiment indicators (Good/Bad)

## Analytics Features (Enhanced Version)
- **User Behavior Analytics**: Track user interactions, sessions, and preferences
- **Content Performance**: Monitor movie popularity and engagement metrics
- **Sentiment Trends**: Analyze sentiment patterns over time
- **Fake Engagement Detection**: Identify suspicious activity and bot behavior
- **Popularity Prediction**: ML-based forecasting of content popularity
- **Real-time Dashboard**: Interactive analytics dashboard with auto-refresh
- **EDA Tools**: Comprehensive exploratory data analysis capabilities

## API Endpoints

### Main Application
- `GET /` or `/home` - Home page with movie search
- `POST /similarity` - Get similar movies
- `POST /recommend` - Get movie recommendations with details

### Analytics API (Enhanced Version)
- `GET /analytics` - Analytics dashboard
- `GET /api/analytics/dashboard` - Dashboard metrics (JSON)
- `GET /api/analytics/user-behavior` - User behavior data (JSON)
- `GET /api/analytics/trending` - Trending movies (JSON)
- `GET /api/analytics/eda` - EDA statistics (JSON)
- `GET /api/analytics/sentiment` - Sentiment analysis results (JSON)
- `GET /api/analytics/ml-predictions` - ML predictions (JSON)
- `GET /api/analytics/movie/<movie_id>` - Movie-specific analytics (JSON)
- `POST /api/analytics/fake-detection` - Fake engagement detection (JSON)

# Project Structure

```
End-to-End-Movie-Recommendation-System/
├── app.py                          # Original Flask application
├── app_enhanced.py                 # Enhanced app with analytics
├── analytics_engine.py             # Analytics & ML modules
├── sentiment_analyzer.py           # Sentiment analysis module
├── ml_predictor.py                 # ML prediction module
├── eda_stats.py                    # EDA statistics generator
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── Artifacts/                      # Models & datasets
│   ├── nlp_model.pkl              # Sentiment analysis model
│   ├── tranform.pkl               # Text vectorizer
│   ├── main_data.csv              # Movie dataset
│   └── movies.csv                 # Extended movie data
├── templates/                      # HTML templates
│   ├── home.html                  # Landing page
│   ├── recommend.html             # Recommendation results
│   └── analytics.html             # Analytics dashboard
├── static/                         # CSS, JS, images
│   ├── style.css
│   ├── recommend.js
│   └── autocomplete.js
├── NoteBook_Experiments/           # Jupyter notebooks
│   ├── Exploratory Data Analysis.ipynb
│   ├── Movie Recommendation System.ipynb
│   └── Sentimental Analysis on Reviews.ipynb
├── ENHANCED_FEATURES.md            # Detailed feature documentation
├── QUICK_START.md                  # Quick start guide
├── EDA_README.md                   # EDA documentation
└── README.md                       # This file
```



Contributions make the open-source community such an amazing place to learn, inspire, and create. I would greatly appreciate any contributions you make.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch 
3. Commit your Changes 
4. Push to the Branch 
5. Open a Pull Request

<!-- LICENSE -->
# Contributing

Contributions make the open-source community such an amazing place to learn, inspire, and create. I would greatly appreciate any contributions you make.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch 
3. Commit your Changes 
4. Push to the Branch 
5. Open a Pull Request

<!-- LICENSE -->
# License

Distributed under the GNU General Public License v3.0. See `LICENSE.txt` for more information.

# Acknowledgements

This project demonstrates end-to-end machine learning implementation for movie recommendations with advanced analytics capabilities. We acknowledge the open-source Python libraries used in this project, the TMDB API for movie data, IMDB for review data, and their contributors. Special thanks to the machine learning and data science community for their valuable resources and tools.

