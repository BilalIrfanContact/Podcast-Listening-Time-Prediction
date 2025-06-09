from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

app = Flask(__name__)

# Load and train the model (or load a pre-trained model)
def load_and_train_model():
    # Load training data
    df = pd.read_csv('train.csv')
    df.drop('id', axis=1, inplace=True)
    
    # Handle missing values
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].mean())
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].mean())
    
    # Encode categorical variables
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                   'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    df['Publication_Day'] = df['Publication_Day'].map(day_mapping)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_mapping)
    df = pd.get_dummies(df, columns=['Genre', 'Publication_Time'], drop_first=True, dtype=int)
    
    # Prepare features and target
    X = df.drop(['Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'], axis=1)
    y = df['Listening_Time_minutes']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save model to file
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, X.columns

# Load model and feature columns
if os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv('train.csv')
    df.drop('id', axis=1, inplace=True)
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].mean())
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].mean())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].mean())
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                   'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['Publication_Day'] = df['Publication_Day'].map(day_mapping)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_mapping)
    df = pd.get_dummies(df, columns=['Genre', 'Publication_Time'], drop_first=True, dtype=int)
    feature_columns = df.drop(['Podcast_Name', 'Episode_Title', 'Listening_Time_minutes'], axis=1).columns
else:
    model, feature_columns = load_and_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        episode_length = float(request.form['Episode_Length_minutes'])
        host_popularity = float(request.form['Host_Popularity_percentage'])
        guest_popularity = float(request.form['Guest_Popularity_percentage'])
        number_of_ads = float(request.form['Number_of_Ads'])
        publication_day = request.form['Publication_Day']
        publication_time = request.form['Publication_Time']
        genre = request.form['Genre']
        episode_sentiment = request.form['Episode_Sentiment']
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Episode_Length_minutes': [episode_length],
            'Host_Popularity_percentage': [host_popularity],
            'Guest_Popularity_percentage': [guest_popularity],
            'Number_of_Ads': [number_of_ads],
            'Publication_Day': [publication_day],
            'Publication_Time': [publication_time],
            'Genre': [genre],
            'Episode_Sentiment': [episode_sentiment]
        })
        
        # Encode categorical variables
        day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                       'Friday': 5, 'Saturday': 6, 'Sunday': 7}
        sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        
        input_data['Publication_Day'] = input_data['Publication_Day'].map(day_mapping)
        input_data['Episode_Sentiment'] = input_data['Episode_Sentiment'].map(sentiment_mapping)
        input_data = pd.get_dummies(input_data, columns=['Genre', 'Publication_Time'], drop_first=True, dtype=int)
        
        # Align input data with training feature columns
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', prediction=f'Predicted Listening Time: {prediction:.2f} minutes')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)