#importing the essential libraries
from flask import Flask , render_template , request
import pickle
import numpy as np

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

#creating flask instance
app = Flask(__name__)

#home page flask code
@app.route('/')
def home():
  return render_template('home.html')


#prediction page flask code
@app.route('/predict' , methods=['POST'])
def predict():
  message = request.form['text']
  result = sid.polarity_scores(message)
  value = result['compound']
  return render_template('result.html' , prediction = value , msg = message)


#Running the flask app
if __name__ == '__main__':
  app.run(debug=True)
