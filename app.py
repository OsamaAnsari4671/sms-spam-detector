# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 07:46:23 2023

@author: Osama Ansari
"""
import numpy as np
import streamlit as st
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()




# Loading the model
model = pickle.load(open("D:/osama/sms-spam-detector/model.pkl", "rb"))

    
# Vectorizer
vectorize_text = pickle.load(open("D:/osama/sms-spam-detector/vectorizer.pkl", "rb"))




def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    filtered_words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

def predict_spam(text):
    #text preprocessing
    preprocessed_text = preprocess_text(text)
    
    #vectorization
    vect_features = vectorize_text.transform([preprocessed_text])
    
    #prediction using the model
    prediction = model.predict(vect_features)[0]
    
    if prediction == 1:
        label = 'Spam'
    else:
        label = 'Not Spam'
    return label

# app
st.title("Spam Detector")

# Sidebar with app information
st.sidebar.markdown(
    """
    ## About
    This app predicts whether a given SMS message is spam or not.

    **Instructions:**
    1. Enter a message in the text box.
    2. Click the "Detect" button to see the prediction result.
    """
)

#user input
user_input = st.text_input("Enter a message....")
if st.button("Detect"):
    if user_input:
        result = predict_spam(user_input)
        st.write("Prediction:", result)
    else:
        st.write("Please enter a message to detect.")