# Realtime-Emotion-Detection-and-Song-Recommendation
Trained a Keras model using FER-2013 Dataset to predict emotions from Images. Using web camera, the model predicts the user's current emotion from their facial expression and recommends a song based off of current emotion.

vid.py --> web camera access and model prediction
emo.ipynb --> accessing data and training model
music.py --> contains function to recommend music from model prediction
music.ipynb --> display data from data_moods.csv

Data Files:
data_moods.csv --> data music set
FER2013 dataset
haarcascade_frontalface_default.xml --> web camera access 

Model: der.h5, fer.json
