import numpy as np
import pandas as pd
import random

music_stats = pd.read_csv("/Users/ada/Desktop/archive/data_moods.csv")

def recommend(prediction):
    if(prediction == "happy" or prediction == "sad"):
        song = music_stats[music_stats["mood"] == "Happy"]
        song = np.random.choice(song["name"])
        return(song)
    
    if(prediction == "fear" or prediction == "angry"):
        song = music_stats[music_stats["mood"] == "Calm"]
        song = np.random.choice(song["name"])
        return(song)
    if(prediction == "disgust"):
        song = music_stats[music_stats["mood"] == "Sad"]
        song = np.random.choice(song["name"])
        return(song)

    if(prediction == "surprise" or prediction == "neutral"):
        song = music_stats[music_stats["mood"] == "Energetic"]
        song = np.random.choice(song["name"])
        return(song)
