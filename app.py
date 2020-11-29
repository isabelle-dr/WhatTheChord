import pandas as pd
import numpy as np
from numpy import asarray
from IPython.display import Audio
import pickle

import librosa
import librosa.display
import matplotlib.pyplot as plt
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import sys

# Import functions

def create_chromagram(data, sr):
    path_predictions = "prediction/"
    chromagram = librosa.feature.chroma_cens(data, sr)
    chromagram_mag = librosa.magphase(chromagram)[0]**4
    
    # save image
    fig = plt.figure(figsize=(2, 2))
    librosa.display.specshow(chromagram_mag, sr=sr, cmap='gray')
    fig.savefig(path_predictions + 'picture')
    return fig

def prep_pixels():
    path_predictions = "prediction/"
    # load photo
    data = load_img(path_predictions + 'picture.png', color_mode = "grayscale", target_size=(64,64))
    data = img_to_array(data)
    # convert to np array
    data = asarray(data)
    # normalize to range 0-1
    data = data/255.0
    data = data.reshape(1, 64, 64, 1)
    return data

def make_predictions(data):
    # to not get error messages
    model = load_model('final_model.h5')
#    y_pred = model.predict_classes(data)
    y_pred = np.argmax(model.predict(data), axis=-1)
    filename = 'labelencoder.sav'
    le = pickle.load(open(filename, 'rb'))
    y_chord = le.inverse_transform(y_pred)
    return y_chord

def predict(filename):
    path_predictions = "prediction/"
    data, sr = librosa.load(path_predictions + filename)
    create_chromagram(data, sr)
    data = prep_pixels()
    chord = make_predictions(data)
    print(f'Predicted chord{chord}')
    return chord

def main():
    if len(sys.argv) == 1:
        print('Please specify a file name')
    else:
        predict(sys.argv[1])
        
if __name__ == '__main__':
    sys.exit(main())