# This is used to make predictions fromt he command line

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
from settings import PRED_DIR, MODEL_H5, LE

# Import functions
def create_chromagram(data, sr):
    chromagram = librosa.feature.chroma_cens(data, sr, fmin=75)
    chromagram_mag = librosa.magphase(chromagram)[0]**4
    # save image
    fig = plt.figure(figsize=(2, 2))
    librosa.display.specshow(chromagram_mag, sr=sr, cmap='gray')
    fig.savefig(PRED_DIR + 'picture')
    return fig

def prep_data():
    # load photo
    data = load_img(PRED_DIR + 'picture.png', color_mode = "grayscale", target_size=(64,64))
    data = img_to_array(data)
    # convert to np array
    data = asarray(data)
    # normalize to range 0-1
    data = data/255.0
    data = data.reshape(1, 64, 64, 1)
    return data

def make_predictions(data):
    model = load_model(MODEL_H5)
    y_pred = np.argmax(model.predict(data), axis=-1)
    le = pickle.load(open(LE, 'rb'))
    y_chord = le.inverse_transform(y_pred)
    return y_chord

def predict(filename):
    data, sr = librosa.load(PRED_DIR + filename)
    create_chromagram(data, sr)
    data = prep_data()
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