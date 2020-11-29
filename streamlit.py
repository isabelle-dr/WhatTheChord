# imports for model
import pandas as pd
import numpy as np
from numpy import asarray
from IPython.display import Audio
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sys


# other imports
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE, CHROMAGRAM_FILE
import streamlit as st



def create_chromagram(data, sr):
    #path_predictions = "prediction/"
    chromagram = librosa.feature.chroma_cens(data, sr)
    chromagram_mag = librosa.magphase(chromagram)[0]**4
    
    # save image
    fig = plt.figure(figsize=(2, 2))
    librosa.display.specshow(chromagram_mag, sr=sr, cmap='gray')
    fig.savefig(os.path.join(IMAGE_DIR, 'chromagram.jpg'))
    return fig

def prep_pixels():
    #path_predictions = "prediction/"
    # load photo
    data = load_img(os.path.join(IMAGE_DIR, 'chromagram.jpg'), color_mode = "grayscale", target_size=(64,64))
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

def predict(file):
    data, sr = librosa.load(file)
    create_chromagram(data, sr)
    data = prep_pixels()
    chord = make_predictions(data)
    return chord


def display(file):
    fig, ax = plt.subplots(figsize=(10,4))
    data, sr = librosa.load(file)
    chromagram = librosa.feature.chroma_stft(data, sr=sr)
    librosa.display.specshow(chromagram, sr=sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Chromagram')
    st.pyplot(fig, clear_figure=False)

    
def main():
    title = "Guitar Chord Recognition"
    st.title(title)
    
    # upload file - working!
    file =  st.file_uploader("Choose an audio file...", type="wav")
    
    # play - working !
    if st.button('Play'):
        if file:
            audio_bytes = file.read()
            st.audio(audio_bytes, format='audio/wav')
        else:
            st.write("Please upload a file first")
    
    # classify - need to link to uploaded file...
    if st.button('Classify'):
        if file:
            audio_bytes = file.read()
            with st.spinner("Classifying the chord"):
                chord = predict(WAVE_OUTPUT_FILE)
            st.success("Classification completed")
            st.write("### The recorded chord is...", list(chord)[0], "!")
            if chord == 'N/A':
                st.write("Please record sound first")
            st.write("\n")
            st.balloons()
        else: 
            st.write("Please upload a file first")
     
    # display image - need to link to uploaded file...
    if st.button('Display Chromagram'):
        if os.path.exists(WAVE_OUTPUT_FILE):
            display(WAVE_OUTPUT_FILE)

        else:
            st.write("Please record sound first")
    
if __name__ == '__main__':
    main()    
    