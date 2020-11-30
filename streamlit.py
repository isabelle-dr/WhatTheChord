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

# imports for streamlit app
from settings import DURATION, WAVE_UPLOADED_FILE, WAVE_RECORDED_FILE, CHROMAGRAM_FILE, IMAGE_DIR
import streamlit as st

# for recording
import sounddevice as sd
from scipy.io.wavfile import write

# for image display
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    y_pred = np.argmax(model.predict(data), axis=-1)
    filename = 'labelencoder.sav'
    le = pickle.load(open(filename, 'rb'))
    y_chord = le.inverse_transform(y_pred)
    return y_chord


def predict(file, offset=0.0):
    data, sr = librosa.load(file, offset=offset)
    create_chromagram(data, sr)
    data = prep_pixels()
    chord = make_predictions(data)
    return chord

def display(file, offset=0.0):
    fig, ax = plt.subplots(figsize=(10,4))
    data, sr = librosa.load(file,  offset=offset)
    chromagram = librosa.feature.chroma_stft(data, sr=sr)
    librosa.display.specshow(chromagram, sr=sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.xlabel('seconds')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Chromagram')
    st.pyplot(fig, clear_figure=False)
    

    # record
def record(sr=22050, channels=1, duration=3):
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels)
    sd.wait()
    write('streamlit/recorded.wav', 22050, recording)
    return recording
    
    
def main():
    st.set_page_config(page_title="WhatTheChord", page_icon="🎵", layout="centered", initial_sidebar_state="expanded",)
    
    
    title = "What the chord?!"
    header = "Play it, get it."
    st.title(title)
    st.header(header)
    st.write("\n")
    st.write("\n")
    image = Image.open('streamlit/images/background.JPG')
    st.image(image, width=1000)
    
    # upload file
    st.write('Pick an audio file...')
    file =  st.file_uploader("", type="wav")
    # save it
    if file:
        with open("streamlit/uploaded.wav", 'wb') as f:
            f.write(file.read())
    
    # record 
    st.write("Or record your own using your favourite instrument!")
    if st.button("🎙️ Record it"):
        with st.spinner("Recording (2sec)..."):
            audio_file = record()
            st.success("Recording completed")
    st.write("\n")
    st.write("\n")
    
    # play
    if st.button('💿 Play it  '):
        if file:
            st.audio(WAVE_UPLOADED_FILE)
        else:
            if os.path.exists(WAVE_RECORDED_FILE):
                st.audio(WAVE_RECORDED_FILE)
            else:
                st.write("Please upload or record a file first")
    
    # classify 
    if st.button('🎶 Find the chord'):
        if file:
            with st.spinner("Classifying the chord of the file uploaded..."):
                chord = predict(WAVE_UPLOADED_FILE)
            st.success("Classification completed")
            st.write("### The recorded chord is...         ", list(chord)[0], "!")
            st.write("\n")
            #st.balloons()
        else: 
            if os.path.exists(WAVE_RECORDED_FILE):
                with st.spinner("Classifying the chord of the file recorded..."):
                    chord = predict(WAVE_RECORDED_FILE, offset=0.9)
                st.success("Classification completed")
                st.write("### The recorded chord is...         ", list(chord)[0], "!")
                st.write("\n")
                #st.balloons()
            else:
                st.write("Please upload or record a file first")
     
    # display chromagram
    if st.button('📊 Display Chromagram'):
        if file:
            display(WAVE_UPLOADED_FILE)
            st.write("Did you know humans perceive two musical pitches as similar colors if they differ by an octave?")
            st.write("A chromagram indicates how much energy of each pitch class is present, by aggregating it's decibel values over the 10 octaves.")
            st.write("The classifier used in this app is a trained Neural Network that uses images like these to predict the chord of an audio input.")
        else:
            if os.path.exists(WAVE_RECORDED_FILE):
                display(WAVE_RECORDED_FILE, offset=0.9)
                st.write("Did you know humans perceive two musical pitches as similar colors if they differ by an octave?")
                st.write("A chromagram indicates how much energy of each pitch class is present, by aggregating it's decibel values over the 10 octaves.")
                st.write("The classifier used in this app is a trained Neural Network that uses images like these to predict the chord of an audio input.")
            else:
                st.write("Please upload or record a file first")
    
if __name__ == '__main__':
    main()    
    
