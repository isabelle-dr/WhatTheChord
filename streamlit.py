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
# background image2
import base64

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
def record(sr=22050, channels=1, duration=4):
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels)
    sd.wait()
    write('streamlit/recorded.wav', 22050, recording)
    return recording
    
    import base64

# bacground  - from local file
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    height: 700px;
    position: relative;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
    #    position: relative;
    #     height: 1000px;
   # size: cover;
    
def main():
    # layout
    st.set_page_config(page_title="WhatTheChord", page_icon="🎵", layout="centered", initial_sidebar_state="expanded",)

    set_png_as_page_bg('streamlit/images/s1.gif')
    
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    
    # record 
    #st.write("Record it with your favourite instrument!")
    if st.button("🎙️ Record it"):
        set_png_as_page_bg('streamlit/images/s2.gif')
        with st.spinner("Recording for 4 sec..."):
            audio_file = record()
            st.success("Recording completed!")

    
    #upload file
    if st.button("🎙️ Upload it"):
        file =  st.file_uploader("", type="wav")
        #save it
        if file:
            with open("streamlit/uploaded.wav", 'wb') as f:
                f.write(file.read())
        #st.success("File uploaded!")
    
    st.write("\n")
    st.write("\n")
    st.write("\n")
    
    #play - with uploading button
    if st.button('💿 Play it  '):
        # from local
        #set_png_as_page_bg('streamlit/images/chromagram.jpg')
        if file:
            st.audio(WAVE_UPLOADED_FILE)
        else:
            if os.path.exists(WAVE_RECORDED_FILE):
                st.audio(WAVE_RECORDED_FILE)
            else:
                st.write("Please upload or record a file first")
    
    # classify - with file uploading button
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
                st.success("Classification completed!")
                st.write("### The recorded chord is...         ", list(chord)[0])
                st.write("\n")
                #st.balloons()
            else:
                st.write("Please upload or record a file first")
     
   # display chromagram - - with file uploading button
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
    
    
    
    

    # display an image
#     image = Image.open('streamlit/images/test.gif')
#     st.image(image, width=1000)

    # disply video
    #vid=open("streamlit/images/test.gif","rb")
    #st.video(vid)
    
    
    
    # background - url
#     page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# height: 1300px;
# background-size: cover;
# position: relative;}
# </style>
# '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)
    
    
    
#     # background - local
#     def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = '''
#     <style>
#     body {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return
    
#     set_png_as_page_bg('streamlit/images/chromagram.jpg')


    # gifs
# """### gif from url"""
# st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")

# """### gif from local file"""
# file_ = open("/home/rzwitch/Desktop/giphy.gif", "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# file_.close()

# st.markdown(
#     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
#     unsafe_allow_html=True,
# )





    # play - with uploading button
#     if st.button('💿 Play it  '):
#         # from local
#         #set_png_as_page_bg('streamlit/images/chromagram.jpg')
#         if file:
#             st.audio(WAVE_UPLOADED_FILE)
#         else:
#             if os.path.exists(WAVE_RECORDED_FILE):
#                 st.audio(WAVE_RECORDED_FILE)
#             else:
#                 st.write("Please upload or record a file first")
    
    
    
#     # classify - with file uploading button
#     if st.button('🎶 Find the chord'):
#         if file:
#             with st.spinner("Classifying the chord of the file uploaded..."):
#                 chord = predict(WAVE_UPLOADED_FILE)
#             st.success("Classification completed")
#             st.write("### The recorded chord is...         ", list(chord)[0], "!")
#             st.write("\n")
#             #st.balloons()
#         else: 
#             if os.path.exists(WAVE_RECORDED_FILE):
#                 with st.spinner("Classifying the chord of the file recorded..."):
#                     chord = predict(WAVE_RECORDED_FILE, offset=0.9)
#                 st.success("Classification completed!")
#                 st.write("### The recorded chord is...         ", list(chord)[0])
#                 st.write("\n")
#                 #st.balloons()
#             else:
#                 st.write("Please upload or record a file first")



    # display chromagram - - with file uploading button
#     if st.button('📊 Display Chromagram'):
#         if file:
#             display(WAVE_UPLOADED_FILE)
#             st.write("Did you know humans perceive two musical pitches as similar colors if they differ by an octave?")
#             st.write("A chromagram indicates how much energy of each pitch class is present, by aggregating it's decibel values over the 10 octaves.")
#             st.write("The classifier used in this app is a trained Neural Network that uses images like these to predict the chord of an audio input.")
#         else:
#             if os.path.exists(WAVE_RECORDED_FILE):
#                 display(WAVE_RECORDED_FILE, offset=0.9)
#                 st.write("Did you know humans perceive two musical pitches as similar colors if they differ by an octave?")
#                 st.write("A chromagram indicates how much energy of each pitch class is present, by aggregating it's decibel values over the 10 octaves.")
#                 st.write("The classifier used in this app is a trained Neural Network that uses images like these to predict the chord of an audio input.")
#             else:
#                 st.write("Please upload or record a file first")