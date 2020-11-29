# Chord-predictor
- feature extraction of chromagrams from audio data
- train deep learning model
- make prediction on a new audio sample

## The chords
The data is composed of 180 audio samples containing the following chords:
0  --> A#:7
1  --> A#:hdim7
2  --> A#:maj
3  --> A#:min
4  --> A:7
5  --> A:hdim7
6  --> A:maj
7  --> A:min
8  --> B:maj
9  --> B:min
10  --> C#:7
11  --> C#:hdim7
12  --> C#:maj
13  --> C#:min
14  --> C:7
15  --> C:hdim7
16  --> C:maj
17  --> C:min
18  --> D#:7
19  --> D#:maj
20  --> D#:min
21  --> D:7
22  --> D:maj
23  --> D:min
24  --> E:7
25  --> E:hdim7
26  --> E:maj
27  --> E:min
28  --> F#:7
29  --> F#:maj
30  --> F#:min
31  --> F:7
32  --> F:hdim7
33  --> F:maj
34  --> F:min
35  --> G#:7
36  --> G#:hdim7
37  --> G#:maj
38  --> G#:min
39  --> G:hdim7
40  --> G:maj
41  --> G:min

## The chromagrams
I used CENS : Chroma Energy Normalized using Librosa

## The model
I used a CNN

# To run the whole project - do I need this here?
- Install requirements.txt
- Dowbload the audio data in a folder called audio
- Download the annotations data in a folder called annotation
- run extract_features.py

# To make predicitons
Clone this repo\
``
$ git clone https://github.com/Isabelle-Dr/Chord-predictor
$ cd Chord-predictor
``

Install requirements.txt\
``
pip install -r requirements.txt
``

The model is trained on samples from audio data containing only one chord. The notes don't have to be played simultaneously :)\
Put an audio file of your choice in the prediction folder (.wav or .m4a are working)\
Call the following command
``python -W ignore app.py <filename>`` in your terminal. (the W - ignore is here so that the warnings aren't being printed out)\
replace <filename> with the name of your file including the extension ex: audio_sample.wav
Make sure you are located in the folder containing the app.py file and the prediction folder
In the prediciton folder, you'll also see the chromagram image of your song!
