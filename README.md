# What the chord
Chord predictor using a CNN on choma vectors


## About the project
- feature extraction of chromagrams from audio data
- train deep learning model
- make prediction on a new audio sample

### The chords
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

### The data
Guitarset dataset

### The images
I used chomagrams CENS : Chroma Energy Normalized using Librosa

## The model
I used a CNN

# Make a prediction
The model is trained on samples from audio data containing only one chord. The notes don't have to be played simultaneously :)\
Call python -W ignore app.py in your terminal. (the W - ignore is here so that the warnings aren't being printed out)\
Make sure you are located in the folder containing the app.py file and the prediction folder

- Clone this repo\
``
$ git clone https://github.com/Isabelle-Dr/what-the-chord.git
$ cd what-the-chord
``
- Install virtual environment using pip, activate it and install requirements.txt\
``
$ pip install virtualenv
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
``
- Make a prediction from the command line
Put the audio files you want to predict in the `prediction` folder (.wav or .mp3). Then, run this command from the comand line. \

``
$ python -W ignore app.py <yourfilename.wav>
``

The `W - ignore` is here so that the warnings aren't being printed out, it makes a cleaner output.
Make sure you're situated in the `what-the-chord`directory.\

- That's it! You'll see the predicted chord right after this command :) In the prediciton folder, you'll also see the chromagram image of your song!

# Build the project from the source
- Clone this repo\
``
$ git clone https://github.com/Isabelle-Dr/what-the-chord.git
$ cd what-the-chord
``
- Install virtual environment using pip, activate it and install requirements.txt\
``
$ pip install virtualenv
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
``
- Create two folders in the directory: `data` and `annotation`

- Dowbload the [guitarset dataset](https://zenodo.org/record/1492449#.X8QhCGhKhPY), store all the audio files in the data folder and the annotation files in the annotation folder
- run extract_features.py



