# What the chord
Chord predictor using a CNN on choma vectors


## About the project
This project was part of LighthouseLabs data science bootcamp and took two weeks to complete. It consisted of:
- research on audio transformations
- get a dataset: (I used the [guitarset dataset](https://zenodo.org/record/1492449#.X8QhCGhKhPY)) 
- feature extraction of chromagrams from audio data
- train deep learning model
- make prediction on a new audio sample: using the command line or through a streamlit web app.

I ended up an accuracy of 80% which is acceptable for the scope of the project. I was using data containing 42 labels with a high class imbalance, with audio files varying a lot around the 'root' chord. I'm planning on tuning the model to reach a higher accuracy :)
The goal of this project is more to build an MVP model from start to finish using all steps of a data science project, and use this model for prediction smoothly. 

### The chords
The data is composed of 180 audio samples containing the following chords:
0  --> A#:7\
1  --> A#:hdim7\
2  --> A#:maj\
3  --> A#:min\
4  --> A:7\
5  --> A:hdim7\
6  --> A:maj\
7  --> A:min\
8  --> B:maj\
9  --> B:min\
10  --> C#:7\
11  --> C#:hdim7\
12  --> C#:maj\
13  --> C#:min\
14  --> C:7\
15  --> C:hdim7\
16  --> C:maj\
17  --> C:min\
18  --> D#:7\
19  --> D#:maj\
20  --> D#:min\
21  --> D:7\
22  --> D:maj\
23  --> D:min\
24  --> E:7\
25  --> E:hdim7\
26  --> E:maj\
27  --> E:min\
28  --> F#:7\
29  --> F#:maj\
30  --> F#:min\
31  --> F:7\
32  --> F:hdim7\
33  --> F:maj\
34  --> F:min\
35  --> G#:7\
36  --> G#:hdim7\
37  --> G#:maj\
38  --> G#:min\
39  --> G:hdim7\
40  --> G:maj\
41  --> G:min\

### The data
This dataset consists of 360 audio files of approx. 30sec and annotations about the chord instructed to the player as well as the chord played (it differs slightly since the musicians had some leeway to fit a speicif music style.
The dataset is composed of 180 tracks, each having one 'comp' audio file and one 'solo' audio file. I took only the 'comp' in this project to get cleaner images and because the musicians didn't go too far from what was instructed.

### The images
I used chomagrams CENS : Chroma Energy Normalized using Librosa

## The model
 I used a CNN

# Make a prediction
## Using the command line
The model is trained on samples from audio data containing only one chord. The notes don't have to be played simultaneously :)\
Make sure you are located in the folder containing the app.py file and the prediction folder

- Clone this repo\
``
$ git clone https://github.com/Isabelle-Dr/WhatTheChord.git
$ cd WhatTheChord
``
- Install virtual environment using pip, activate it and install requirements.txt\
``
$ pip install virtualenv
$ virtualenv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
``
- Make a prediction from the command line
Create a folder called `prediction` in the repo and put the audio files you want to predict in the `prediction` folder (.wav or .mp3). Then, run this command from the comand line. 

``
$ python -W ignore app.py <yourfilename.wav>
``

The `W - ignore` is here so that the warnings aren't being printed out, it makes a cleaner output.
Make sure you're situated in the `what-the-chord`directory.

- That's it! You'll see the predicted chord right after this command :) In the prediciton folder, you'll also see the chromagram image of your song!

## Using the Streamlit app

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
- Create three folders in the directory: `audio`, `annotation`, `images`, `labels`, `prediction`
Your repo folder should have this structure:
```bash
org/repo/
├── WhatTheChord/
|           ├── annotation/
|           ├── audio/
|           ├── images/
|           ├── labels/
|           ├── prediction/
|           ├── app.py
|           ├── data_extraction.py
|           ├── final_model.h5
|           ├── labelencoder.sav
|           ├── modeling.py
|           └── requirements.txt
```

- Dowbload the [guitarset dataset](https://zenodo.org/record/1492449#.X8QhCGhKhPY), store all the audio files in the `audio` folder and the annotation files in the annotation folder you just created
- run ` python data_extraction.py`
It might take a while. You'll see chomagrams being created in the images folder, exciting!
- run `python modeling.py`
It might take 15-20min. After it, you're done! Your brand new model will be saved as 'final_model.h5' and it's ready for prediction!
