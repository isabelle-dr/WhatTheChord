# What the chord
Chord predictor using a Convolutional Neural Network on image representations of audio

## About the project
This project was part of LighthouseLabs data science bootcamp and took two weeks to complete. It consisted of:
- research on audio transformation and manipulations in python
- get a dataset with chord annotations
- data processing
- feature extraction of chromagrams from audio data
- train deep learning model
- make prediction on a new audio sample: using the command line or through a streamlit web app.

### The [dataset](https://zenodo.org/record/1492449#.X8QhCGhKhPY)
The dataset is composed of 180 tracks, each having one *comp* audio file and one *solo* audio file, making a total of 360 audio files of approximatively 30sec.
This audio data comes with precise annotations about the chords. 
There are two informations regarding the chords: the chords instructed to the player and chord that they actually played (the two differ slightly since the musicians had some leeway to fit a speicif music style, they were improvising around a root chord on a given style). 

After exploring the data, I noticed that in the *solo* audio files, the musicians got a bit far from the instructed chord, compared to the *comp* audio files. 
I narrowed down this dataset in the following way during the data processing phrase:
- use only *comp* audio tracks (180)
- remove very under representated classes (all hdim and 7 chords)
- keep only slices of audio where the chord played is "close enough" to the chord instructed

### The chords
The chords that this model is capable of transcribing are the 24 most common chords:
A:maj, A:min, A#:maj, A#:min\
B:maj, B:min\
C:maj, C:min, C#:maj, C#:min\
D:maj, D:min, D#:maj, D#:min\
E:maj, E:min\
F:maj, F:min, F#:maj, F#:min\
G:maj, G:min, G#:maj, G#:min\

### The image representations
I used chomagrams CENS (Chroma Energy Normalized) using the [Librosa](https://librosa.org/doc/latest/index.html) pakage. This transformation smoothes out local deviations by taking statistics over large windows. It gave better performance for this particular dataset model because of the high variability around root chords in the dataset. 
I also used the [magphase](https://librosa.org/doc/0.8.0/generated/librosa.magphase.html) transformation to de-noise images.

![alt text](https://github.com/Isabelle-Dr/WhatTheChord/blob/main/readme_images/chromagrams.png?raw=true)

## The model
 I used a Convolutional Neural network with two hidden layers and used Maxpooling layers.
 
# Make a prediction
## Using the command line
Make sure the audio files you feed the model only contain a variation of one chord for best results.

- Clone this repo\
``
$ git clone https://github.com/Isabelle-Dr/WhatTheChord.git
$ cd WhatTheChord
``
- Install virtual environment using pip, activate it and install requirements.txt\
``
$ pip install virtualenv\
$ virtualenv .venv\
$ source .venv/bin/activate\
$ pip install -r requirements.txt
``
- Make a prediction from the command line
Put the audio files you want to predict in the `prediction` folder (.wav or .mp3). There is already some files in that folder if you want to use them for prediction
Then, run this command from the comand line (make sure you're situated in the `what-the-chord`directory)

``
$ python app.py <yourfilename.wav>
``

- That's it! You'll see the predicted chord right after this command :) In the prediciton folder, you'll also see the chromagram image of your song!

## Using the Streamlit app
- Run the following code\
``
$ streamlit run streamlit.py
``
- The webapp is now launched in your browser! If not, you can open it at [http://localhost:8501](http://localhost:8501)
Play aorund with it, you can either record your own sample or upload a file.

# Build the project from the source
- Clone this repo\
``
$ git clone https://github.com/Isabelle-Dr/what-the-chord.git
$ cd what-the-chord
``
- Install virtual environment using pip, activate it and install requirements.txt\
``
$ pip install virtualenv\
$ virtualenv .venv\
$ source .venv/bin/activate\
$ pip install -r requirements.txt
``

- Dowbload the [guitarset dataset](https://zenodo.org/record/1492449#.X8QhCGhKhPY), store all the audio files in the `audio` folder and the annotation files in the `annotation` folders under the `data_prepared` folder
- run ` python data_extraction.py`
It might take a while. You'll see chomagrams being created in the `data_prepared/images` folder, exciting!
- run `python modeling.py`
It might take 15-20min. After it, you're done! Your brand new model will be saved in the `pickles` folder, and it's ready for prediction!


# Repo structure
```bash
org/repo/
├── WhatTheChord/
|           ├── data_prepared/
|           |        └── images/           # where chromagrams will be stored when running data_extraction.py
|           ├── data_raw/          
|           |        └── annotations/      # contains raw annotations data
|           |        └── audio/            # contains raw audio data
|           ├── readme_images/             # contains images used in the readme
|           ├── pickles/                   # contains the model and label encoder files
|           ├── temp_prediction/           # used to store audio files used for predictions from the command line
|           ├── streamlit/                 
|           |        └── images/           # images for the streamlit app background and the chromagram image
|           ├── notebooks/
|           ├── app.py
|           ├── data_extraction.py
|           ├── modeling.py
|           ├── settings.py
|           ├── streamlit.py
|           └── requirements.txt
```
