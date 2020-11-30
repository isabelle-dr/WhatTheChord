# imports

import pandas as pd
import numpy as np
import librosa
import librosa.display
import os
import jams

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt

import IPython.display as ipd
from IPython.display import Audio

# Define paths for audio, annotations, images, labels
path_audio = "audio/"
path_annotations = "annotation/"
path_images = "images/"
path_labels = "labels/"

audio_files = sorted(os.listdir(path_audio))
annotation_files = sorted(os.listdir(path_annotations))
track_names = [name[:-5] for name in annotation_files]

############################ extract chords from annotations_files #########################
print('Extracting annotations data...')
chords_instructed = pd.DataFrame()
chords_played = pd.DataFrame()

for file in annotation_files:
    # load the file
    jam = jams.load(path_annotations + file)

    # create chords_song dataframe
    start_list_instructed = []
    start_list_played = []
    chord_list_instructed = []
    chord_list_played = []
    duration_list_instructed = []
    duration_list_played = []
    id_list_played = []
    id_list_instructed = []
    
    for i in range(0, len(jam['annotations'])):   
        if jam['annotations'][int(i)]['namespace'] == 'chord':
            if jam['annotations'][int(i)]['annotation_metadata']['annotation_rules'] == '':
                for j in range(0, len(sorted(jam['annotations'][int(i)]['data']))):
                    line = list(jam['annotations'][int(i)]['data'][int(j)])[:-1]
                    start_list_instructed.append(line[0])
                    duration_list_instructed.append(line[1])
                    chord_list_instructed.append(line[2])
                    id_list_instructed.append(file[:-5] + '_' + str(j))
            else:
                for j in range(0, len(sorted(jam['annotations'][int(i)]['data']))):
                    line = list(jam['annotations'][int(i)]['data'][int(j)])[:-1]
                    start_list_played.append(line[0])
                    duration_list_played.append(line[1])
                    chord_list_played.append(line[2])
                    id_list_played.append(file[:-5] + '_' + str(j))

        chords_song_instructed = pd.DataFrame({'id_instructed': id_list_instructed, 'time_instructed':start_list_instructed, 'duration_instructed':duration_list_instructed, 'chord_instructed':chord_list_instructed})
        chords_song_played = pd.DataFrame({'id_played': id_list_played, 'time_played':start_list_played, 'duration_played':duration_list_played, 'chord_played':chord_list_played})
        chords_song_instructed['track_name'] = file[:-5]
        chords_song_played['track_name'] = file[:-5]

    chords_instructed = pd.concat([chords_instructed, chords_song_instructed])
    chords_played = pd.concat([chords_played, chords_song_played])
    
# merge and format dataframes
chords_instructed = chords_instructed.rename(columns={"id_instructed":"id"})
chords_played = chords_played.rename(columns={"id_played":"id"})
chords_all = chords_instructed.merge(chords_played, on='id')

chords_all = chords_all.drop(['time_played', 'duration_played', 'track_name_y'], axis=1)
chords_all = chords_all.rename(columns = {'time_instructed': 'time', 'duration_instructed': 'duration', 'track_name_x': 'track_name'})
chords_all = chords_all[['id', 'track_name', 'time', 'duration', 'chord_instructed', 'chord_played']]

# get composition only
chords_comp = chords_all[chords_all['track_name'].str.contains("comp")]

# keep only rows where chord_played contains chord instructed (discrad rows where player did something too different)
chords_comp["to_keep"] = False

for index, row in chords_comp.iterrows():
    if str(row['chord_instructed']) in str(row['chord_played']):
        chords_comp.loc[index, "to_keep"] = True
chords_comp_clean = chords_comp[chords_comp.to_keep == True]
chords_comp_clean = chords_comp_clean.reset_index().drop('index', axis=1)
chords_comp_clean=chords_comp_clean.dropna()

# export dataframe
chords_comp_clean.to_csv("labels/chords.csv")

print('Annotations extracted')

##################### extract images from audio_files ##################################
print('Extracting images...')
for file in audio_files:
    chords_data = chords_comp_clean[chords_comp_clean.track_name == file[:-12]]
    
    for i in range(len(chords_data.time)):
        name = list(chords_data.id)[i]
        data, sr = librosa.load(path_audio + file, offset=list(chords_data.time)[i], duration=list(chords_data.duration)[i])

        # create chromagram
        chromagram = librosa.feature.chroma_cens(data, sr=sr)
        chromagram_mag = librosa.magphase(chromagram)[0]**4

        # save image
        fig = plt.figure(figsize=(2, 2))
        librosa.display.specshow(chromagram_mag, sr=sr, cmap='gray')
        fig.savefig(path_images + name)
        plt.close()
        
print('images extracted')
