# imports

import pandas as pd
import numpy as np
import librosa
import librosa.display
import os
import jams
import matplotlib.pyplot as plt
from settings import AUDIO_DIR, ANNOTATIONS_DIR, LABELS_DIR, IMAGES_DIR

audio_files = sorted(os.listdir(AUDIO_DIR))
annotation_files = sorted(os.listdir(ANNOTATIONS_DIR))

'''
Extract chord labels from annotations_files 
The annotaitons of this dataset come in the form of midi files. 
The chords can be found in two different places: instructed and played. The chords instructed have a sentence in the 'annotation_rules' section.
This is how they are differenciated here.
Steps:
1- Create empty dataframes chord_instructed and chord_played
2- Loop through the annotation file of the 'comp' tracks, filling the dataframes with one row = one chord and the relevant information (starting time, duration). The ID of the dataframe is the name of the track with a number 
3- Merge dataframes on the id
4- Filter dataframe: 
Keep only rows where the instructed chord is included in the played chord (discard when the players went too far from instructed chord).
Discard very underrepresented classes (seventh)
''' 

print('Extracting chord labels...')
chords_instructed = pd.DataFrame()
chords_played = pd.DataFrame()

for file in annotation_files:
    # loop only over the files that have 'comp' in their name (discard solos)
    if 'comp' in str(file):
        # load the file
        jam = jams.load(ANNOTATIONS_DIR + file)

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

# filter dataframe
# 1.keep only rows where chord_played contains chord instructed (discrad rows where player did something too different)
#2. discard all seventh
chords_all.insert(loc = 0, column = "to_keep", value = False, allow_duplicates = True)

for index, row in chords_all.iterrows():
    if (str(row['chord_instructed']) in str(row['chord_played'])) and ('7' not in str(row.chord_instructed)):
        chords_all.loc[index, "to_keep"] = True
        
chords_clean = chords_all[chords_all.to_keep == True]
chords_clean = chords_clean.reset_index().drop('index', axis=1)
chords_clean = chords_clean.dropna()

# export dataframe
chords_clean.to_csv(LABELS_DIR + "labels.csv")

print('Chord labels extracted')


'''
Create images for each selected slice of audio
For each audio file in the list of selected tracks
    Extract corresponding information form dataframe
    For each line in the dataframe
        Create a chromagram for the corresponding start and duration, and store it in the images folder
'''
print('Extracting images...')
tracks_list = list(chords_clean.track_name.unique())
for file in audio_files:
    
    # if the audio track is selected for the model
    if file[:-12] in tracks_list:
        # get chords data
        chords_data = chords_clean[chords_clean.track_name == file[:-12]]
        
        # loop through lines in dataframe
        for i in range(len(chords_data.time)):
            name = list(chords_data.id)[i]
            data, sr = librosa.load(AUDIO_DIR + file, offset=list(chords_data.time)[i], duration=list(chords_data.duration)[i])

            # create chromagram
            chromagram = librosa.feature.chroma_cens(data, sr=sr, fmin=75)
            chromagram_mag = librosa.magphase(chromagram)[0]**4

            # save image
            fig = plt.figure(figsize=(2, 2))
            librosa.display.specshow(chromagram_mag, sr=sr, cmap='gray')
            fig.savefig(IMAGES_DIR + name)
            plt.close()

print('Images extracted')