import os

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# directories for modeling
AUDIO_DIR = os.path.join(ROOT_DIR, 'data_raw/audio/')
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, 'data_raw/annotations/')
LABELS_DIR = os.path.join(ROOT_DIR, 'data_prepared/')
IMAGES_DIR = os.path.join(ROOT_DIR, 'data_prepared/images/')
MODEL_H5 = os.path.join(ROOT_DIR, 'pickles/final_model.h5')
LE = os.path.join(ROOT_DIR, 'pickles/labelencoder.sav')

# used for predictions from command line
PRED_DIR = os.path.join(ROOT_DIR, 'temp_prediction/')

# used for streamlit app
OUT_DIR = os.path.join(ROOT_DIR, 'streamlit/')
OUT_IMAGE_DIR = os.path.join(OUT_DIR, 'images/')
WAVE_UPLOADED_FILE = os.path.join(OUT_DIR, "uploaded.wav") 
WAVE_RECORDED_FILE = os.path.join(OUT_DIR, "recorded.wav") 
CHROMAGRAM_FILE = os.path.join(OUT_DIR, "chromagram.jpg") 
