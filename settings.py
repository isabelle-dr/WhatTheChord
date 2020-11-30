import os

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

PRED_DIR = os.path.join(ROOT_DIR, 'prediction')

MODEL_H5 = os.path.join(ROOT_DIR, 'final_model.h5')
LE = os.path.join(ROOT_DIR, 'labelencoder.sav')

OUT_DIR = os.path.join(ROOT_DIR, 'streamlit/')
IMAGE_DIR = os.path.join(OUT_DIR, 'images')

WAVE_UPLOADED_FILE = os.path.join(OUT_DIR, "uploaded.wav") 
WAVE_RECORDED_FILE = os.path.join(OUT_DIR, "recorded.wav") 
CHROMAGRAM_FILE = os.path.join(OUT_DIR, "chromagram.jpg") 


# Audio configurations
INPUT_DEVICE = 0
MAX_INPUT_CHANNELS = 1  # Max input channels
DEFAULT_SAMPLE_RATE = 44100   # Default sample rate of microphone or recording device
DURATION = 3   # 3 seconds
CHUNK_SIZE = 1024
