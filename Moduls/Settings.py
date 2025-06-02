import os
from pathlib import Path as pt

MAIN = pt(__file__).parent.parent

class Settings():
    DATASET_PATCH = MAIN / 'morse_dataset'
    AUDIO_FILES = DATASET_PATCH / 'morse_dataset'

    # Mel specs params
    MAX_TIME = 48
    SAMPLE_RATE = 8000
    N_MELS = 128
    N_FFT = 400
    HOP_LENGTH = 160

    # Augmengt params
    FREQ_MASK = 30
    TIME_MASK = 40
# print(MAIN)
