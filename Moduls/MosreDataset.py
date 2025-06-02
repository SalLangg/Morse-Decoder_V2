import numpy as np
import pandas as pd
import os
import random
from pathlib import Path as pt
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset

class MosreDataset(Dataset):
    def __init__(self, data_patch, train=True, transforms=None):
        self.is_train = train
        self.data_path = data_patch
        self.audio_paths = self.data_path / 'morse_dataset'
        self.transforms = transforms
        
        if self.is_train:
            self.data =  pd.read_csv(pt.joinpath(self.data_path,'train.csv'))
            self.messeges = self.data.message.values
        else:
            self.data =  pd.read_csv(pt.joinpath(self.data_path,'test.csv'))

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        #Получение аугментрованых спектрограмм
        audio_file = self.audio_paths / self.data.id.values[index]
        print(audio_file)
        waveform = self.change_time(audio_file)
        augmented_spectrogram = self.transforms(waveform)
        
        #Получение one-hot векторов 
        if self.is_train:
            return augmented_spectrogram[index], self.messeges[index]
        else:
            return augmented_spectrogram[index]
        
    def change_time(audio_file, max_len = 384000):
        waveform, sample_rate = torchaudio.load(audio_file)
        cahanal, sig_len = waveform.shape

        if sig_len < max_len:
            pad_len = torch.zeros(max_len - sig_len).unsqueeze(0)
            waveform = torch.cat([waveform, pad_len], dim=1)

        return waveform