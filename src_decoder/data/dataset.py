import torch
import torchaudio
from torchaudio import transforms
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src_decoder.configs.config import Config
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Union, Literal
from functools import partial


class MosreDataset(Dataset):
    def __init__(self,
                 w_type: Literal['inference', 'training'],
                 config: Config = None,
                 is_validation=False):
        """
        Dataset by Morse decoder.
        
        Args:
            data: Current dataframe for training ot testin/ Current audio file patch (from inference)
            audio_dir
            char_to_int
            transforms=None: Transformaions from spectogtams

        Returns:
            if data it's a dataframe
                Tuple of (augmented_spectrogram, target, target_len, message)
            else
                Tuple of (augmented_spectrogram, None, None, None)
        """

        # ===== Init paramemers =====
        self.config = config
        self.data = None
        # self.data = data
        self.is_validation = is_validation
        self.char_to_int = self.config.char_to_int
        self.config = self.config.data

        self._transforms = [
                transforms.MelSpectrogram(sample_rate=self.config.sample_rate, 
                                          n_fft=self.config.n_fft, 
                                          hop_length=self.config.hop_length, 
                                          n_mels=self.config.n_mels),
                transforms.AmplitudeToDB(top_db=self.config.top_db)
            ]
        
        # ===== Init transformations for spectrogram =====
        if w_type == 'training':
            self._transforms = nn.Sequential(
                *self._transforms,
                transforms.FrequencyMasking(freq_mask_param=self.config.freq_mask),
                transforms.TimeMasking(time_mask_param=self.config.time_mask),
                )

            self.audio_paths = self.config.audio_dir
            self.messeges = self.data.message.values
        else:
            self._transforms = nn.Sequential(*self._transforms)

    def setup_data(self, data: Union[str, pd.DataFrame]):
        self.data = data
        if isinstance(self.data, pd.DataFrame):
            self.messeges = self.data.message.values
    
    def __len__(self):
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        else: 
            return 1
    
    def __getitem__(self, index):
        if self.data is not None:
            try:
                if isinstance(self.data, pd.DataFrame):
                    audio_file = self.audio_paths / self.data.id.values[index]
                else:
                    audio_file = self.data

                waveform, sample_rate = torchaudio.load(audio_file)
                augmented_spectrogram = self._transforms(waveform)
                if self.is_validation:
                    message = self.messeges[index]
                    target = torch.tensor([self.char_to_int[char] for char in message], 
                                        dtype=torch.long)
                    target_len = torch.tensor(len(target), dtype=torch.long)
                    return augmented_spectrogram, target, target_len, message
                else:
                    return augmented_spectrogram, None, None, None
            except Exception as ex:
                print(str(ex))


def __my_collate(batch,padding_value):
    """Gereration butches of the same length"""
    spectrograms = [item[0].squeeze(0) for item in batch]

    # ===== Padding sequences of the spectrogram by max length =====
    spectrograms_permuted = [s.permute(1, 0) for s in spectrograms]
    spectrograms_padded = nn.utils.rnn.pad_sequence(spectrograms_permuted, batch_first=True, padding_value=0.0)
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1).unsqueeze(1)

    if batch[0][3] is not None:
        # ===== Padding sequences of the messege by max length =====
        target = torch.nn.utils.rnn.pad_sequence(
                                                [item[1] for item in batch], 
                                                batch_first=True, 
                                                padding_value=padding_value)
        label_len = torch.stack([item[2] for item in batch])
        msg = [item[3] for item in batch]
        return [spectrograms_padded, target, label_len, msg]
    else: 
        return spectrograms_padded, None, None, None
    

def data_to_inference(data:Union[str, pd.DataFrame],dataset, config: Config) -> DataLoader:
    """
    Converts data into inference DataLoaders.
        
        Args:
            df: The original DataFrame with the data
            data: Dataset for processing
            config: Configuration with parameters
            
        Returns:
            inference dataloared
    """
    dataset.setup_data(data)
    callate = partial(__my_collate,padding_value=config.blank_ind)
    dataloared = DataLoader(dataset, 
                            # batch_size=config.data.batch_size, 
                            batch_size=1, 
                            shuffle = False, 
                            collate_fn=callate,
                            num_workers=0)
    return dataloared 


def data_to_training(df: pd.DataFrame, data: MosreDataset, config: Config):
    """
    Converts data into training and validation DataLoaders.
        
        Args:
            df: The original DataFrame with the data
            data: Dataset for processing
            config: Configuration with parameters
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
            
        Raises (Optional):
            ValueError: If the configuration parameters are incorrect
    """
    val_size = config.data.val_size
    batch_size = config.data.batch_size
    seed  = config.data.seed

    train_dataframe, val_dataframe = train_test_split(data, 
                                                        test_size=val_size, 
                                                        random_state=seed)
    
    callate = partial(__my_collate, config.data.blank_char)
    t_dataloader = DataLoader(train_dataframe, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            collate_fn=callate, 
                            drop_last=True)

    val_dataloader = DataLoader(val_dataframe, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        collate_fn=callate, 
                        drop_last=True)
    
    return t_dataloader, val_dataloader


