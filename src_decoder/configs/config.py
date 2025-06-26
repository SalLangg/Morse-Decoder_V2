from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Literal, Dict
from pathlib import Path
import yaml


class DataConfig(BaseModel):
    sample_rate: PositiveInt = 8000
    n_mels: PositiveInt = 128
    n_fft: PositiveInt = 400
    hop_length: PositiveInt = 180
    top_db: PositiveInt = 80

    freq_mask: PositiveInt = 15
    time_mask: PositiveInt = 20


    audio_dir: str = 'morse_dataset/morse_dataset'
    audio_save_dir: str = 'morse_dataset/morse_dataset'
    train_csv: str = 'morse_dataset/train.csv'
    test_csv: str = 'morse_dataset/test.csv'

    seed: PositiveInt = 42
    val_size: PositiveFloat = 0.15
    batch_size: PositiveInt = 64

    blank_char: str = '_'    


class ModelConfig(BaseModel):
    num_classes: PositiveInt = 45
    epochs: PositiveInt = 70

    lr: PositiveFloat = 0.002
    weight_decay: PositiveFloat = 0.00001
    early_stopping_patience: PositiveInt = 5
    lr_patience: PositiveInt = 3
    lr_factor: PositiveFloat = 0.5 

    first_fe_count: PositiveInt = 16
    second_fe_count: PositiveInt = 32
    third_fe_count: PositiveInt = 32
    quad_fe_count: PositiveInt = 32
    padding: Literal['same', 'valid'] = 'same'
    maxpool_kernel: PositiveInt = 2
    kernel_size: PositiveInt = 3
    neuron_count: PositiveInt = 128
    gru_hidden: PositiveInt = 256
    dropout: PositiveFloat = 0.3


class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()

    blank_char: str = '_'
    morsealph: str = ' АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ1234567890#'
    vocab_list: str = morsealph + blank_char
    num_classes: PositiveInt = len(vocab_list)
    int_to_char: dict = dict(enumerate(vocab_list))
    char_to_int: dict = {char:enum for enum, char in int_to_char .items()}
    blaknk_ind: PositiveInt = char_to_int[blank_char]


def load_config(config_path: str = 'config.yaml', base=False) -> Config:
    '''Load config
    
    Parameters: 
        config_path: srt -> 
        base: bool -> use/unuse yaml values
    '''
    if base:
        return Config()
    else:
        with open(Path(__file__).parent / config_path) as f:
            raw_config = yaml.safe_load(f)
        return Config(**raw_config)
