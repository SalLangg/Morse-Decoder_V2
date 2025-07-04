from pydantic import BaseModel, PositiveInt, PositiveFloat
from typing import Literal, Dict
from pathlib import Path
import mlflow
import yaml


class DataConfig(BaseModel):
    sample_rate: PositiveInt = 8000                     # Частота дискретизации
    n_mels: PositiveInt = 128                           # Количество Mel-фильтров
    n_fft: PositiveInt = 400                            # Размер окна FFT
    hop_length: PositiveInt = 180                       # Шаг между FFT окнами
    top_db: PositiveInt = 80                            # Максимальная громкость в dB

    freq_mask: PositiveInt = 15                         # FrequencyMasking спектрограмм
    time_mask: PositiveInt = 20                         # TimeMasking спектрограмм

    seed: PositiveInt = 42
    val_size: PositiveFloat = 0.15                      # Размер валидационной выборки
    batch_size: PositiveInt = 64

    blank_char: str = "_"                               # Символ CTC blank  


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
    padding: Literal["same", "valid"] = "same"
    maxpool_kernel: PositiveInt = 2
    kernel_size: PositiveInt = 3
    neuron_count: PositiveInt = 128
    gru_hidden: PositiveInt = 256
    dropout: PositiveFloat = 0.3


class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()

    blank_char: str = "_"
    morsealph: str = " АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ1234567890#"
    vocab_list: str = sorted(morsealph) + [blank_char]
    num_classes: PositiveInt = len(vocab_list)
    int_to_char: dict = dict(enumerate(vocab_list))
    char_to_int: dict = {char:enum for enum, char in int_to_char .items()}
    blank_ind: PositiveInt = char_to_int[blank_char]


def load_config(config_path: str = "config.yaml", base=False) -> Config:
    """Load config
    
    Parameters: 
        config_path: srt
        base: bool -> use/unuse yaml values
    """
    if base:
        return Config()
    else:
        with open(Path(__file__).parent / config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        return Config(**raw_config)


def setup_mlflow(tracking_uri: str = 'http://127.0.0.1:5001'):
    mlflow.set_tracking_uri(f"{tracking_uri}")
    return tracking_uri