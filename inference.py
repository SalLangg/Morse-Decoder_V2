from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset

from src_decoder.configs import config
from src_decoder.data.dataset import MosreDataset, data_to_inference, data_to_training
from src_decoder.models.MorseNet import MorseNet

app = FastAPI()
conf = None
model = None
dataset = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialization at server startup.

    Heavy initialization MosreDataset only on start. When requested, 
    only a light init a data.
    """
    global model, dataset, conf
    conf = config.load_config(base=True)
    model = MorseNet(config=conf)
    model.load()
    model.eval()
    dataset = MosreDataset(w_type='inference', 
                           config=conf, 
                           is_validation=False)
    yield

@app.post("/predict")
async def predict(audio_path: str):
    dataset = dataset.setup_data(audio_path)
    dataloader = data_to_inference(dataset, config=conf)

@app.post("/fit")
async def fir(audio_path: pd.DataFrame):
    dataset = dataset.setup_data(audio_path)
    dataloader = data_to_training(dataset, config=conf)
    