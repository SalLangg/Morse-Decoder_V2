from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import APIRouter
from typing import Union, Tuple, Dict
from fastapi import APIRouter, HTTPException, Form, File, UploadFile

import gdown
import zipfile

import torch
import torchaudio
from torch.utils.data import Dataset

from src_decoder.configs import config
from src_decoder.data.dataset import MosreDataset, data_to_inference, data_to_training
from src_decoder.models.MorseNet import MorseNet

router_training = APIRouter(
    prefix="/ML_training",
    tags=["ML_training"],
    responses={404: {"description": "Not found"}}
)

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
    conf = config.load_config(base=False)
    model = MorseNet(config=conf)
    dataset = MosreDataset(w_type='training', 
                           config=conf, 
                           is_validation=False)
    yield

@router_training.post("/load_{model_name}", summary='Load model')
async def load(model_name: str):
    """
    Loading model by name
    """
    model.load(name=model_name)

@router_training.put("/loading_data", summary='Loading data files')
async def load(link: str = 'https://drive.google.com/file/d/1JuWfEGOHMiV6n934aBWiHocZhtRmyVG-/view?usp=sharing'):
    """Files loading with google drive by link
    
    Args:
        link: str. Link to loading dataset. The default value is download the prepared data.
    """
    try:
        url = link
        output_path = 'src_data/data_to_treaning'
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        
        zip_location = 'src_data/data_to_treaning/morse_dataset.zip'
    
        with zipfile.ZipFile(zip_location, 'r') as zip_ref:
            zip_ref.extractall()

        zip_location.unlink('src_data/data_to_treaning/morse_dataset.zip')
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading audiofile: {str(ex)}"
        )

@router_training.post("/fit_inferencet")
async def fit_inference(audio_path: str):
    dataset = dataset.setup_data(audio_path)
    dataloader = data_to_inference(dataset, config=conf)

@router_training.post("/fit")
async def fir(audio_path):
    if not isinstance(audio_path, pd.DataFrame):
        raise HTTPException(
            status_code=500,
            detail=f"Error loading file: {str(e)}"
        )
    dataset = dataset.setup_data(audio_path)
    dataloader = data_to_training(dataset, config=conf)
    
    