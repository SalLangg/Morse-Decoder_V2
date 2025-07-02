from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
from typing import Union, Tuple, Dict
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse

import gdown
import patoolib
import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from src_decoder.configs import config
from src_decoder.data.dataset import MosreDataset, data_to_inference, data_to_training
from src_decoder.models.MorseNet import MorseNet
import re


class TreaningStartup():
    def __init__(self):
        self.conf = None
        self.model = None
        self.dataset = None
        self.audio_path = None

        self.extract_dir = 'src_data'
        self.extracted_folder = None
        self.full_path = None
        self.test_csv_path = None
        self.train_csv_path = None


def get_extracted(extract_dir):
    extracted_folder = None
    for item in os.listdir(extract_dir):
        item_path = os.path.join(extract_dir, item)
        if os.path.isdir(item_path):
            extracted_folder = item
            break

    train_startup.extracted_folder = Path(extracted_folder)
    train_startup.full_path = Path.joinpath(extract_dir, extracted_folder)
    train_startup.test_csv_path = Path.joinpath(extract_dir, extracted_folder,"test.csv")
    train_startup.train_csv_path = Path.joinpath(extract_dir, extracted_folder,"train.csv")

    if not extracted_folder:
        raise HTTPException(
            status_code=400,
            detail="No folder found in extracted archive"
            )

router_training = APIRouter(
    prefix="/ML_training",
    tags=["ML_training"],
    responses={404: {"description": "Not found"}}
)

train_startup = TreaningStartup()


@router_training.post("/init", summary='Initialization dase models')
async def init():
    """
    Initialization at server startup.

    Heavy initialization MosreDataset only on start. When requested, 
    only a light init a data.
    """
    train_startup.conf = config.load_config(base=False)
    print('MorseNet - initializing model')
    train_startup.model = MorseNet(config=train_startup.conf)
    train_startup.model.load()
    train_startup.dataset = MosreDataset(w_type='inference', 
                           config=train_startup.conf, 
                           is_validation=False)
    
    total_params = sum(p.numel() for p in train_startup.model.parameters() if p.requires_grad)    
    print(f'\nMorseNet - Number of parameters to be trained: {total_params:,}')

    extract_dir = train_startup.extract_dir
    get_extracted(extract_dir=extract_dir)
    print('File patchs were initialized')


@router_training.put("/loading_data", summary='Data loading with google drive by link')
async def load_data(url: str = 'https://drive.google.com/file/d/1JuWfEGOHMiV6n934aBWiHocZhtRmyVG-/view?usp=sharing'):
    """Data with google drive is base dataset to model treaning with a base structure:
        ├── file_name/
            ├── file_name - data to train
            ├── text.csv
            └── train.csv
    
    Args:
        url: str. Link to loading dataset. The default value is download the prepared data.
    """
    file_id = re.search(r'/d/([^/]+)', url).group(1)
    url = f'https://drive.google.com/uc?id={file_id}&export=download'
    output = 'src_data/morse_dataset.rar'
    gdown.download(url, output, quiet=False)

    extract_dir = train_startup.extract_dir
    patoolib.extract_archive(output, outdir=extract_dir)

    # ===== Find extracted folder =====
    try:
        get_extracted(extract_dir=extract_dir)

        patoolib.extract_archive(output, outdir='src_data')

        return JSONResponse(
            status_code=200,
            content={"message": "File extracted successfully"})
    except Exception as ex:
        raise HTTPException(
            status_code=404,
            detail=f"Extraction failed: {str(ex)}"
        )
    

@router_training.post("/load_{model_name}", summary='Load model')
async def load_model(model_name: str):
    """
    Loading model by name
    """
    train_startup.model.load(name=model_name)

    if train_startup.model is None:
        raise HTTPException(
            status_code=400,
            detail=f"MorseNet model is't loaded"
        )


@router_training.post("/fit")
async def fit():
    f_patch = train_startup.test_csv_path
    if f_patch is None:
        raise HTTPException(
            status_code=400,
            detail=f"File patchs not initialized. Use /init"
        )
    
    train_data = pd.read_csv(f_patch)

    dataset = dataset.setup_data(train_data)
    dataloader = data_to_training(dataset, config=train_startup.conf)
    
@router_training.post("/fit_inference")
async def fit_inference(audio_path: str):
    dataset = dataset.setup_data(audio_path)
    dataloader = data_to_inference(dataset, config=train_startup.conf)
