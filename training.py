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
import subprocess

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

        self.extract_dir = "src_data"
        self.extracted_folder = None
        self.full_path = None
        self.test_csv_path = None
        self.train_csv_path = None


@asynccontextmanager
async def training_lifespan(app: FastAPI):
    subprocess.Popen(
        ["mlflow", "ui", "--backend-store-uri", "mlruns", "--port", "5001"]
        # ["mlflow", "ui", "--backend-store-uri", "--port", "5001"]
        )
    print("MLflow UI started at http://127.0.0.1:5001")

    yield

def get_extracted(extract_dir):
    extracted_folder = None
    for item in Path(extract_dir).iterdir():
        if 'dataset' in item.name:
            extracted_folder = item.name
            break
    
    extract_dir = Path(extract_dir)
    train_startup.extracted_folder = Path(extracted_folder)
    train_startup.full_path = Path.joinpath(extract_dir, extracted_folder)
    train_startup.audio_path = Path.joinpath(extract_dir, extracted_folder, extracted_folder)
    train_startup.test_csv_path = Path.joinpath(extract_dir, extracted_folder,"test.csv")
    train_startup.train_csv_path = Path.joinpath(extract_dir, extracted_folder,"train.csv")
    print(train_startup.full_path, train_startup.audio_path,train_startup.test_csv_path,train_startup.train_csv_path)

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


@router_training.post("/init", summary="Initialization dase models")
async def init(name):
    """
    Methods initialization at server.

    Arg:
        name - name treaning model. It will be used when saving the model.
    """
    train_startup.conf = config.load_config(base=False)
    print("MorseNet to train - initializing model")
    train_startup.model = MorseNet(config=train_startup.conf, name_to_save=name)
    train_startup.model.load()
    # train_startup.dataset = MosreDataset(w_type="training", 
    #                        config=train_startup.conf, 
    #                        is_validation=False)
    
    total_params = sum(p.numel() for p in train_startup.model.parameters() if p.requires_grad)    
    print(f"\nMorseNet to train  - Number of parameters to be trained: {total_params:,}")

    extract_dir = train_startup.extract_dir
    get_extracted(extract_dir=extract_dir)
    return JSONResponse(
        status_code=200,
        content="Initialization was completed successfully"
    )


@router_training.put("/loading_data", summary="Data loading with google drive by link")
async def load_data(url: str = "https://drive.google.com/file/d/1JuWfEGOHMiV6n934aBWiHocZhtRmyVG-/view?usp=sharing"):
    """Data with google drive is base dataset to model treaning with a base structure:
        ├── file_name/
            ├── file_name - data to train
            ├── text.csv
            └── train.csv
    
    Args:
        url: str. Link to loading dataset. The default value is download the prepared data.
    """
    file_id = re.search(r"/d/([^/]+)", url).group(1)
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    output = "src_data/morse_dataset.rar"
    gdown.download(url, output, quiet=False)

    extract_dir = train_startup.extract_dir
    patoolib.extract_archive(output, outdir=extract_dir)

    # ===== Find extracted folder =====
    try:
        get_extracted(extract_dir=extract_dir)

        patoolib.extract_archive(output, outdir="src_data")

        return JSONResponse(
            status_code=200,
            content={"message": "File extracted successfully"})
    except Exception as ex:
        raise HTTPException(
            status_code=404,
            detail=f"Extraction failed: {str(ex)}"
        )
    

@router_training.post("/load_{model_name}", summary="Load model")
async def load_model(model_name: str):
    """
    Loading model by name
    """
    if train_startup.model is None:
        raise HTTPException(
            status_code=400,
            detail=f"MorseNet model is't loaded"
        )
    
    train_startup.model.load(name=model_name)


# @router_training.get("/start_mlfow")
# def start_mlflow_ui():
#     try:
#         subprocess.Popen(
#             ["mlflow", "ui", "--backend-store-uri", "mlruns", "--port", "5001"]
#             )
#         return JSONResponse(
#             status_code=200,
#             content={"message": "MLflow UI started at http://localhost:5001"})
#     except Exception as ex:
#         raise HTTPException(
#             status_code=400,
#             detail=f"MLflow starting error: {str(ex)}"
#         )
    

@router_training.post("/fit")
async def fit():
    f_patch = train_startup.train_csv_path
    if f_patch is None:
        raise HTTPException(
            status_code=400,
            detail=f"File patchs not initialized. Use /init"
        )
    
    train_df = pd.read_csv(f_patch)

    # dataset = train_startup.dataset.setup_data(train_df, train_startup.audio_path)
    train_loader, val_loader = data_to_training(df=train_df,
                                           config=train_startup.conf)
    
    model = train_startup.model
    model.fit(thain_data=train_loader, val_data=val_loader)
    # print(len(train_loader), next(iter(train_loader)))

@router_training.post("/fit_inference")
async def fit_inference(audio_path: str):
    model_name = train_startup.model.name
    if model_name is None:
        raise HTTPException(
            status_code=400,
            detail=f"Modet is't load to treaning inference. Use /init -> /start_mlfow -> /fit"
        )
    dataset = dataset.setup_data(audio_path)
    dataloader = data_to_inference(dataset, config=train_startup.conf)
