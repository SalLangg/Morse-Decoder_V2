from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset

from src_decoder.configs import config
from src_decoder.data.dataset import MosreDataset, data_to_inference, data_to_training
from src_decoder.models.MorseNet import MorseNet

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List

router_inference = APIRouter(
    prefix="/ML_inference",
    tags=["ML_inference"],
    responses={404: {"description": "Not found"}}
)

LOAD_AUDIO_DIR = f'src_data/loaded_audio'

conf = None
model = None
dataset = None
audio_path = None

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


@router_inference.post("/load", summary="Load audiofile")
async def load_audio(file: UploadFile = File(...)):
    """
    Upload a file for training the models
    """
    global audio_path
    try:
        audio_path = Path.joinpath(LOAD_AUDIO_DIR, file.filename)

        if audio_path.exists():
            audio_path.unlink()
        
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return JSONResponse(
            status_code=200,
            content={"message": "File loaded successfully", 
                     "file name": file.filename, 
                     "path": str(audio_path)}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading file: {str(e)}"
        )


@router_inference.delete("/delete_{file_name}", summary="Delete file")
async def delet_file(file_name: str):
    """
    Delete file by name
    """
    try:
        file_location = Path.joinpath(LOAD_AUDIO_DIR, file_name)
        file_location.unlink()
        
        return JSONResponse(
            status_code=200,
            content={"message": "File deleted successfully", 
                     "file name": file_name, 
                     "path": str(file_location)}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting file: {str(e)}"
        )
    

@router_inference.delete("/delete_all", summary="Delete all files")
async def delet_file():
    """
    Delete all files
    """
    try:
        files_to_del = Path.joinpath(LOAD_AUDIO_DIR).iterdir()
        file_names = []
        for file in files_to_del:
            file_names.append(file.name)
            file.unlink()
        
        return JSONResponse(
            status_code=200,
            content={"message": "Files deleted successfully", 
                     "files name": file_names}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting files: {str(e)}"
        )
    

@router_inference.post("/predict")
async def predict():
    if audio_path is None:
        raise HTTPException(
            status_code=400,
            detail="Audio file isn't loaded"
        )
    
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found at path: {audio_path}"
        )
    
    try:
        dataset = dataset.setup_data(audio_path)
        dataloader = data_to_inference(dataset, config=conf)

        return model.predict(dataloader)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
