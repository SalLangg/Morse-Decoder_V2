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

LOAD_AUDIO_DIR = Path('src_data/loaded_audio')


class TreaningStartup():
    def __init__(self):
        self.conf = None
        self.model = None
        self.dataset = None
        self.audio_path = None

test_startup = TreaningStartup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialization at server startup.

    Heavy initialization MosreDataset only on start. 
    When requested, only a light init a data.
    """
    test_startup.conf = config.load_config(base=True)
    print('MorseNet - initializing model')
    test_startup.model = MorseNet(config=test_startup.conf)
    test_startup.model.load()
    test_startup.model.eval()
    test_startup.dataset = MosreDataset(w_type='inference', 
                           config=test_startup.conf, 
                           is_validation=False)
    
    total_params = sum(p.numel() for p in test_startup.model.parameters() if p.requires_grad)    
    print(f'\nMorseNet - Number of parameters to be trained: {total_params:,}')
    yield


@router_inference.post("/load", summary="Load audiofile")
async def load_audio(file: UploadFile = File(...)):
    """
    Upload a file for training the models to server
    """
    try:
        audio_path = Path.joinpath(LOAD_AUDIO_DIR, file.filename)
        test_startup.audio_path = audio_path
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


@router_inference.delete("/delete_all", summary="Delete all files")
async def delet_file():
    """
    Delete all loaded files
    """
    print(LOAD_AUDIO_DIR)
    try:
        files_to_del = LOAD_AUDIO_DIR.iterdir()
        print(files_to_del)
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
    audio_path = test_startup.audio_path
    
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
        dataloader = data_to_inference(data=audio_path, 
                                       dataset=test_startup.dataset, 
                                       config=test_startup.conf)
        return test_startup.model.predict(dataloader)
    
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(ex)}"
        )
    
