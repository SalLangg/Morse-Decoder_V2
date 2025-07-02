from fastapi import FastAPI, Form, File, UploadFile
from inference import router_inference, lifespan
from training import router_training
from typing import Annotated
from pathlib import Path as pt
from fastapi.responses import JSONResponse
from fastapi import APIRouter, HTTPException

import gdown
import patoolib

app = FastAPI(lifespan=lifespan)

app.include_router(router_inference)
app.include_router(router_training)
# app.include_router(routers_req)

@app.put('/base_data',summary="Loading a base dataset.")
async def load_base_data():
    file_id = '1JuWfEGOHMiV6n934aBWiHocZhtRmyVG-'
    url = f'https://drive.google.com/uc?id={file_id}&export=download'
    output = 'src_data/morse_dataset.rar'
    gdown.download(url, output, quiet=False)
    try:
        patoolib.extract_archive(output, outdir='src_data')
        return JSONResponse(
            status_code=200,
            content={"message": "File extracted successfully"}
        )
    except Exception as ex:
        raise HTTPException(
            status_code=404,
            detail=f"Extraction failed: {str(ex)}"
        )
