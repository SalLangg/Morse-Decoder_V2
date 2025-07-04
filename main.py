from fastapi import FastAPI, Form, File, UploadFile
from inference import router_inference, inference_lifespan
from training import router_training, training_lifespan
from typing import Annotated
from pathlib import Path as pt
from fastapi.responses import JSONResponse
from fastapi import APIRouter, HTTPException
from contextlib import asynccontextmanager, AsyncExitStack

import gdown
import patoolib

@asynccontextmanager
async def _lifespan_manager(app: FastAPI):
    async with AsyncExitStack() as exit_stack:
        if inference_lifespan:
            await exit_stack.enter_async_context(inference_lifespan(app))
        
        if training_lifespan:
            await exit_stack.enter_async_context(training_lifespan(app))

        yield

app = FastAPI(lifespan=_lifespan_manager)

app.include_router(router_inference)
app.include_router(router_training)
# app.include_router(routers_req)

@app.put('/base_data',summary='Loading a base dataset.')
async def load_base_data():
    """
    Load a base dataset to treaning
    Warning! File size is 6.45G
    """
    file_id = '1JuWfEGOHMiV6n934aBWiHocZhtRmyVG-'
    url = f'https://drive.google.com/uc?id={file_id}&export=download'
    output = 'src_data/morse_dataset.rar'
    gdown.download(url, output, quiet=False)
    try:
        patoolib.extract_archive(output, outdir='src_data')
        output.unlink()
        return JSONResponse(
            status_code=200,
            content={'message': 'File extracted successfully'}
        )
    except Exception as ex:
        raise HTTPException(
            status_code=404,
            detail=f'Extraction failed: {str(ex)}'
        )
