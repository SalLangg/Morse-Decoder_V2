from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from inference import router_inference, lifespan
from training import router_training
from typing import Annotated
from pathlib import Path as pt

app = FastAPI(lifespan=lifespan)

app.include_router(router_inference)
app.include_router(router_training)
# app.include_router(routers_req)