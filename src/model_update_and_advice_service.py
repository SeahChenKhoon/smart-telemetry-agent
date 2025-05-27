from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import yaml
import os

import src.util as util

env_variables = util.read_env()
config = util.load_config(env_variables["cloud_config_path"])
app = FastAPI()

class DiagnosticInput(BaseModel):
    message: str


@app.post("/load_local_model")
def load_local_model():
    model_path = config["output"]["model_path"]

    if not os.path.exists(model_path):
        return {"error": f"Model not found at: {model_path}"}

    return FileResponse(
        path=model_path,
        filename=os.path.basename(model_path),
        media_type="application/octet-stream"
    )
    
@app.post("/load_local_rules")
def load_local_rules():
    rules_path = config["output"]["model_rules"]

    if not os.path.exists(rules_path):
        return {"error": f"Model not found at: {rules_path}"}

    return FileResponse(
        path=rules_path,
        filename=os.path.basename(rules_path),
        media_type="application/octet-stream"
    )
    
