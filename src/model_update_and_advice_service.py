from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import yaml
import os

app = FastAPI()

class DiagnosticInput(BaseModel):
    message: str

def load_config()-> Dict[str, Any]:
    """
    Load configuration settings from a YAML file.

    Reads the 'config.yml' file from the current directory
    and returns its contents as a dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing configuration keys and values.
    """    
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config

@app.post("/generate_advice")
def generate_advice(input: DiagnosticInput):
    msg = input.message.lower()
    if "thermal" in msg or "90°" in msg:
        return {
            "advice": "Your system appears to be running hot. Check for blocked vents and reduce CPU load."
        }
    return {
        "advice": "No specific advice. Please provide a more detailed issue."
    }


@app.post("/load_local_model")
def load_local_model():
    config = load_config()
    model_path = config["output"]["model_path"]

    if not os.path.exists(model_path):
        return {"error": f"Model not found at: {model_path}"}

    return FileResponse(
        path=model_path,
        filename=os.path.basename(model_path),
        media_type="application/octet-stream"
    )
    
