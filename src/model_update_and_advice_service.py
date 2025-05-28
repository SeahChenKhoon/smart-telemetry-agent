from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import json

import yaml
import os

from src.cls_env import cls_Env 
from src.cls_LLM import cls_LLM
from src.cls_telemetric import cls_SystemDiagnostics
import src.util as util

cls_env = cls_Env()
cls_llm = cls_LLM(cls_env)

config = util.load_config(cls_env.clould_config_path)


app = FastAPI()

class DiagnosticInput(BaseModel):
    message: str

class DiagnosticsItem(BaseModel):
    item: int

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
    rules_path = config["output"]["rules_path"]

    if not os.path.exists(rules_path):
        return {"error": f"Model not found at: {rules_path}"}

    return FileResponse(
        path=rules_path,
        filename=os.path.basename(rules_path),
        media_type="application/octet-stream"
    )
    
@app.post("/escalate_issue")
def escalate_issue(diagnostic: DiagnosticInput):
    rules_path = config["output"]["clould_path"]

    with open(rules_path, "r") as f:
        rules = json.load(f)

    llm_parameter = {
        "insert telemetry string here": diagnostic,
        "insert rules JSON here": rules
    }
    output = cls_llm.execute_llm_prompt(config["llm_prompt"], llm_parameter)
    print(output)
    return {
        "response": output
    }

@app.post("/generate_advice")
def generate_advice():
    return {
        "advice": "No specific advice. Please provide a more detailed issue."
    }

        
