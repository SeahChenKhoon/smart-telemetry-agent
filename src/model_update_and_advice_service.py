from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import json
import ast
import yaml
import os

from src.cls_env import cls_Env 
from src.cls_LLM import cls_LLM
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
def load_local_model() -> Dict[str, str]:
    """
    Endpoint to download the locally stored machine learning model file.

    Returns:
        Dict[str, Any] or FileResponse: The model file as a binary stream if it exists,
        otherwise an error message.
    """    
    model_path = config["output"]["model_path"]

    if not os.path.exists(model_path):
        return {"error": f"Model not found at: {model_path}"}

    return FileResponse(
        path=model_path,
        filename=os.path.basename(model_path),
        media_type="application/octet-stream"
    )
    
@app.post("/load_local_rules")
def load_local_rules() -> Dict[str, str]:
    """
    Endpoint to download the locally stored rule set file.

    Returns:
        Dict[str, Any] or FileResponse: The rules file if it exists,
        otherwise an error message.
    """    
    rules_path = config["output"]["rules_path"]

    if not os.path.exists(rules_path):
        return {"error": f"Model not found at: {rules_path}"}

    return FileResponse(
        path=rules_path,
        filename=os.path.basename(rules_path),
        media_type="application/octet-stream"
    )
    
@app.post("/escalate_issue")
def escalate_issue(diagnostic: DiagnosticInput)  -> Dict[str, str]:
    """
    Uses telemetry and rules to generate a recommended action via LLM prompt.

    Args:
        diagnostic (DiagnosticInput): Telemetry data string to be processed.

    Returns:
        Dict[str, str]: Response from LLM after evaluating rules against telemetry.
    """    
    rules_path = config["output"]["clould_path"]

    with open(rules_path, "r") as f:
        rules = json.load(f)

    llm_parameter = {
        "insert telemetry string here": diagnostic,
        "insert rules JSON here": rules
    }
    output = cls_llm.execute_llm_prompt(config["llm_prompt"], llm_parameter)
    return {
        "response": output
    }

@app.post("/generate_advice")
def generate_advice() -> Dict[str, str]:
    """
    Return a placeholder generic advice message.

    Returns:
        Dict[str, str]: Default message requesting more specific input.
    """    
    return {
        "advice": "No specific advice. Please provide a more detailed issue."
    }

@app.post("/log_error")
def log_error(error_log_str: DiagnosticInput) -> None:
    """
    Append a structured error log entry to the configured error log path.

    Args:
        error_log_str (DiagnosticInput): JSON string of the error log object.

    Returns:
        None
    """    
    error_log_dict = json.loads(error_log_str.message)
    cls_error_log = util.cls_ErrorLog(**error_log_dict)
    error_log_path=config["output"]["log_err_path"]
    with open(error_log_path, "a") as f:
        f.write(cls_error_log.model_dump_json() + "\n")
