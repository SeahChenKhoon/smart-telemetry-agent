from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

import yaml

app = FastAPI()

class DiagnosticsItem(BaseModel):
    item: int

class DiagnosticOutput(BaseModel):
    cpu_usage: float
    memory_usage: float
    network_traffic: float
    power_consumption: float
    num_executed_instructions: float
    execution_time: float
    energy_efficiency: float
    task_priority: int
    temperature: float
    task_type_compute: bool
    task_type_io: bool
    task_type_network: bool

def load_config()-> Dict[str, Any]:
    """
    Load configuration settings from a YAML file.

    Reads the 'machine_config.yml' file from the current directory
    and returns its contents as a dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing configuration keys and values.
    """    
    with open("machine_config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config


@app.post("/generate_system_parameters")
def generate_system_parameters(item: DiagnosticsItem):
    config = load_config()
    df = pd.DataFrame(config["Data"])

    diagnostics: List[DiagnosticOutput] = [
        DiagnosticOutput(**row) for row in df.to_dict(orient="records")
    ]

    # Index access using item index
    selected_diag = diagnostics[item.item]

    # Convert to "key=value" string
    diag_str = " ".join(f"{k}={v}" for k, v in selected_diag.dict().items())
    print(diag_str)
    return {
        "item": item.item,
        "diagnostics": diag_str
    }

@app.post("/reduce_screen_brightness")
def reduce_screen_brightness():
    return {
        "response": "Screen brightness reduced"
    }

@app.post("/increase_fan_speed")
def increase_fan_speed():
    return {
        "response": "Fan Speed increased"
    }

@app.post("/enable_cpu_cooling")
def enable_cpu_cooling():
    return {
        "response": "Additional cooling enabled"
    }

@app.post("/enable_cpu_cooling")
def enable_cpu_cooling():
    return {
        "response": "Emergency shutdown completed"
    }
