import yaml
import os
from typing import Dict, Any
from os import path as Path

from dotenv import load_dotenv

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path (str): The file path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed YAML configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
    
def read_env() -> Dict[str, str]:
    """
    Load environment variables from a .env file and return selected config paths as a dictionary.

    Returns:
        Dict[str, str]: A dictionary containing paths to telemetry, cloud, and machine configs.
    """
    load_dotenv()

    env_vars = {
        "telemetry_config_path": os.getenv("telemetry_config_path", ""),
        "cloud_config_path": os.getenv("cloud_config_path", ""),
        "machine_config_path": os.getenv("machine_config_path", "")
    }
    return env_vars