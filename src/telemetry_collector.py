import os
import json
from os import path as Path

import yaml
import requests
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict, Optional

import src.util as util

NORMAL = 0
WARNING  = 1
CRITICAL  = 2

def download_from_cloud(model_url: str, save_path: str, object_name: str) -> bool:
    """
    Download a file from a cloud API and save it locally.

    Sends a POST request to the given model URL to retrieve binary content
    (e.g., a model or rules file), then writes the content to a specified local path.
    Logs success or failure messages.

    Args:
        model_url (str): URL of the cloud API endpoint to download from.
        save_path (str): Local file path to save the downloaded content.
        object_name (str): Descriptive name of the object being downloaded
                           (used for logging only).

    Returns:
        bool: True if download succeeds and file is saved, False otherwise.
    """    
    try:
        response = requests.post(model_url, verify=False)
        if response.ok:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"[Telemetry] {object_name} downloaded and saved to: {save_path}")
            return True
        print(f"[Telemetry] Failed to download {object_name}: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"[Telemetry] Request error: {e}")
    except Exception as e:
        print(f"[Telemetry] Unexpected error: {e}")
    return False


def show_intelligent_support_prompt(config: dict[str, Any]) -> str:
    """
    Display the intelligent support prompt and return the selected telemetry mode.

    This function prints an introduction message and a prompt for the user to choose
    a telemetry sharing option. The options are loaded from the provided config dictionary.
    It returns the corresponding telemetry mode string based on the user's input.

    Args:
        config (dict[str, Any]): Configuration dictionary containing the telemetry
                                 prompt message and valid options.

    Returns:
        str: The selected telemetry mode (e.g., "share_with_dell", "local_only",
             "disable_telemetry") or "invalid" if the input is not recognized.
    """
    print(config["telemetry"]["intro_message"])
    choice = input(config["telemetry"]["choice_prompt"]).strip()
    options = config["telemetry"]["options"]
    return options.get(choice, "invalid")


def ensure_file_available(config: Dict, endpoint_key: str, storage_key: str, object_name: str) -> None:
    """
    Ensure that a specific file (model or rules) is available locally.
    Downloads the file from the cloud if it's not already present.

    Args:
        config (Dict): Configuration dictionary containing cloud API and storage paths.
        endpoint_key (str): Key to locate the correct download endpoint in config["cloud_api"]["endpoints"].
        storage_key (str): Key to locate the correct local storage path in config["storage"].
        object_name (str): Name of the object for logging purposes.
    """
    base_url = config["cloud_api"]["base_url"]
    download_url = base_url + config["cloud_api"]["endpoints"][endpoint_key]
    save_path = config["storage"][storage_key]

    if not os.path.exists(save_path):
        print(f"[Telemetry] {object_name} not found locally. Downloading from {download_url}")
        download_from_cloud(download_url, save_path, object_name)
    else:
        print(f"[Telemetry] Local {object_name.lower()} already exists at {save_path}")


def update_telemetry_mode(config_path: str, new_mode: str):
    """
    Update the telemetry mode in the given YAML configuration file.

    This function reads the existing YAML config, modifies the telemetry mode,
    and writes the updated configuration back to the file.

    Args:
        config_path (str): Path to the YAML configuration file.
        new_mode (str): New telemetry mode to set (e.g., "share_with_dell", "local_only").

    Returns:
        None
    """    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["telemetry"]["mode"] = new_mode

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    print(f"Telemetry mode set to: {new_mode}")


def show_telemetry_prompt_and_store(config: dict, config_path: str) -> None:
    """
    Prompt the user to select telemetry mode if it has not been set,
    and update the configuration file with the selected mode.

    Args:
        config (dict): The loaded configuration dictionary.
        config_path (str): Path to the telemetry config YAML file.
    """
    current_mode = config.get("telemetry", {}).get("mode", "unset")

    if current_mode == "unset":
        mode = show_intelligent_support_prompt(config)

        if mode in {"share_with_dell", "local_only", "disable_telemetry"}:
            update_telemetry_mode(config_path, mode)
        else:
            print("[!] Invalid selection. No changes made.")
    else:
        print(f"[✔] Telemetry mode already set to '{current_mode}'. Skipping prompt.")


def load_model(model_path: str) -> RandomForestClassifier:
    """
    Load a machine learning model from the specified file path using joblib.

    Args:
        model_path (str): The file path to the saved model.

    Returns:
        RandomForestClassifier: RandomForestClassifier model.

    Raises:
        FileNotFoundError: If the model file does not exist at the specified path.
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file not found at: {model_path}")

def convert_diagnostic_to_df(diag_str)->pd.DataFrame:
    # Step 1: Split into key-value pairs
    pairs = diag_str.strip().split()

    # Step 2: Convert to dictionary
    diag_dict = {}
    for pair in pairs:
        key, value = pair.split("=")
        # Try converting to appropriate type
        if value in {"True", "False"}:
            value = value == "True"
        else:
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass
        diag_dict[key] = value

    # Step 3: Convert to single-row DataFrame
    diagnostics_df = pd.DataFrame([diag_dict])
    return diagnostics_df

def load_diagnostics(model_url:str, item:int)-> str:
    response = requests.post(
            model_url,
            json={"item": item}
        )
    diagnostics_dict:dict = response.json()
    # The diagnostics string
    diag_str = diagnostics_dict["diagnostics"]

    return diag_str


def predict_system_outcome(
    config: Dict[str, Any],
    model: RandomForestClassifier,
    item: int
) -> int:
    """
    Fetch system diagnostics from a remote machine and predict the outcome using a trained model.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing API endpoints.
        model (BaseEstimator): Trained machine learning model with a `.predict()` method.
        item (int, optional): Item identifier to pass to the diagnostics API. Defaults to 2.

    Returns:
        int: Predicted outcome based on the diagnostics.
    """
    base_url = config["machine_api"]["base_url"]
    endpoint = config["machine_api"]["endpoints"]["generate_system_parameters"]
    model_url = base_url + endpoint

    diagnostics_str = load_diagnostics(model_url, item)
    diagnostics_df = convert_diagnostic_to_df(diagnostics_str)
    prediction = model.predict(diagnostics_df)
    print(f"Predicted outcome: {int(prediction[0])}")

    return int(prediction[0]), diagnostics_df, diagnostics_str


def handle_diagnostic_threshold_breach(
    config: Dict[str, Any],
    col: str,
    value: float,
    rule: Dict[str, Any]
) -> None:
    """
    Handle a telemetry threshold breach based on rules, optionally prompting user and calling an API.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing API base URL.
        col (str): The name of the diagnostic column being evaluated.
        value (float): The current value of the diagnostic metric.
        rule (Dict[str, Any]): Rule dictionary containing threshold, action, and optional API info.

    Returns:
        None
    """
    if "max" in rule and value > rule["max"]:
        message = rule.get("action", f"{col} exceeds threshold.")
        print(f"[{col.upper()}] Value = {value} exceeds {rule['max']}")

        if rule.get("requires_confirmation", False):
            user_input = input(f"{message} (Y/N): ").strip().lower()
            if user_input == "y":
                base_url = config["machine_api"]["base_url"]
                endpoint = rule.get("api_service_name")
                
                if not endpoint:
                    print("[WARNING] API service name not specified in rule.")
                    return

                api_url = base_url + endpoint
                try:
                    response = requests.post(api_url)
                    diagnostics_dict: Dict[str, Any] = response.json()
                    print(diagnostics_dict.get("response", "No response received from API."))
                except requests.RequestException as e:
                    print(f"[ERROR] Failed to call API: {e}")
        else:
            print(f"[NOTICE] {message}")


def evaluate_diagnostics_and_respond(
    diagnostics_df: pd.DataFrame,
    config: Dict[str, Any]
) -> None:

    with open(config["storage"]["local_rules_path"], "r") as f:
        rules: Dict[str, Dict[str, Any]] = json.load(f)

    for col, rule in rules.items():
        if col not in diagnostics_df.columns:
            continue

        value = diagnostics_df[col].iloc[0]
        handle_diagnostic_threshold_breach(config, col, value, rule)
    return None

def raise_issue_and_respond(diagnostics_str: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Sends a diagnostics message to the escalation API endpoint and returns the response.

    This function posts the provided diagnostics string to the configured escalation API.
    If the request is successful, it extracts and returns the 'response' field from the JSON response.
    If the endpoint is missing or the request fails, it logs a warning or error and returns None.

    Args:
        diagnostics_str (str): The diagnostics message to be sent in the request body.
        config (Dict[str, Any]): Configuration dictionary containing the API base URL and endpoints.

    Returns:
        Optional[str]: The response string from the API if successful, otherwise None.
    """
    base_url = config["cloud_api"]["base_url"]
    endpoint = config["cloud_api"]["endpoints"]["escalate_issue"]
    
    if not endpoint:
        print("[WARNING] API service name not specified in rule.")
        return None

    api_url = base_url + endpoint
    response = requests.post(
        api_url,
        json={"message": diagnostics_str}
    )

    if response.ok:
        output_value = response.json().get("response")
        return output_value
    else:
        print("Request failed:", response.status_code, response.text)
        return None
    

def main() -> None:
    env_variables = util.read_env()
    config_path = env_variables["telemetry_config_path"]
    config = util.load_config(config_path)

    show_telemetry_prompt_and_store(config, config_path)

    # No performance of telemetry if mode is "local_only"
    if config["telemetry"]["mode"] != "local_only":
        #Update model
        ensure_file_available(config, "model_download", "local_model_path", "Model")
        #Update Rules
        ensure_file_available(config, "rules_download", "local_rules_path", "Rules")
        model: RandomForestClassifier = load_model(config["storage"]["local_model_path"])
        item = config["test_item"]
        outcome, diagnostics_df, diagnostics_str = predict_system_outcome(config, model, item)

        if outcome == WARNING:
            evaluate_diagnostics_and_respond(
                diagnostics_df,
                config
            )
        elif outcome == CRITICAL:
            output = raise_issue_and_respond(diagnostics_str, config)
            print(output)
            

     
if __name__ == "__main__":
    main()
