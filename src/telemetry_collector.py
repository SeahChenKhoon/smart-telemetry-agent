import os
import json
from os import path as Path

import yaml
import requests
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict

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


def ensure_model_available(config: dict, refresh: bool = False):
    """
    Ensure the model file exists locally; download from cloud if not.

    Args:
        config (dict): Configuration dictionary loaded from YAML.
    """
    base_url = config["cloud_api"]["base_url"]
    model_url = base_url + config["cloud_api"]["endpoints"]["model_download"]
    save_path = config["storage"]["local_model_path"]

    if not os.path.exists(save_path) or refresh:
        print(f"[Telemetry] Model not found locally. Downloading from {model_url}")
        download_from_cloud(model_url, save_path, "Model")
    else:
        print(f"[Telemetry] Local model already exists at {save_path}")


def ensure_rules_available(config: dict):
    """
    Ensure that the local telemetry rules file is available.

    This function checks if the local rules file exists at the configured path.
    If the file is missing, it attempts to download it from a cloud API endpoint.
    Otherwise, it confirms that the file already exists.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing cloud API base URL,
                                 endpoints, and local storage path for rules.

    Returns:
        None
    """    
    base_url = config["cloud_api"]["base_url"]
    rules_url = base_url + config["cloud_api"]["endpoints"]["rules_download"]
    save_path = config["storage"]["local_rules_path"]

    if not os.path.exists(save_path):
        print(f"[Telemetry] Rules not found locally. Downloading from {rules_url}")
        download_from_cloud(rules_url, save_path, "Rules")
    else:
        print(f"[Telemetry] Local rules already exists at {save_path}")


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


def load_diagnostics(model_url:str, item:int)-> pd.DataFrame:
    """
    Send a POST request to a model API endpoint to retrieve telemetry diagnostics
    for a specified item and convert the response into a pandas DataFrame.

    The API is expected to return a JSON object containing a "diagnostics" string,
    which is a space-separated set of key=value pairs. These are parsed and
    converted to appropriate Python types and then wrapped into a single-row DataFrame.

    Args:
        model_url (str): The full URL of the diagnostics API endpoint.
        item (int): The test item ID or index to send in the POST request.

    Returns:
        pd.DataFrame: A single-row DataFrame containing structured diagnostic data.
    """    
    response = requests.post(
            model_url,
            json={"item": item}
        )
    diagnostics_dict:dict = response.json()
    # The diagnostics string
    diag_str = diagnostics_dict["diagnostics"]

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

    diagnostics_df = load_diagnostics(model_url, item)
    prediction = model.predict(diagnostics_df)
    print(f"Predicted outcome: {int(prediction[0])}")

    return int(prediction[0]), diagnostics_df


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
    rules_path: str,
    config: Dict[str, Any]
) -> None:
    """
    Evaluate diagnostics against defined rules and trigger responses if thresholds are breached.

    This function loads diagnostic rules from a JSON file, compares them with current
    diagnostics data, and invokes appropriate actions (e.g., user prompts or API calls)
    by delegating to `handle_diagnostic_threshold_breach()`.

    Args:
        diagnostics_df (pd.DataFrame): DataFrame containing one row of telemetry values.
        rules_path (str): Path to the JSON rules file defining max values, actions, etc.
        config (Dict[str, Any]): Configuration dictionary, including machine API details.

    Returns:
        None
    """
    with open(rules_path, "r") as f:
        rules: Dict[str, Dict[str, Any]] = json.load(f)

    for col, rule in rules.items():
        if col not in diagnostics_df.columns:
            continue

        value = diagnostics_df[col].iloc[0]
        handle_diagnostic_threshold_breach(config, col, value, rule)


def main() -> None:
    """
    Main entry point for the intelligent support and telemetry evaluation system.

    This function:
    - Loads environment variables and telemetry configuration from a YAML file.
    - Ensures the local model and rule files are available or downloaded.
    - Prompts the user to select a telemetry sharing mode and updates the config.
    - If telemetry is not limited to local-only, it refreshes resources, loads the model,
      performs system diagnostics, and evaluates them using predefined rules.

    Returns:
        None
    """
    env_variables = util.read_env()
    config_path = env_variables["telemetry_config_path"]
    config = util.load_config(config_path)

    ensure_model_available(config)
    ensure_rules_available(config)
    show_telemetry_prompt_and_store(config, config_path)

    # No performance of telemetry if mode is "local_only"
    if config["telemetry"]["mode"] != "local_only":
        ensure_model_available(config, refresh=True)
        ensure_rules_available(config, refresh=True)

        model: RandomForestClassifier = load_model(config["storage"]["local_model_path"])
        item = config["test_item"]
        outcome, diagnostics_df = predict_system_outcome(config, model, item)

        if outcome == WARNING:
            evaluate_diagnostics_and_respond(
                diagnostics_df,
                config["storage"]["local_rules_path"],
                config
            )
        
     
if __name__ == "__main__":
    main()
