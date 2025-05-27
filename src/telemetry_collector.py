import yaml
import requests
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict
import pandas as pd

NORMAL = 0
WARNING  = 1
CRITICAL  = 2

def load_config():
    with open("telemetry_config.yml", "r") as f:
        return yaml.safe_load(f)

def download_model_from_cloud(model_url: str, save_path: str) -> bool:
    """
    Download a model file from a cloud API and save it locally.

    Args:
        model_url (str): URL to the cloud API endpoint providing the model.
        save_path (str): Local file path to save the downloaded model.

    Returns:
        bool: True if download succeeds, False otherwise.
    """
    try:
        response = requests.post(model_url)
        if response.ok:
            _save_model_file(response.content, save_path)
            _log_success(save_path)
            return True
        _log_failure(response.status_code, response.text)
    except requests.RequestException as e:
        print(f"[Telemetry] Request error: {e}")
    except Exception as e:
        print(f"[Telemetry] Unexpected error: {e}")
    return False

def _save_model_file(content: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(content)

def _log_success(path: str) -> None:
    print(f"[Telemetry] Model downloaded and saved to: {path}")

def _log_failure(status: int, message: str) -> None:
    print(f"[Telemetry] Failed to download model: {status} - {message}")

def show_intelligent_support_prompt() -> str:
    print("\nEnable Intelligent Support?")
    print("To provide proactive, AI-driven system optimization and support, this assistant collects")
    print("diagnostic data like CPU usage, system temperature, and application performance.")
    print("You control what is shared.\n")

    print("1. [ ] Share with Dell to receive full support and updates")
    print("2. [ ] Keep data local to device only")
    print("3. [ ] Do not collect telemetry at all\n")

    choice = input("Enter your choice (1-3): ").strip()

    options = {
        "1": "share_with_dell",
        "2": "local_only",
        "3": "disable_telemetry"
    }

    return options.get(choice, "invalid")

def ensure_model_available(config: dict):
    """
    Ensure the model file exists locally; download from cloud if not.

    Args:
        config (dict): Configuration dictionary loaded from YAML.
    """
    base_url = config["cloud_api"]["base_url"]
    model_url = base_url + config["cloud_api"]["endpoints"]["model_download"]
    save_path = config["storage"]["local_model_path"]

    if not os.path.exists(save_path):
        print(f"[Telemetry] Model not found locally. Downloading from {model_url}")
        download_model_from_cloud(model_url, save_path)
    else:
        print(f"[Telemetry] Local model already exists at {save_path}")

def update_telemetry_mode(config_path: str, new_mode: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["telemetry"]["mode"] = new_mode

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    print(f"[✔] Telemetry mode set to: {new_mode}")


def show_telemetry_prompt_and_store(config: dict, config_path: str = "telemetry_config.yml") -> None:
    """
    Prompt the user to select telemetry mode if it has not been set,
    and update the configuration file with the selected mode.

    Args:
        config (dict): The loaded configuration dictionary.
        config_path (str): Path to the telemetry config YAML file.
    """
    current_mode = config.get("telemetry", {}).get("mode", "unset")

    if current_mode == "unset":
        mode = show_intelligent_support_prompt()

        if mode in {"share_with_dell", "local_only", "disable_telemetry"}:
            update_telemetry_mode(config_path, mode)
        else:
            print("[!] Invalid selection. No changes made.")
    else:
        print(f"[✔] Telemetry mode already set to '{current_mode}'. Skipping prompt.")


def load_model(model_path:str):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("[✓] Model loaded from:", model_path)
        return model
    else:
        raise FileNotFoundError(f"[✗] Model file not found at: {model_path}")

def load_diagnostics(model_url:str, item:int)-> pd.DataFrame:
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
    item: int = 2
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

def evaluate_diagnostics_and_respond(
    diagnostics_df: pd.DataFrame,
    rules_path: str, 
    config: Dict[str, Any]
) -> None:
    with open(rules_path, "r") as f:
        rules: Dict[str, Dict[str, float]] = json.load(f)

    for col, rule in rules.items():
        if col not in diagnostics_df.columns:
            continue

        value = diagnostics_df[col].iloc[0]
        if "max" in rule and value > rule["max"]:
            message = rule.get("action", f"{col} exceeds threshold.")
            print(f"[{col.upper()}] Value = {value} exceeds {rule['max']}")

            if rule.get("requires_confirmation", False):
                user_input = input(f"{message} (Y/N): ").strip().lower()
                if user_input == "y":
                    base_url = config["machine_api"]["base_url"]
                    endpoint = rule.get("api_service_name")
                    api_url = base_url + endpoint
                    response = requests.post(api_url)
                    diagnostics_dict:dict = response.json()
                    message = diagnostics_dict['response']
                    print(message)
            else:
                print(f"[NOTICE] {message}")


def main() -> None:
    config = load_config()
    ensure_model_available(config)
    show_telemetry_prompt_and_store(config)
    if config["telemetry"]["mode"] != "disable_telemetry":
        # Load local Model
        model:RandomForestClassifier = load_model(config["storage"]["local_model_path"])
        item = config["test_item"]
        outcome, diagnostics_df = predict_system_outcome(config, model, item)
        if outcome == WARNING:
            evaluate_diagnostics_and_respond(diagnostics_df, config["storage"]["local_rules_path"], config)
        
      

if __name__ == "__main__":
    main()
