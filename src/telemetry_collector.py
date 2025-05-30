import os
import json
from os import path as Path

import yaml
import requests
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict, Optional, List, Tuple, Union

import src.util as util
import src.cls_telemetric as cls_Rule
import src.cls_telemetric as cls_ErrorLog



NORMAL = 0
WARNING  = 1
CRITICAL  = 2

def download_from_cloud(verify_cert:bool, model_url: str, save_path: str, object_name: str) -> bool:
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
        response = requests.post(model_url, verify=verify_cert)
        if response.ok:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"\n{object_name} downloaded and saved to: {save_path}\n")
            return True
        print(f"\nFailed to download {object_name}: {response.status_code} - {response.text}\n")
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
    print(config["privacy"]["intro_message"])
    choice = input(config["privacy"]["choice_prompt"]).strip()
    options = config["privacy"]["options"]
    return options.get(choice, "invalid")


def ensure_file_available(config: Dict, endpoint_key: str, storage_key: str, object_name: str) -> None:
    base_url = config["cloud_api"]["base_url"]
    download_url = base_url + config["cloud_api"]["endpoints"][endpoint_key]
    save_path = config["storage"][storage_key]

    print(f"\nDownloading/Updating {object_name} from {download_url}\n")
    download_from_cloud(config["cloud_api"]["verify_cert"], download_url, save_path, object_name)


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
    config["privacy"]["mode"] = new_mode

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    print(f"Privacy mode set to: {new_mode}")


def show_telemetry_prompt_and_store(config: dict, config_path: str) -> str:
    """
    Display a prompt to configure telemetry privacy mode if it is currently unset.

    If the telemetry mode in the configuration is set to 'unset', this function
    prompts the user to select a mode using `show_intelligent_support_prompt()`.
    If the selected mode is valid, it updates the mode in the configuration file
    using `update_telemetry_mode()`. If the mode is already set, it skips the prompt
    and returns the existing mode.

    Args:
        config (dict): The configuration dictionary containing the current privacy mode.
        config_path (str): The file path to the telemetry configuration file to update.

    Returns:
        str: The telemetry mode selected by the user or the existing mode if already set.
    """    
    current_mode = config["privacy"]["mode"]

    if current_mode == "unset":
        privacy_mode = show_intelligent_support_prompt(config)

        if privacy_mode in {"share_with_dell", "local_only", "disable_telemetry"}:
            update_telemetry_mode(config_path, privacy_mode)
        else:
            print("[!] Invalid selection. No changes made.")
        return privacy_mode
    else:
        print(f"\nPrivacy mode already set to '{current_mode}'. Skipping privacy selection mode.\n")
    return current_mode


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

def convert_diagnostic_to_df(diagnostics: Dict[str, float]) -> pd.DataFrame:
    """
    Convert a dictionary of telemetry diagnostics into a single-row DataFrame
    suitable for model prediction.

    Args:
        diagnostics (Dict[str, float]): The telemetry data as key-value pairs.

    Returns:
        pd.DataFrame: A DataFrame with one row representing the diagnostics.
    """
    return pd.DataFrame([diagnostics])

def parse_telemetry_string(telemetry_str: str) -> Dict[str, Union[float, bool]]:
    """
    Parse a telemetry string into a dictionary with appropriate types.

    Args:
        telemetry_str (str): A space-separated string of key=value pairs.

    Returns:
        Dict[str, Union[float, bool]]: Parsed telemetry data with float or boolean values.
    """
    telemetry_dict: Dict[str, Union[float, bool]] = {}

    for pair in telemetry_str.split():
        key, value = pair.split("=")
        if value.lower() in {"true", "false"}:
            telemetry_dict[key] = value.lower() == "true"
        else:
            telemetry_dict[key] = float(value)

    return telemetry_dict


def fetch_telemetry_from_machine(model_url: str, item: int):
    response = requests.post(
        model_url,
        json={"item": item}
    )
    response_dict = response.json()
    # Convert the diagnostics string to a dictionary of floats
    telemetry_str = response_dict["diagnostics"]
    telemetry_dict: Dict[str, float] = parse_telemetry_string(telemetry_str)

    return telemetry_dict, telemetry_str

def predict_telemetry_outcome(
    diagnostics_df: pd.DataFrame,
    model: RandomForestClassifier
) -> int:
    prediction = model.predict(diagnostics_df)

    print(f"Predicted outcome: {int(prediction[0])}")

    return int(prediction[0])

def retrieve_telemetry_from_machine(
    config: Dict[str, Any],
    item: int
) -> Tuple[int, pd.DataFrame, Dict[str, float]]:
    base_url = config["machine_api"]["base_url"]
    endpoint = config["machine_api"]["endpoints"]["generate_system_parameters"]
    api_url = base_url + endpoint
    telemetry_dict, telemetry_str = fetch_telemetry_from_machine(api_url, item)
    telemetry_df = convert_diagnostic_to_df(telemetry_dict)

    return telemetry_df, telemetry_dict, telemetry_str

def match_rules(rules: List[cls_Rule.cls_Rule], telemetry_dict: Dict[str, float]) -> List[cls_Rule.cls_Rule]:
    """
    Evaluate a list of rules against telemetry data and return all matching rules.

    A rule matches if the telemetry dictionary contains the column specified by the rule,
    and the corresponding value falls within the rule's [min, max) range.

    Args:
        rules (List[cls_Rule]): A list of rule objects to evaluate.
        telemetry_dict (Dict[str, float]): A dictionary of telemetry values keyed by attribute name.

    Returns:
        List[cls_Rule]: A list of rule objects that match the telemetry conditions.
    """
    return [
        rule for rule in rules
        if rule.col in telemetry_dict and rule.min <= telemetry_dict[rule.col] < rule.max
    ]

def evaluate_diagnostics(
    telemetry_dict: Dict[str, float],
    config: Dict[str, Any]
) -> List[cls_Rule.cls_Rule]:
    # Read Rules Table
    with open(config["storage"]["local_rules_path"], "r") as f:
        raw_rules : Dict[str, Dict[str, Any]] = json.load(f)
        rules_list: List[cls_Rule.cls_Rule] = [cls_Rule.cls_Rule(**v) for v in raw_rules.values()]
    matched_rule = match_rules(rules_list, telemetry_dict)
    return matched_rule


def post_error_log_to_api(
    error_log_dict: Dict[str, Any],
    config: Dict[str, Any],
    base_url: str,
    verify_cert: bool
) -> None:
    """
    Send an error log dictionary to the configured cloud API endpoint.

    Args:
        error_log_dict (Dict[str, Any]): The error log to be sent.
        config (Dict[str, Any]): Configuration dictionary containing API endpoints.
        base_url (str): The base URL of the cloud API.
        verify_cert (bool): Whether to verify the SSL certificate.

    Returns:
        None
    """
    error_log_str = json.dumps(error_log_dict)
    endpoint = config["cloud_api"]["endpoints"]["log_error"]
    api_url = base_url + endpoint

    requests.post(
        api_url,
        json={"message": error_log_str},
        verify=verify_cert
    )

def raise_issue_and_get_adviced(verify_cert:bool, telemetry_str: str, config: Dict[str, Any]):
    base_url = config["cloud_api"]["base_url"]
    endpoint = config["cloud_api"]["endpoints"]["escalate_issue"]
    
    if not endpoint:
        print("[WARNING] API service name not specified in rule.")
        return None

    api_url = base_url + endpoint
    response = requests.post(
        api_url,
        json={"message": telemetry_str},
        verify=verify_cert
    )
    if response.ok:
        output_value = response.json().get("response")
        if not output_value:
            error_log_dict = {
                "error_message": "output_value is empty",
                "telemetry_str": telemetry_str,
                "output_value": output_value
            }
            post_error_log_to_api(error_log_dict, config, base_url, verify_cert)

            return []
        else:
            mapped_rules: List[cls_Rule.cls_Rule] = cls_Rule.cls_Rule.parse_rule_string_to_list(output_value)
            return mapped_rules
    else:
        print("Request failed:", response.status_code, response.text)
        return []

def handle_matched_rules(
    matched_rules: List[cls_Rule.cls_Rule],
    telemetry_dict: Dict[str, float],
    config: Dict[str, Any]
) -> None:
    """
    Handle all matched rules by invoking each rule's breach handler.

    Args:
        matched_rules (List[cls_Rule]): List of rule objects that matched the telemetry conditions.
        telemetry_dict (Dict[str, float]): Telemetry data with key-value pairs.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        None
    """
    for rule in matched_rules:
        col = rule.col
        value = telemetry_dict.get(col)
        if value is not None:
            rule.handle_breach(config, value)


def main() -> None:
    env_variables = util.read_env()
    config_path = env_variables["telemetry_config_path"]
    config = util.load_config(config_path)

    privacy_mode = show_telemetry_prompt_and_store(config, config_path)
    # No performance of telemetry if mode is "local_only"
    privacy_stmt = config["privacy"]["stmt"]
    privacy_stmt = privacy_stmt.format(
            privacy_mode=privacy_mode,
            telemetry_config_path=config_path
        )
    if privacy_mode != "disable_telemetry":    
        # Update model
        ensure_file_available(config, "model_download", "local_model_path", "Model")
        # Update Rules
        ensure_file_available(config, "rules_download", "local_rules_path", "Rules")
        # Load Model
        model: RandomForestClassifier = load_model(config["storage"]["local_model_path"])
        
        while True:
            print(config["telemetry"]["menu_option"])
            print(privacy_stmt)
            try:
                user_input = int(input("Enter item ID (0–4) to test, or 9 to exit: "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if user_input == 9:
                print("Exiting loop.")
                break

            if user_input not in [0, 1, 2, 3, 4]:
                print("Please enter a valid item ID between 0 and 4.")
                continue

            item = user_input
            telemetry_df, telemetry_dict, telemetry_str = retrieve_telemetry_from_machine(config, item)
            telemetry_outcome = predict_telemetry_outcome(telemetry_df, model)

            if telemetry_outcome == WARNING:
                matched_rule = evaluate_diagnostics(telemetry_dict, config)
                handle_matched_rules(matched_rule, telemetry_dict, config)

            elif telemetry_outcome == CRITICAL and privacy_mode == "share_with_dell":
                matched_rule = raise_issue_and_get_adviced(
                    config["cloud_api"]["verify_cert"],
                    telemetry_str,
                    config
                )
                handle_matched_rules(matched_rule, telemetry_dict, config)
    else:
        print(privacy_stmt)
     
if __name__ == "__main__":
    main()
