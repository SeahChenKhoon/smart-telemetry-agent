import yaml
import requests
import os

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


def main() -> None:
    config = load_config()
    ensure_model_available(config)
    show_telemetry_prompt_and_store(config)

        

if __name__ == "__main__":
    main()
