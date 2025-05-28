from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import requests

class cls_SystemDiagnostics(BaseModel):
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

class cls_Rule(BaseModel):
    col: str
    min: float
    max: float
    action: str
    requires_confirmation: bool
    api_service_name: Optional[str]

    def handle_breach(self, config: Dict[str, Any], value: float) -> None:
        """
        Handle the threshold breach for this rule.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            value (float): The actual telemetry value that triggered the rule.

        Returns:
            None
        """
        print(f"[{self.col.upper()}] Value = {value} exceeds threshold = {self.max}")
        print(f"[ACTION] {self.action}")

        if self.requires_confirmation:
            user_input = input(f"{self.action} (Y/N): ").strip().lower()
            if user_input != "y":
                print("[INFO] User declined the recommended action.")
                return

        if not self.api_service_name:
            print("[WARNING] No API service specified for this rule.")
            return

        base_url = config["machine_api"]["base_url"]
        api_url = base_url + self.api_service_name

        try:
            response = requests.post(api_url)
            response.raise_for_status()
            result = response.json()
            print("[API RESPONSE]", result.get("response", "No response received."))
        except requests.RequestException as e:
            print(f"[ERROR] Failed to call API at {api_url}: {e}") 

    def parse_rule_string_to_list(rule_str: str) -> List["cls_Rule"]:
        """
        Parse a single pipe-delimited rule string into a list containing one cls_Rule object.

        Args:
            rule_str (str): Rule string in the format "col|min|max|action|requires_confirmation|api_service_name"

        Returns:
            List[cls_Rule]: A list containing a single cls_Rule object.
        """
        parts = rule_str.split("|")
        rule = cls_Rule(
            col=parts[0],
            min=float(parts[1]),
            max=float(parts[2]),
            action=parts[3],
            requires_confirmation=parts[4].lower() == "true",
            api_service_name=parts[5] if parts[5] else None
        )
        return [rule]