from dotenv import load_dotenv
import os

class cls_Env:
    def __init__(self):
        load_dotenv(override=True)
        # Config Path
        self.telemetry_conig_path = os.getenv("TELEMETRY_CONFIG_PATH")
        self.clould_config_path = os.getenv("CLOUD_CONFIG_PATH")
        self.machine_config_path = os.getenv("MACHINE_CONFIG_PATH")

        # LLM Config
        self.llm_provider  = os.getenv("LLM_PROVIDER")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model_name = os.getenv("LLM_MODEL_NAME")
        
        self.azure_api_version = os.getenv("AZURE_API_VERSION")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.azure_deployment_id = os.getenv("AZURE_DEPLOYMENT_ID")

        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE"))
