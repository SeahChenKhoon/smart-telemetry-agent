from dotenv import load_dotenv
import os

class cls_Env:
    def __init__(self) -> None:
        """
        Initialize environment configuration variables from a .env file.

        This method loads key configuration values used throughout the application,
        including telemetry paths, cloud and machine config paths, and settings
        for both OpenAI and Azure OpenAI Large Language Model (LLM) providers.

        Attributes:
            telemetry_conig_path (str): Path to telemetry configuration YAML file.
            clould_config_path (str): Path to cloud configuration YAML file.
            machine_config_path (str): Path to machine configuration YAML file.
            llm_provider (str): Provider name for the LLM (e.g., "openai" or "azure").
            openai_api_key (str): API key for OpenAI access.
            llm_model_name (str): LLM model name or Azure deployment ID.
            azure_api_version (str): Azure OpenAI API version.
            azure_openai_endpoint (str): Endpoint URL for Azure OpenAI.
            azure_openai_key (str): API key for Azure OpenAI.
            azure_deployment_id (str): Deployment ID for Azure OpenAI.
            llm_temperature (float): Temperature setting for LLM response variability.
        """
        load_dotenv(override=True)

        # Config Paths
        self.telemetry_conig_path: str = os.getenv("TELEMETRY_CONFIG_PATH")
        self.clould_config_path: str = os.getenv("CLOUD_CONFIG_PATH")
        self.machine_config_path: str = os.getenv("MACHINE_CONFIG_PATH")

        # LLM Config
        self.llm_provider: str = os.getenv("LLM_PROVIDER")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY")
        self.llm_model_name: str = os.getenv("LLM_MODEL_NAME")

        self.azure_api_version: str = os.getenv("AZURE_API_VERSION")
        self.azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key: str = os.getenv("AZURE_OPENAI_KEY")
        self.azure_deployment_id: str = os.getenv("AZURE_DEPLOYMENT_ID")

        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
