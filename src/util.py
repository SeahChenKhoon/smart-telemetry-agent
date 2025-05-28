import yaml
import os
from typing import Dict, Any, Optional
from os import path as Path
from openai import OpenAI, AzureOpenAI
from datetime import datetime
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from src.cls_env import cls_Env

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
        # Config Path
        "telemetry_config_path": os.getenv("TELEMETRY_CONFIG_PATH", ""),
        "cloud_config_path": os.getenv("CLOUD_CONFIG_PATH", ""),
        "machine_config_path": os.getenv("MACHINE_CONFIG_PATH", ""),

        ## LLM Selection Flag OpenAI ot Azure OpenAI
        "llm_provider": os.getenv("LLM_PROVIDER", ""),

        # Open AI Setting
        "open_api_key": os.getenv("OPENAI_API_KEY", ""),
        "llm_model_name": os.getenv("LLM_MODEL_NAME", ""),

        # Open AI Settings
        "azure_api_version": os.getenv("AZURE_API_VERSION", ""),
        "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "azure_openai_endpoint": os.getenv("LLM_MODEL_NAME", ""),
        "azure_openai_key": os.getenv("AZURE_OPENAI_KEY", "")
    }
    return env_vars

class LLMPromptExecutor:
    def __init__(
        self, cls_env:cls_Env
    ):
        provider = cls_env.llm_provider
        client = self._get_llm_client(cls_env)
        model_arg = self._get_model_arguments(
            provider=provider,
            model_name=cls_env.model_name,
            azure_deployment_id=cls_env.azure_deployment_id
        )
        
        self.provider=client
        self.model_arg=model_arg
        self.llm_temperature=cls_env.llm_temperature


    def _get_llm_client(self, env_vars: Any) -> Any:
        """
        Returns an LLM client instance based on the provider specified in the environment variables.

        Supports 'azure' (AzureOpenAI) and 'openai' (OpenAI). Raises an error for unsupported providers.

        Args:
            env_vars (Any): An object containing environment variables such as provider name, API keys, etc.

        Returns:
            Any: An instance of either AzureOpenAI or OpenAI depending on the provider.

        Raises:
            ValueError: If the provider is not 'openai' or 'azure'.
        """
        provider = env_vars.llm_provider.lower()

        if provider == "azure":
            return AzureOpenAI(
                api_key=env_vars.azure_openai_key,
                api_version=env_vars.azure_api_version,
                azure_endpoint=env_vars.azure_openai_endpoint,
            )

        if provider == "openai":
            return OpenAI(api_key=env_vars.openai_api_key)

        raise ValueError(f"Unsupported provider: '{provider}'. Expected 'openai' or 'azure'.")


    def _get_model_arguments(
        self,
        provider: str,
        model_name: str = "",
        azure_deployment_id: str = ""
    ) -> str:
        """
        Returns the appropriate model argument based on the LLM provider.

        For 'azure', it returns the Azure deployment ID.  
        For 'openai', it returns the OpenAI model name.  
        Raises an error if required arguments are missing or if the provider is unsupported.

        Args:
            provider (str): LLM provider name (e.g., 'openai' or 'azure').
            model_name (str, optional): Model name for OpenAI. Required if provider is 'openai'.
            azure_deployment_id (str, optional): Deployment ID for Azure. Required if provider is 'azure'.

        Returns:
            str: The resolved model identifier (either `model_name` or `azure_deployment_id`).

        Raises:
            ValueError: If required parameters are missing or provider is unsupported.
        """
        provider = provider.lower()

        if provider == "azure":
            if not azure_deployment_id:
                raise ValueError("azure_deployment_id must be provided for Azure OpenAI")
            return azure_deployment_id

        if provider == "openai":
            if not model_name:
                raise ValueError("model_name must be provided for OpenAI")
            return model_name

        raise ValueError(f"Unsupported provider: '{provider}'.")


    def _get_chat_completion(self, provider: Any, model: str, prompt: str, llm_temperature: float = 0.2) -> Any:
        """
        Sends a prompt to the chat model and returns the response.

        Args:
            provider (Any): The provider instance with a chat.completions.create method.
            model (str): The model name to use.
            prompt (str): The user prompt to send.
            llm_temperature (float, optional): Sampling llm_temperature. Defaults to 0.2.

        Returns:
            Any: The response object from the provider.
        """
        return provider.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_temperature,
        )


    def _strip_markdown_fences(self, text: str) -> str:
        """
        Removes all Markdown-style triple backtick fences from LLM output.
        Logs a warning if any stripping was performed.

        Args:
            text (str): The raw LLM output string.

        Returns:
            str: The cleaned string without Markdown-style code fences.
        """
        lines = text.strip().splitlines()
        cleaned_lines = []

        for line in lines:
            if line.strip().startswith("```"):
                continue  # Skip the fence line
            cleaned_lines.append(line)


        return "\n".join(cleaned_lines)


    def execute_llm_prompt(
        self,
        llm_import_prompt: str,
        llm_parameter: dict
    ) -> str:
        """
        Formats the given prompt with provided parameters, sends it to the LLM,
        and returns the cleaned response content without Markdown fences.

        Args:
            llm_import_prompt (str): The base prompt template to send to the LLM.
            llm_parameter (dict): Dictionary of parameters to format into the prompt.

        Returns:
            str: The LLM-generated string response with formatting artifacts removed.
        """
        formatted_prompt = llm_import_prompt.format(**llm_parameter)
        response = self._get_chat_completion(
            self.provider,
            self.model_arg,
            formatted_prompt,
            self.llm_temperature
        )
        return self._strip_markdown_fences(response.choices[0].message.content.strip())

class cls_ErrorLog(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    error_message: str
    telemetry_str: str
    output_value: Optional[str]

    def to_json(self) -> str:
        return self.model_dump_json()