from src.cls_env import cls_Env 
from typing import Any
from openai import OpenAI, AzureOpenAI

class cls_LLM:
    def __init__(
        self, cls_env:cls_Env
    ):
        provider = cls_env.llm_provider
        client = self._get_llm_client(cls_env)
        model_arg = self._get_model_arguments(
            provider = provider,
            model_name = cls_env.llm_model_name,
            azure_deployment_id = cls_env.azure_deployment_id
        )
        self.llm_temperature=cls_env.llm_temperature
        self.provider=client
        self.model_arg=model_arg

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