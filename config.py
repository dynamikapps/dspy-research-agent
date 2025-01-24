import os
import sys
from typing import Optional
from dotenv import load_dotenv, find_dotenv
import dspy


class LLMConfig:
    """Configuration class for managing LLM providers and settings."""

    def __init__(self):
        # Force reload of .env file
        load_dotenv(find_dotenv(), override=True)

        # Debug print
        print(f"Loading environment variables...")
        print(f"LLM_PROVIDER from env: {os.getenv('LLM_PROVIDER')}")

        # Load all environment variables
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_api_base = os.getenv(
            'OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1')
        self.ollama_api_base = os.getenv(
            'OLLAMA_API_BASE', 'http://localhost:11434')
        self.llm_provider = os.getenv('LLM_PROVIDER', 'openai')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
        self.openrouter_model = os.getenv(
            'OPENROUTER_MODEL', 'deepseek/deepseek-chat')

        # Debug print
        print(f"Loaded LLM provider: {self.llm_provider}")

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration based on selected provider."""
        if self.llm_provider == 'openai' and not self.openai_api_key:
            print("Error: OpenAI API key is required when using OpenAI provider.")
            sys.exit(1)
        elif self.llm_provider == 'openrouter' and not all([self.openrouter_api_key, self.openrouter_api_base]):
            print(
                "Error: OpenRouter API key and base URL are required when using OpenRouter provider.")
            sys.exit(1)
        elif self.llm_provider == 'ollama' and not self.ollama_api_base:
            print("Error: Ollama API base URL is required when using Ollama provider.")
            sys.exit(1)

    def get_llm_provider(self) -> str:
        """Get the current LLM provider name."""
        return self.llm_provider

    def get_dspy_lm(self) -> dspy.LM:
        """Configure and return the appropriate DSPy language model based on provider."""
        # Debug print
        print(f"Configuring DSPy for provider: {self.llm_provider}")

        if self.llm_provider == 'openai':
            return dspy.LM(f'openai/{self.openai_model}',
                           api_key=self.openai_api_key)

        elif self.llm_provider == 'ollama':
            return dspy.LM(f'ollama/{self.ollama_model}',
                           api_key='',
                           api_base=self.ollama_api_base)

        elif self.llm_provider == 'openrouter':
            return dspy.LM(f'openai/{self.openrouter_model}',
                           api_key=self.openrouter_api_key,
                           api_base=self.openrouter_api_base)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")


# Create a singleton instance
config = LLMConfig()
