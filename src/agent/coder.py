import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import AzureOpenAI

from src.constants.env import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
)
from src.prompts.code_prompt import CODER_PROMPT, EXAMPLES_CODER


class CoderModel:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.deployment_name = AZURE_OPENAI_DEPLOYMENT

    def __call__(self, plan: str):
        """
        Generate code based on plan.

        Args:
            plan: The coding plan or task
        Returns:
            Generated code or explanation
        """
        # Format prompt with examples
        formatted_prompt = CODER_PROMPT.format(example=EXAMPLES_CODER)

        messages = [
            {
                "role": "system",
                "content": formatted_prompt
            },
            {
                "role": "user",
                "content": plan
            }
        ]
        
        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content