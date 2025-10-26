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
from src.prompts.reporter_prompt import REPORTER_PROMPT


class Reporter:
    """
    Agent responsible for generating comprehensive medical reports from vision model outputs.
    Uses Azure OpenAI to create detailed, patient-friendly explanations with visual evidence.
    """

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.deployment_name = AZURE_OPENAI_DEPLOYMENT

    def __call__(self, query: str, output: str, plan: str) -> dict:
        """
        Generate a comprehensive medical report using Azure OpenAI.

        Args:
            query: The user's medical question
            output: The results from the execute code
            plan: The analysis plan that was executed

        Returns:
            dict with 'answer' and 'explanation' keys
        """
        # Format the prompt with query, plan, and output
        formatted_prompt = REPORTER_PROMPT.format(
            query=query,
            plan=plan,
            output=output
        )

        messages = [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]

        
        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.3,
            max_tokens=3000
        )

        content = response.choices[0].message.content

        # Parse answer and explanation from XML-like tags
        answer = ""
        explanation = ""

        if "<answer>" in content and "</answer>" in content:
            answer_start = content.find("<answer>") + len("<answer>")
            answer_end = content.find("</answer>")
            answer = content[answer_start:answer_end].strip()

        if "<explanation>" in content and "</explanation>" in content:
            explanation_start = content.find("<explanation>") + len("<explanation>")
            explanation_end = content.find("</explanation>")
            explanation = content[explanation_start:explanation_end].strip()

        return {
            "answer": answer,
            "explanation": explanation,
            "report": content  # Full content for backward compatibility
        }
