import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import AzureOpenAI
import google.generativeai as genai

from src.constants.env import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from src.prompts.reporter_prompt import REPORTER_PROMPT


class Reporter:
    """
    Agent responsible for generating comprehensive medical reports from vision model outputs.
    Uses Azure OpenAI or Google Gemini to create detailed, patient-friendly explanations with visual evidence.
    """

    def __init__(self, model_provider: str = "openai"):
        """
        Initialize Reporter with specified model provider.

        Args:
            model_provider: Either "openai" or "gemini"
        """
        self.model_provider = model_provider.lower()

        if self.model_provider == "openai":
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
            )
            self.deployment_name = AZURE_OPENAI_DEPLOYMENT
        elif self.model_provider == "gemini":
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai.GenerativeModel(GEMINI_MODEL)
            self.deployment_name = GEMINI_MODEL
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}. Use 'openai' or 'gemini'.")

    def __call__(self, query: str, output: str, plan: str) -> dict:
        """
        Generate a comprehensive medical report.

        Args:
            query: The user's medical question
            output: The results from the execute code
            plan: The analysis plan that was executed

        Returns:
            dict with 'answer' and 'explanation' keys
        """
        if self.model_provider == "openai":
            content = self._call_openai(query, output, plan)
        elif self.model_provider == "gemini":
            content = self._call_gemini(query, output, plan)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

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

    def _call_openai(self, query: str, output: str, plan: str) -> str:
        """Call Azure OpenAI API"""
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

        return response.choices[0].message.content

    def _call_gemini(self, query: str, output: str, plan: str) -> str:
        """Call Google Gemini API"""
        # Format the prompt with query, plan, and output
        formatted_prompt = REPORTER_PROMPT.format(
            query=query,
            plan=plan,
            output=output
        )

        response = self.client.generate_content(
            formatted_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=3000,
            )
        )

        return response.text
