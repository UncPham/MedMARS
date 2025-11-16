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
    LLM_MODEL
)
from src.prompts.code_prompt import CODER_PROMPT, EXAMPLES_CODER


class CoderModel:
    def __init__(self, model_provider: str = LLM_MODEL):
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

    def __call__(self, plan: str):
        if self.model_provider == "openai":
            return self._call_openai(plan)
        elif self.model_provider == "gemini":
            return self._call_gemini(plan)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def _call_openai(self, plan: str) -> str:
        """Call Azure OpenAI API"""
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

    def _call_gemini(self, plan: str) -> str:
        """Call Google Gemini API"""
        # Format prompt with examples
        formatted_prompt = CODER_PROMPT.format(example=EXAMPLES_CODER)

        # Combine system prompt with user query
        full_prompt = f"{formatted_prompt}\n\n{plan}"

        response = self.client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=8000,
            )
        )

        # Handle response safely
        if not response.candidates:
            raise ValueError("No response candidates returned from Gemini API")

        candidate = response.candidates[0]

        # Check finish reason
        # 0: FINISH_REASON_UNSPECIFIED, 1: STOP (normal), 2: MAX_TOKENS, 3: SAFETY, 4: RECITATION, 5: OTHER
        if candidate.finish_reason == 1:  # STOP - normal completion
            return response.text
        elif candidate.finish_reason == 2:  # MAX_TOKENS
            # Try to return partial response if available
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
            raise ValueError("Response exceeded max tokens. Try increasing max_output_tokens.")
        elif candidate.finish_reason == 3:  # SAFETY
            raise ValueError("Response blocked by safety filters")
        elif candidate.finish_reason == 4:  # RECITATION
            raise ValueError("Response blocked due to recitation")
        else:
            raise ValueError(f"Unexpected finish_reason: {candidate.finish_reason}")