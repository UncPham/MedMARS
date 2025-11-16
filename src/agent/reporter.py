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
from src.prompts.reporter_prompt import REPORTER_PROMPT


class Reporter:
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

    def __call__(self, query: str, output: str, code: str) -> dict:
        if self.model_provider == "openai":
            content = self._call_openai(query, output, code)
        elif self.model_provider == "gemini":
            content = self._call_gemini(query, output, code)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        # Parse answer and reason/explanation from XML-like tags
        answer = ""
        reason = ""

        if "<answer>" in content and "</answer>" in content:
            answer_start = content.find("<answer>") + len("<answer>")
            answer_end = content.find("</answer>")
            answer = content[answer_start:answer_end].strip()

        # Try to parse <reason> first (new format), then fall back to <explanation> (old format)
        if "<reason>" in content and "</reason>" in content:
            reason_start = content.find("<reason>") + len("<reason>")
            reason_end = content.find("</reason>")
            reason = content[reason_start:reason_end].strip()
        elif "<explanation>" in content and "</explanation>" in content:
            reason_start = content.find("<explanation>") + len("<explanation>")
            reason_end = content.find("</explanation>")
            reason = content[reason_start:reason_end].strip()

        return {
            "answer": answer,
            "reason": reason,
            "explanation": reason,  # Backward compatibility
            "report": content  # Full content for backward compatibility
        }

    def _call_openai(self, query: str, output: str, code: str) -> str:
        """Call Azure OpenAI API"""
        # Format the prompt with query, plan, and output
        formatted_prompt = REPORTER_PROMPT.format(
            query=query,
            code=code,
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

    def _call_gemini(self, query: str, output: str, code: str) -> str:
        """Call Google Gemini API"""
        # Format the prompt with query, plan, and output
        formatted_prompt = REPORTER_PROMPT.format(
            query=query,
            code=code,
            output=output
        )

        response = self.client.generate_content(
            formatted_prompt,
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
