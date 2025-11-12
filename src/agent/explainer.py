import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import AzureOpenAI
import google.generativeai as genai
import base64
from pathlib import Path
from PIL import Image

from src.constants.env import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from src.prompts.explainer_prompt import EXPLAINER_PROMPT


class Explainer:
    def __init__(self, model_provider: str = "openai"):
        """
        Initialize Explainer with specified model provider.

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


    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def __call__(self, query: str, list_image: list = None) -> str:
        """
        Generate explanation based on query and images.

        Args:
            query: The user's question
            list_image: Optional list of image paths

        Returns:
            Explanation text
        """
        if self.model_provider == "openai":
            return self._call_openai(query, list_image)
        elif self.model_provider == "gemini":
            return self._call_gemini(query, list_image)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def _call_openai(self, query: str, list_image: list = None) -> str:
        """Call Azure OpenAI API"""
        messages = [
            {"role": "system", "content": EXPLAINER_PROMPT}
        ]

        mime_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }

        # Build user message content
        if list_image and len(list_image) > 0:
            # Start with text content and medical disclaimer
            content = [{"type": "text", "text": query}]

            # Add all images to content
            for image_path in list_image:
                if Path(image_path).exists():
                    # Get image extension to determine type
                    image_extension = Path(image_path).suffix.lower()
                    mime_type = mime_type_map.get(image_extension, "image/jpeg")

                    # Encode image
                    base64_image = self._encode_image(image_path)

                    # Add image to content
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })

            # Add user message with text and images
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            # Add user message without image
            messages.append({
                "role": "user",
                "content": query
            })

        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content

    def _call_gemini(self, query: str, list_image: list = None) -> str:
        """Call Google Gemini API"""
        # Combine system prompt with user query
        full_prompt = f"{EXPLAINER_PROMPT}\n\n{query}"

        # Build content parts
        content_parts = [full_prompt]

        # Add images if provided
        if list_image and len(list_image) > 0:
            for image_path in list_image:
                if Path(image_path).exists():
                    image = Image.open(image_path)
                    content_parts.append(image)

        response = self.client.generate_content(
            content_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
            )
        )

        return response.text