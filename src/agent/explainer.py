import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import AzureOpenAI
import base64
from pathlib import Path

from src.constants.env import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
)
from src.prompts.explainer_prompt import EXPLAINER_PROMPT


class Explainer:
    def __init__(self,):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.deployment_name = AZURE_OPENAI_DEPLOYMENT,


    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def __call__(self, query: str, list_image: list = None) -> str:
        # Add medical context to avoid content filter issues
        medical_disclaimer = (
            "IMPORTANT: This is a medical/healthcare professional analysis task. "
            "Any images provided are medical images (X-rays, CT scans, MRI, pathology slides, etc.) "
            "for legitimate diagnostic and educational purposes. "
            "Please analyze them in a clinical context.\n\n"
        )

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
            content = [{"type": "text", "text": medical_disclaimer + query}]

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

        # Call Azure OpenAI API with error handling for content filter
        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content

        except Exception as e:
            error_message = str(e)
            if "content_filter" in error_message or "ResponsibleAIPolicyViolation" in error_message:
                # Content filter triggered - return a safe fallback
                print(f"Warning: Content filter triggered for images. Using text-only fallback.")
                # Retry without images
                fallback_messages = [
                    {"role": "system", "content": EXPLAINER_PROMPT},
                    {"role": "user", "content": f"{query}\n\nNote: Image analysis was blocked by content policy. Please provide a text-based explanation."}
                ]
                response = self.client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=fallback_messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            else:
                # Other error - re-raise
                raise e

if __name__ == "__main__":
    explainer = Explainer()
    explanation = explainer(
        "The patient has a cough and difficulty breathing.",
    )
    print("Explanation:", explanation)