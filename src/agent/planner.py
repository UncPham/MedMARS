import os
import sys
import base64
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import AzureOpenAI
import google.generativeai as genai
from PIL import Image

from src.constants.env import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from src.prompts.planner_prompt import PLANNER_PROMPT, EXAMPLES_PLANNER


class PlannerModel:
    def __init__(self, model_provider: str = "openai"):
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

    def __call__(self, query: str, image_path: str = None, planning: str = ""):
        """
        Generate an analysis plan based on the user's medical query and optional image.

        Args:
            query: The user's medical imaging question
            image_path: Optional path to the medical image
            planning: Previous planning context (optional)

        Returns:
            Step-by-step analysis plan
        """
        if self.model_provider == "openai":
            content = self._call_openai(query, image_path, planning)
        elif self.model_provider == "gemini":
            content = self._call_gemini(query, image_path, planning)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        # Parse thought and plan from XML-like tags
        thought = ""
        plan = ""

        if "<thought>" in content and "</thought>" in content:
            thought_start = content.find("<thought>") + len("<thought>")
            thought_end = content.find("</thought>")
            thought = content[thought_start:thought_end].strip()

        if "<plan>" in content and "</plan>" in content:
            plan_start = content.find("<plan>") + len("<plan>")
            plan_end = content.find("</plan>")
            plan = content[plan_start:plan_end].strip()

        return thought, plan

    def _call_openai(self, query: str, image_path: str = None, planning: str = "") -> str:
        """Call Azure OpenAI API"""
        # Format prompt with examples and planning
        formatted_prompt = PLANNER_PROMPT.format(
            examples=EXAMPLES_PLANNER,
            planning=planning
        )

        messages = [
            {
                "role": "system",
                "content": formatted_prompt
            }
        ]

        # Build user message with image if provided
        if image_path and Path(image_path).exists():
            # Determine image type
            mime_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            image_extension = Path(image_path).suffix.lower()
            mime_type = mime_type_map.get(image_extension, "image/jpeg")

            # Encode image
            base64_image = self._encode_image(image_path)

            # Add user message with text and image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
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
            max_tokens=1500
        )

        return response.choices[0].message.content

    def _call_gemini(self, query: str, image_path: str = None, planning: str = "") -> str:
        """Call Google Gemini API"""
        # Format prompt with examples and planning
        formatted_prompt = PLANNER_PROMPT.format(
            examples=EXAMPLES_PLANNER,
            planning=planning
        )

        # Combine system prompt with user query
        full_prompt = f"{formatted_prompt}\n\n{query}"

        # Build content parts
        content_parts = [full_prompt]

        # Add image if provided
        if image_path and Path(image_path).exists():
            image = Image.open(image_path)
            content_parts.append(image)

        response = self.client.generate_content(
            content_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500,
            )
        )

        return response.text
    
if __name__ == "__main__":
    planner = PlannerModel(model_provider="openai")
    thought, plan = planner("Is there pneumonia in this chest X-ray?", image_path="/Users/uncpham/Repo/Medical-Assistant/src/data/vindr_cxr_vqa/images/0a0ac65c40a9ac441651e4bfbde03c4e.jpg")
    print("Thought:", thought)
    print("Plan:", plan)