from google import genai
from google.genai import types

from constants.env import GEMINI_API_KEY
from prompts.planner_prompt import planer_prompt


class GeminiModel:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    async def __call__(self, query: str):
        contents = [
            types.Content(
                role="model", 
                parts=[types.Part(text=planer_prompt)]
            ),
            types.Content(
                role="user", 
                parts=[types.Part(text=query)]
            )
        ]
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=contents
        )
        return response.text