from google import genai
from google.genai import types

from constants.env import GEMINI_API_KEY
from prompts.code_prompt import code_prompt

class CodeModel:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def run(self, query: str, image_path: str):
        contents = [
            types.Content(
                role="model", 
                parts=[types.Part(text=code_prompt)]
            ),
            types.Content(
                role="user", 
                parts=[types.Part(text=query)]
            ),
            types.Content(
                role="user",
                parts=[types.Part(text=image_path)]
            )
        ]
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=contents
        )
        return response.text