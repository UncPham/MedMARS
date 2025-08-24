from google import genai
from constants.env import GEMINI_API_KEY

class GeminiModel:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def run(self, query: str):
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", contents=query
        )
        return response.text