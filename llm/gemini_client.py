from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

def run_gemini(prompt_text):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_text
    )
    return response.text
