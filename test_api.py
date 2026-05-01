import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

print(f"Key loaded: {API_KEY[:10]}... (hidden for security)")

try:
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='If you can hear me, say "Loud and clear detective!"'
    )
    print("API Response:", response.text)
except Exception as e:
    print("API Error:", e)