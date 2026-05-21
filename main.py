import os
import base64
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def estimate_cost(uploaded_file, location):
    """
    Predict food name and approximate home-made cost for one person
    based on uploaded food image and location.
    """
    ext = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.read()
    b64_image = base64.b64encode(file_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI that identifies food from images and estimates "
                    "the cost of making it at home for one person. "
                    "Always return valid JSON with keys: food_name, cost_usd. "
                    "The cost must be a number in US Dollars (USD)."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Location: {location}\nIdentify the food and estimate cost."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{ext[1:]};base64,{b64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=150
    )

    result = response.choices[0].message.content.strip()

    # Try to parse JSON
    import json
    try:
        data = json.loads(result)
        food_name = data.get("food_name", "Unknown")
        cost = data.get("cost_bdt", "N/A")
    except:
        food_name = "Unknown"
        cost = result

    return food_name, cost

