import requests
import os

HF_KEY = os.getenv("HF_API_KEY_1")
GEMINI_KEY = os.getenv("GEMINI_API_KEY_1")

async def run_sentiment():
    # Misol: HuggingFace bilan
    headers = {"Authorization": f"Bearer {HF_KEY}"}
    data = {"inputs": "Bitcoin ETF yangiliklari juda ijobiy."}
    response = requests.post(
        "https://api-inference.huggingface.co/models/distilbert-base-uncased",
        headers=headers,
        json=data
    )
    result = response.json()
    print(result)
