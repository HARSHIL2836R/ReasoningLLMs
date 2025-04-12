import os
from openai import OpenAI

# Get Together.ai API key
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")  # Set this in your environment
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

client = OpenAI(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)

response = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Or any other supported model
    messages=[
        {
            "role": "user",
            "content": "What is "
        }
    ],
    temperature=0.2,
    extra_body={"optillm_approach": "self_consistency"}
)

print(response.choices[0].message.content)
