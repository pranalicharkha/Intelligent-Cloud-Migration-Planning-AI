import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="novita",
    api_key=os.getenv("HF_TOKEN")
)

question = input("Ask a migration question: ")

messages = [
    {
        "role": "system",
        "content": "You are an AI Copilot for cloud migration. Answer migration questions clearly and concisely."
    },
    {
        "role": "user",
        "content": question
    }
]

response = client.chat_completion(
    messages=messages,
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=300,
    temperature=0.2
)

print("\nCopilot answer:\n")
print(response.choices[0].message.content)