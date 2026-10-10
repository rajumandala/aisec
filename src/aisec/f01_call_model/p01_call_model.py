from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-6.1-sol"
instructions = "You are an order assistant for an Indian broker that reads the investor's message and states the order: stock, buy or sell, quantity, order type and price."

def call_model(instructions, prompt):
    response = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=prompt
    )
    return response.output_text

if __name__ == "__main__":
    print(call_model(instructions, "buy Reliance worth 50000"))