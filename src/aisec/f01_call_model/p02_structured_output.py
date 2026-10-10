from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-6.1-sol"
instructions = "You are an order assistant for an Indian broker that reads the investor's message and states the order: stock, buy or sell, quantity, order type and price."

class Order(BaseModel):
    stock: str
    buy_or_sell: Literal["buy", "sell"]
    quantity: int
    order_type: Literal["market", "limit"]
    price: float

def call_model(instructions, prompt):
    response = client.responses.parse(
        model=MODEL,
        instructions=instructions,
        input=prompt,
        text_format=Order
    )
    return response.output_parsed

if __name__ == "__main__":
    prompts = [
        "buy 10 Reliance at 2900 limit",
        "buy Reliance worth 50000",
        "sell all my Infosys",
    ]
    for prompt in prompts:
        order = call_model(instructions, prompt)
        if order is None:
            print(prompt, "->", "No order found")
        else:
            print(prompt, "->", order.model_dump_json())