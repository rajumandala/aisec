from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-6.1-sol"
instructions = (
    "You are an order assistant for an Indian broker that reads the investor's message and states the order: "
    "stock, buy or sell, quantity, order type and price. "
    "For stock, use the NSE trading symbol, such as RELIANCE or INFY."
)
INSTRUMENTS = {"RELIANCE", "INFY", "TCS", "HDFCBANK"}   # our instrument master (made up)
MAX_ORDER_VALUE = 500000

class Order(BaseModel):
    stock: str
    buy_or_sell: Literal["buy", "sell"]
    quantity: int
    order_type: Literal["market", "limit"]
    price: float

def check_order(order: Order) -> list[str]:
    problems = []
    if order.stock not in INSTRUMENTS:
        problems.append(f"{order.stock} is not in the instrument master")
    if order.quantity <= 0:
        problems.append("quantity must be more than 0")
    if order.order_type != "limit":
        problems.append("the assistant may only create limit orders")
    if order.price <= 0:
        problems.append("price must be more than 0")
    if order.quantity * order.price > MAX_ORDER_VALUE:
        problems.append(f"order value is above the limit of ₹{MAX_ORDER_VALUE:,}")
    return problems

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
        "buy 1000 TCS at 3100 limit",
    ]
    for prompt in prompts:
        order = call_model(instructions, prompt)
        if order is None:
            print(prompt, "->", "No order found")
            continue
        problems = check_order(order)
        decision = "ACCEPTED" if not problems else "REJECTED: " + "; ".join(problems)
        print(prompt, "->", order.model_dump_json(), "->", decision)