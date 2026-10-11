from openai import OpenAI
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
import os
import json 

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-6.1-sol"
instructions = (
    "You are an order assistant for an Indian broker that reads the investor's message and states the order: "
    "stock, buy or sell, quantity, order type and price. "
    "For stock, use the NSE trading symbol, such as RELIANCE or INFY. "
    "If the investor gives an amount instead of a quantity, call get_quote and buy as many whole shares as the amount allows, "
    "as a limit order at the quoted price."
)
INSTRUMENTS = {"RELIANCE", "INFY", "TCS", "HDFCBANK"}   # our instrument master (made up)
MAX_ORDER_VALUE = 500000
PRICE_BAND = 0.05  # limit price must be within 5% of the live price

QUOTES = {"RELIANCE": 2905.50, "INFY": 1480.00, "TCS": 3105.00, "HDFCBANK": 1650.00}  # made up; a real app calls the broker's quote API

def get_quote(stock: str) -> float | None:
    return QUOTES.get(stock)

tools = [{
    "type": "function",
    "name": "get_quote",
    "description": "Get the last traded price of an NSE stock.",
    "parameters": {
        "type": "object",
        "properties": {"stock": {"type": "string", "description": "NSE trading symbol, such as RELIANCE"}},
        "required": ["stock"],
        "additionalProperties": False,
    },
    "strict": True,
},
]

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
    live_price = get_quote(order.stock)
    if live_price is not None and abs(order.price - live_price) > live_price * PRICE_BAND:
        problems.append(f"price {order.price} is more than {PRICE_BAND:.0%} away from the live price {live_price}")
    return problems

def call_model(instructions, prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client.responses.parse(
        model=MODEL, instructions=instructions, input=messages, tools=tools, text_format=Order
    )
    calls = [item for item in response.output if item.type == "function_call"]
    if not calls:
        return response.output_parsed
    messages += response.output                      # keep the model's request in the conversation
    for call in calls:
        if call.name == "get_quote":                 # code decides which function runs
            args = json.loads(call.arguments)
            result = get_quote(args["stock"])
        else:
            result = "unknown tool"
        messages.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)})
    response = client.responses.parse(
        model=MODEL, instructions=instructions, input=messages, tools=tools, text_format=Order
    )
    return response.output_parsed

if __name__ == "__main__":
    prompts = [
        "buy 10 Reliance at 2900 limit",
        "buy Reliance worth 50000",
        "sell all my Infosys",
        "buy 1000 TCS at 3100 limit",
        "buy 10 Reliance at 29000 limit",
    ]
    for prompt in prompts:
        order = call_model(instructions, prompt)
        if order is None:
            print(prompt, "->", "No order found")
            continue
        problems = check_order(order)
        decision = "ACCEPTED" if not problems else "REJECTED: " + "; ".join(problems)
        print(prompt, "->", order.model_dump_json(), "->", decision)
