from mcp.server import MCPServer

mcp = MCPServer("Market Data Vendor")

QUOTES = {"RELIANCE": 2905.50, "INFY": 1480.00, "TCS": 3105.00, "HDFCBANK": 1650.00}  # the vendor's prices (made up)

@mcp.tool()
def get_quote(stock: str) -> float | None:
    """Get the last traded price of an NSE stock, such as RELIANCE."""
    return QUOTES.get(stock)

if __name__ == "__main__":
    mcp.run()   # stdio: our app starts this file and talks to it over stdin and stdout
