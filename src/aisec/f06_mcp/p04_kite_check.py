import asyncio
from mcp import Client

# Zerodha's hosted Kite MCP server. No model and no login: we only review what the server publishes.
KITE = "https://mcp.kite.trade/mcp"

async def main():
    async with Client(KITE, read_timeout_seconds=60) as kite:
        print("Server:", kite.server_info.name, kite.server_info.version)
        tools = (await kite.list_tools()).tools
        print(f"{len(tools)} tools published:")
        for t in tools:
            hints = t.annotations
            print(f"  {t.name:22} read_only={hints.read_only_hint} destructive={hints.destructive_hint}  {t.description.splitlines()[0][:70]}")

asyncio.run(main())
