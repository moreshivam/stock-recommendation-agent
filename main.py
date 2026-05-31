import os
import asyncio
import truststore
truststore.inject_into_ssl()  # makes Python use Windows system cert store — fixes SSL on Windows
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()

async def run_agent():
    # Connect to our MCP server (stock_mcp_server.py runs as a subprocess).
    # The client will auto-discover all 3 tools via tools/list handshake.
    client = MultiServerMCPClient(
        {
            "stock_server": {
                "command": "python",
                "args": ["stock_mcp_server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()

    print(f"Tools discovered from MCP server: {[t.name for t in tools]}\n")

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    agent = create_agent(
        model,
        tools,
        system_prompt="You are a stock research assistant with access to live NSE market data tools.",
    )

    response = await agent.ainvoke({
        "messages": "What is the current price and RSI of Infosys stock?"
    })

    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run_agent())
