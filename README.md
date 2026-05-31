# Stock Recommendation Agent

An interactive multi-agent stock recommendation system for NSE India — built with **MCP (Model Context Protocol)**, **LangGraph**, and **Groq** (free tier). No paid APIs required.

## What It Does

Runs a 4-agent pipeline that picks NSE stocks, fetches live market data, analyses recent news, and gives you a Buy/Sell/Hold recommendation — with you in control at every step.

```
Phase 1  Auto-picks 2 promising NSE stocks
   |
   You decide what Phase 2 does
   |
Phase 2  Fetches live price, RSI, volume, moving averages
   |
   You decide what Phase 3 does
   |
Phase 3  Fetches recent news headlines + sentiment
   |
   You decide what Phase 4 does
   |
Phase 4  Gives Buy/Sell/Hold with target prices in INR
```

At each decision point you can press Enter (use previous phase's stocks), type your own tickers, or give any free-text instruction.

## Architecture

```
multi_agent_demo.py          LangGraph StateGraph (orchestrator)
      |
      |-- MCP Client (stdio transport)
      |         |
      |    stock_mcp_server.py   Custom MCP server
      |         |-- find_promising_nse_stocks()   curated NSE list
      |         |-- get_stock_market_data()        yfinance (live data)
      |         |-- get_stock_news()               Google News RSS
      |
      |-- 4 LangGraph Agents (Groq llama-3.3-70b)
               |-- stock_finder_agent
               |-- market_data_agent
               |-- news_analyst_agent
               |-- price_recommender_agent
```

## Tech Stack

| Layer | Technology | Cost |
|---|---|---|
| LLM | Groq llama-3.3-70b-versatile | Free tier |
| Stock data | yfinance (Yahoo Finance) | Free |
| News | Google News RSS | Free |
| MCP server | FastMCP | Open source |
| Agent framework | LangGraph + LangChain | Open source |
| MCP transport | stdio | Built-in |

## Setup

**Prerequisites:** Python 3.11+, Node.js (for npx)

**1. Clone the repo**
```bash
git clone https://github.com/moreshivam/stock-recommendation-agent.git
cd stock-recommendation-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

## Running

**Quick test (single agent):**
```bash
python main.py
```

**Full interactive pipeline:**
```bash
python multi_agent_demo.py
```

## Usage

When the pipeline runs, you will be prompted 3 times:

**After Phase 1:**
```
1. Press Enter      use the 2 stocks Phase 1 picked
2. Type tickers     TATAMOTORS.NS HDFCBANK.NS  (fetches ONLY these)
3. Free text        compare TCS and WIPRO
```

**After Phase 2:**
```
1. Press Enter      get news for same stocks
2. Type names       Reliance  Bajaj Finance
3. Free text        get news about Indian EV sector
```

**After Phase 3:**
```
1. Press Enter      AI decides Buy/Sell/Hold
2. Set price        buy Reliance below 1300, sell Infosys above 1250
3. Free text        which is safer for a 2-week hold?
```

## NSE Tickers to Test

**Large Cap**
```
RELIANCE.NS   INFY.NS   TCS.NS   HDFCBANK.NS
ICICIBANK.NS  SBIN.NS   WIPRO.NS  SUNPHARMA.NS
```

**Mid Cap**
```
PERSISTENT.NS  MPHASIS.NS   KPITTECH.NS   COFORGE.NS
MANKIND.NS     CHOLAFIN.NS  MUTHOOTFIN.NS ZOMATO.NS
BHARATFORG.NS  DIXON.NS     VBL.NS        PRESTIGE.NS
```

## Project Structure

```
stock-recommendation-agent/
|-- stock_mcp_server.py    Custom MCP server (tools: stock data + news)
|-- multi_agent_demo.py    Interactive 4-agent LangGraph pipeline
|-- main.py                Single agent test / MCP connection check
|-- requirements.txt       Python dependencies
|-- .env                   API keys (not committed)
|-- LICENSE
|-- README.md
```

## How MCP Works Here

MCP (Model Context Protocol) is the standard protocol that lets AI agents discover and call tools. Instead of using a paid MCP server (like Bright Data), we built our own:

1. `stock_mcp_server.py` starts as a subprocess
2. The MCP client sends a `tools/list` request
3. The server responds with all `@mcp.tool()` functions + their docstrings
4. LangGraph injects tool descriptions into the LLM's context
5. The LLM decides which tool to call based on the docstring
6. Client sends `tools/call`, server executes the function, returns result

## Notes

- Tested on Windows 11 with Python 3.14 (includes SSL fixes for Windows)
- Groq's llama model occasionally generates malformed tool calls — prompts are written to guard against this
- For more reliable tool calling, swap Groq for Claude (`langchain-anthropic`) or OpenAI (`langchain-openai`)

## License

MIT License — see [LICENSE](LICENSE)
