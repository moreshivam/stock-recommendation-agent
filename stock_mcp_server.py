import yfinance as yf
import feedparser
from mcp.server.fastmcp import FastMCP
from curl_cffi import requests as cffi_requests

# FastMCP registers this process as an MCP server named "stock-server".
# When the client runs this file as a subprocess, it sends a tools/list request
# and the server responds with every function decorated with @mcp.tool() below —
# including its name, docstring, and parameter types. The LLM uses those
# docstrings to decide which tool to call for a given query.
mcp = FastMCP("stock-server")

# Shared HTTP session with SSL verification disabled (Windows cert chain issue)
# and Chrome impersonation so Yahoo Finance / Google News don't block us.
_session = cffi_requests.Session(verify=False, impersonate="chrome")


def compute_rsi(closes, period=14):
    """RSI calculation using Wilder's smoothing — no external TA library needed."""
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Stock Discovery
# The LLM will call this tool when it needs to decide WHICH stocks to analyse.
# It returns a curated list of 10 liquid, actively traded NSE stocks.
#
# NOTE: This list is hardcoded for demo purposes. In production you could
# replace this with a live screener (e.g. yfinance sector scan, NSE API)
# to dynamically pick top gainers / most active stocks of the day.
# The other two tools (market data + news) work for ANY NSE ticker — they
# are not limited to this list.
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def find_promising_nse_stocks() -> str:
    """
    Returns a curated list of actively traded NSE-listed stocks for short-term trading research.
    This function takes NO arguments — call it with empty parentheses: find_promising_nse_stocks()
    Use this tool first to get the list, then select 2 stocks from the result.
    """
    stocks = [
        {"ticker": "RELIANCE.NS",  "name": "Reliance Industries", "sector": "Conglomerate"},
        {"ticker": "INFY.NS",      "name": "Infosys",             "sector": "IT"},
        {"ticker": "TCS.NS",       "name": "Tata Consultancy",    "sector": "IT"},
        {"ticker": "HDFCBANK.NS",  "name": "HDFC Bank",           "sector": "Banking"},
        {"ticker": "ICICIBANK.NS", "name": "ICICI Bank",          "sector": "Banking"},
        {"ticker": "BAJFINANCE.NS","name": "Bajaj Finance",       "sector": "NBFC"},
        {"ticker": "WIPRO.NS",     "name": "Wipro",               "sector": "IT"},
        {"ticker": "TATAMOTORS.NS","name": "Tata Motors",         "sector": "Auto"},
        {"ticker": "ADANIENT.NS",  "name": "Adani Enterprises",   "sector": "Conglomerate"},
        {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma",          "sector": "Pharma"},
    ]
    lines = ["Actively traded NSE stocks for short-term research:\n"]
    for s in stocks:
        lines.append(f"- {s['name']} | Ticker: {s['ticker']} | Sector: {s['sector']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Live Market Data
# The LLM will call this tool when it has a specific ticker and needs numbers:
# price, volume, RSI, moving averages. Works for ANY NSE ticker (not just the
# ones in find_promising_nse_stocks). Always append ".NS" for NSE stocks.
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_stock_market_data(ticker: str) -> str:
    """
    Fetches live market data for any NSE stock ticker (e.g. RELIANCE.NS, INFY.NS).
    Returns current price, previous close, today's volume, 7-day and 30-day
    price change, RSI(14), 50-day and 200-day moving averages, company name,
    and sector. Always use the NSE ticker format: SYMBOL.NS
    """
    try:
        stock = yf.Ticker(ticker.upper().replace(" ", ""), session=_session)
        info  = stock.info
        hist  = stock.history(period="60d")

        if hist.empty:
            return f"No data found for {ticker}. Check the ticker symbol."

        closes  = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()

        current_price = round(closes[-1], 2)
        prev_close    = round(closes[-2], 2) if len(closes) > 1 else current_price
        today_volume  = int(volumes[-1]) if volumes else 0

        change_7d  = round(((closes[-1] - closes[-8])  / closes[-8])  * 100, 2) if len(closes) >= 8  else None
        change_30d = round(((closes[-1] - closes[-31]) / closes[-31]) * 100, 2) if len(closes) >= 31 else None

        ma50  = round(sum(closes[-50:]) / min(50, len(closes)), 2)
        ma200 = round(sum(closes) / len(closes), 2)
        rsi   = compute_rsi(closes)

        return (
            f"Market Data for {ticker}\n"
            f"  Current Price   : INR {current_price}\n"
            f"  Previous Close  : INR {prev_close}\n"
            f"  Today's Volume  : {today_volume:,}\n"
            f"  7-Day Change    : {change_7d}%\n"
            f"  30-Day Change   : {change_30d}%\n"
            f"  RSI (14)        : {rsi}\n"
            f"  50-Day MA       : INR {ma50}\n"
            f"  200-Day MA (est): INR {ma200}\n"
            f"  Company         : {info.get('longName', ticker)}\n"
            f"  Sector          : {info.get('sector', 'N/A')}\n"
        )
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Recent News
# The LLM calls this when it needs sentiment/news context for a stock.
# Fetches from Google News RSS (free, no API key). Falls back to Yahoo Finance
# RSS if Google News returns nothing. Works for any company name or ticker.
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_stock_news(query: str) -> str:
    """
    Fetches the 5 most recent news headlines for a stock using its company name
    or NSE ticker (e.g. 'Reliance Industries' or 'RELIANCE.NS'). Use this tool
    to understand recent sentiment, announcements, or events affecting a stock.
    """
    try:
        import re
        search_term = query.strip().replace(".NS", "").replace(".ns", "").replace(".BO", "").replace(".bo", "").replace(" ", "+")

        url  = f"https://news.google.com/rss/search?q={search_term}+NSE+stock+India&hl=en-IN&gl=IN&ceid=IN:en"
        resp = _session.get(url)
        feed = feedparser.parse(resp.text)

        if not feed.entries:
            url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={search_term}.NS&region=IN&lang=en-IN"
            resp = _session.get(url)
            feed = feedparser.parse(resp.text)

        if not feed.entries:
            return f"No recent news found for '{query}'."

        lines = [f"Recent news for {query}:\n"]
        for i, entry in enumerate(feed.entries[:5], 1):
            title     = entry.get("title", "No title")
            published = entry.get("published", "Unknown date")
            summary   = re.sub(r"<[^>]+>", "", entry.get("summary", ""))[:200].strip()
            lines.append(f"{i}. {title}")
            lines.append(f"   Published : {published}")
            if summary:
                lines.append(f"   Summary   : {summary}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching news for '{query}': {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — runs the MCP server over stdio transport.
# The client (MultiServerMCPClient) starts this file as a subprocess and
# communicates via stdin/stdout using the MCP JSON-RPC protocol.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
