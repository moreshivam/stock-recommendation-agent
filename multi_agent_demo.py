import os
import asyncio
import time
import truststore
truststore.inject_into_ssl()
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ── Shared State ─────────────────────────────────────────────────────────────
# This is the single source of truth passed between every node in the graph.
# `messages` accumulates the full conversation history across all agents.
# The instruction fields carry user decisions from "ask" nodes into agent nodes.
class State(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    p2_instruction: str
    p2_is_custom:   bool   # True = user gave their own stocks, False = use Phase 1's picks
    p3_instruction: str
    p3_is_custom:   bool   # True = user gave their own stocks, False = use Phase 2's stocks
    p4_instruction: str


def divider(title: str = ""):
    if title:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")
    else:
        print(f"\n{'-'*60}")


def last_ai_message(messages: list) -> str:
    """Return content of the last AIMessage in a list."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return ""


async def build_and_run():

    # ── MCP + Model setup ────────────────────────────────────────────────────
    client = MultiServerMCPClient({
        "stock_server": {
            "command": "python",
            "args": ["stock_mcp_server.py"],
            "transport": "stdio",
        }
    })
    all_tools = await client.get_tools()
    tool_map  = {t.name: t for t in all_tools}

    # Groq requires sequential tool calls — bind this to avoid validation errors
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
    ).bind(parallel_tool_calls=False)

    # ── Agents ───────────────────────────────────────────────────────────────
    stock_finder_agent = create_agent(
        model, [tool_map["find_promising_nse_stocks"]],
        system_prompt="""You are a stock research analyst for NSE India.
You have exactly ONE tool available: find_promising_nse_stocks(). Do not invent or call any other tool.

Instructions:
1. Call find_promising_nse_stocks() with NO arguments.
2. Read the returned list.
3. Write your response as plain text — pick 2 stocks, give their name, ticker, and a one-line reason each.
   Do NOT call any tool for step 3. Just write the answer.""",
        name="stock_finder_agent",
    )
    market_data_agent = create_agent(
        model, [tool_map["get_stock_market_data"]],
        system_prompt="You are a market data analyst for NSE stocks. Call get_stock_market_data() once per ticker. Report: price, volume, 7d/30d change, RSI, 50d/200d MA in INR.",
        name="market_data_agent",
    )
    news_analyst_agent = create_agent(
        model, [tool_map["get_stock_news"]],
        system_prompt="You are a financial news analyst for NSE stocks. Call get_stock_news() once per stock. Summarise headlines, classify sentiment, note short-term price impact.",
        name="news_analyst_agent",
    )
    price_recommender_agent = create_agent(
        model, [],
        system_prompt="""You are a trading strategy advisor for NSE India.
You have NO tools. Do not call any function or tool. Do not generate any function call.
Read the market data and news from the conversation history provided to you.
Write your response as plain text only.

For each stock in the conversation:
1. Action: Buy / Sell / Hold
2. Target price in INR
3. One-line reason

If the user specified a price constraint (e.g. "below 1000"), honour it in your answer.
Just write the answer directly. No tool calls. No function calls. Plain text only.""",
        name="price_recommender_agent",
    )

    # ── Graph Nodes ──────────────────────────────────────────────────────────
    # Each agent node runs its agent and writes results back to shared State.
    # Each "ask" node prints options and calls interrupt() — graph pauses here
    # until the user resumes with Command(resume="their input").

    async def stock_finder_node(state: State):
        divider("PHASE 1 / 4 - Stock Finder Agent")
        result = await stock_finder_agent.ainvoke({
            "messages": [HumanMessage(content="Find 2 promising NSE stocks for short-term trading.")]
        })
        print(last_ai_message(result["messages"]))
        # Write agent messages into shared state
        return {"messages": result["messages"]}

    async def ask_p2_node(_state: State):
        divider("YOUR TURN - What should Phase 2 do?")
        print("  1. Enter        = market data for Phase 1 stocks only")
        print("  2. Tickers      = e.g.  TATAMOTORS.NS HDFCBANK.NS  (fetches ONLY these)")
        print("  3. Free text    = e.g.  'get data for TCS and WIPRO'  (fetches ONLY these)")
        user_input = interrupt("phase_2_decision")
        if user_input:
            return {"p2_instruction": user_input, "p2_is_custom": True}
        else:
            return {
                "p2_instruction": "Fetch live market data for the 2 stocks above. Call the tool once per stock.",
                "p2_is_custom": False,
            }

    async def market_data_node(state: State):
        divider("PHASE 2 / 4 - Market Data Agent")
        print(f"Instruction: {state['p2_instruction']}\n")

        if state["p2_is_custom"]:
            # User gave specific stocks — fresh context, no Phase 1 history
            messages = [HumanMessage(content=state["p2_instruction"])]
        else:
            # User pressed Enter — pass only the Phase 1 final summary (not full
            # message history) to keep context small and prevent malformed tool calls
            phase1_summary = last_ai_message(state["messages"])
            messages = [HumanMessage(
                content=f"Phase 1 identified these stocks:\n{phase1_summary}\n\n{state['p2_instruction']}"
            )]

        result = await market_data_agent.ainvoke({"messages": messages})
        print(last_ai_message(result["messages"]))
        return {"messages": result["messages"]}

    async def ask_p3_node(_state: State):
        divider("YOUR TURN - What should Phase 3 do?")
        print("  1. Enter        = news for Phase 2 stocks only")
        print("  2. Names        = e.g.  Reliance  Bajaj Finance  (fetches ONLY these)")
        print("  3. Free text    = e.g.  'get news about Indian EV sector'  (fetches ONLY these)")
        user_input = interrupt("phase_3_decision")
        if user_input:
            return {"p3_instruction": user_input, "p3_is_custom": True}
        else:
            return {
                "p3_instruction": "Get the latest news for the stocks discussed above. Call the tool once per stock.",
                "p3_is_custom": False,
            }

    async def news_analyst_node(state: State):
        divider("PHASE 3 / 4 - News Analyst Agent")
        print(f"Instruction: {state['p3_instruction']}\n")

        if state["p3_is_custom"]:
            # User gave specific stocks — fresh context, no prior history
            messages = [HumanMessage(content=state["p3_instruction"])]
        else:
            # User pressed Enter — pass only Phase 2 final summary, not full history
            phase2_summary = last_ai_message(state["messages"])
            messages = [HumanMessage(
                content=f"Phase 2 market data summary:\n{phase2_summary}\n\n{state['p3_instruction']}"
            )]

        result = await news_analyst_agent.ainvoke({"messages": messages})
        print(last_ai_message(result["messages"]))
        return {"messages": result["messages"]}

    async def ask_p4_node(_state: State):
        divider("YOUR TURN - What should Phase 4 do?")
        print("  1. Enter        = AI decides Buy/Sell/Hold with target prices")
        print("  2. Set prices   = e.g.  'buy Reliance below 1300, sell Infosys above 1250'")
        print("  3. Free text    = e.g.  'which is safer for a 2-week hold?'")
        user_input = interrupt("phase_4_decision")
        instruction = user_input if user_input else \
            "Based on all market data and news above, give Buy/Sell/Hold with target prices in INR."
        return {"p4_instruction": instruction}

    async def recommender_node(state: State):
        divider("PHASE 4 / 4 - Price Recommender Agent")
        print(f"Instruction: {state['p4_instruction']}\n")
        result = await price_recommender_agent.ainvoke({
            "messages": state["messages"] + [HumanMessage(content=state["p4_instruction"])]
        })
        divider("FINAL RECOMMENDATION")
        print(result["messages"][-1].content)
        return {"messages": result["messages"]}

    # ── Build LangGraph StateGraph ────────────────────────────────────────────
    #
    #  START
    #    │
    #    ▼
    #  stock_finder  → ask_p2 ──(interrupt)──► market_data
    #                                               │
    #                            ask_p3 ◄───────────┘
    #                              │
    #                         (interrupt)
    #                              │
    #                              ▼
    #                         news_analyst → ask_p4 ──(interrupt)──► recommender → END
    #
    workflow = StateGraph(State)

    workflow.add_node("stock_finder", stock_finder_node)
    workflow.add_node("ask_p2",       ask_p2_node)
    workflow.add_node("market_data",  market_data_node)
    workflow.add_node("ask_p3",       ask_p3_node)
    workflow.add_node("news_analyst", news_analyst_node)
    workflow.add_node("ask_p4",       ask_p4_node)
    workflow.add_node("recommender",  recommender_node)

    workflow.add_edge(START,          "stock_finder")
    workflow.add_edge("stock_finder", "ask_p2")
    workflow.add_edge("ask_p2",       "market_data")
    workflow.add_edge("market_data",  "ask_p3")
    workflow.add_edge("ask_p3",       "news_analyst")
    workflow.add_edge("news_analyst", "ask_p4")
    workflow.add_edge("ask_p4",       "recommender")
    workflow.add_edge("recommender",  END)

    # MemorySaver checkpoints state after every node — if graph is interrupted,
    # state is preserved and the run resumes exactly where it left off.
    graph = workflow.compile(checkpointer=MemorySaver())

    # ── Run loop — handles interrupts automatically ────────────────────────
    # thread_id scopes the checkpoint so multiple runs don't share state.
    config     = {"configurable": {"thread_id": f"stock_session_{int(time.time())}"}}
    input_data = {
        "messages":       [],
        "p2_instruction": "",
        "p2_is_custom":   False,
        "p3_instruction": "",
        "p3_is_custom":   False,
        "p4_instruction": "",
    }

    while True:
        interrupted = False

        async for chunk in graph.astream(input_data, config, stream_mode="updates"):

            if "__interrupt__" in chunk:
                # Graph paused at an interrupt() call — ask user and resume
                interrupt_value = chunk["__interrupt__"][0].value
                print(f"\n>> Decision point: {interrupt_value}")
                user_answer = input("   Your input (Enter to skip): ").strip()
                input_data  = Command(resume=user_answer)
                interrupted = True
                break   # exit inner loop → re-enter while loop to resume

        if not interrupted:
            break   # no interrupt hit → graph ran to END


if __name__ == "__main__":
    asyncio.run(build_and_run())
