import os
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Load env variables at the top
load_dotenv()

SYSTEM_PROMPT = (
    "You have access to tools. When you choose to call a tool, do not narrate status updates. "
    "After tools run, return only a concise final answer."
)

st.set_page_config(page_title="MCP Chat", page_icon="🧰", layout="centered")
st.title("🧰 MCP Chat")

SERVERS = { 
    # "math": {
    #     "transport": "stdio",
    #     "command": "uv",
    #     "args": [
    #         "run",
    #         "fastmcp",
    #         "run",
    #         "/Users/nitish/Desktop/mcp-math-server/main.py"
    #    ]
    # },
    # "manim-server": {
    #     "transport": "stdio",
    #     "command": "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
    #     "args": [
    #     "/Users/nitish/desktop/manim-mcp-server/src/manim_server.py"
    #   ],
    #     "env": {
    #     "MANIM_EXECUTABLE": "/Library/Frameworks/Python.framework/Versions/3.11/bin/manim"
    #   }
    # },
    "expense": {
        "transport": "streamable_http",  # if this fails, try "sse"
        "url": "https://sai-expense-tracker.fastmcp.app/mcp",
        "headers": {
            "Authorization": f"Bearer {os.getenv('HORIZON_API_KEY')}"
        }
    }
}

# Safely run async functions within Streamlit
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If an event loop is already active in current thread
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        # Standard execution if no loop is running
        return asyncio.run(coro)

# One-time initialization
if "initialized" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o")

    # 1) Connect MCP Client and await get_tools() safely
    st.session_state.client = MultiServerMCPClient(SERVERS)
    tools = run_async(st.session_state.client.get_tools())
    st.session_state.tools = tools
    st.session_state.tool_by_name = {t.name: t for t in tools}

    # 2) Bind tools
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    # 3) Conversation state
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.initialized = True

# Render chat history (hide intermediate tool steps)
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input processing
user_text = st.chat_input("Type a message…")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.history.append(HumanMessage(content=user_text))

    # First pass: Ask LLM if it wants to invoke tools
    first = run_async(st.session_state.llm_with_tools.ainvoke(st.session_state.history))
    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        with st.chat_message("assistant"):
            st.markdown(first.content or "")
        st.session_state.history.append(first)
    else:
        st.session_state.history.append(first)

        tool_msgs = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            
            tool = st.session_state.tool_by_name[name]
            
            # Execute tool safely
            res = run_async(tool.ainvoke(args))
            tool_msgs.append(ToolMessage(tool_call_id=tc["id"], content=str(res)))

        st.session_state.history.extend(tool_msgs)

        # Final pass: LLM response with tool outputs
        final = run_async(st.session_state.llm.ainvoke(st.session_state.history))
        with st.chat_message("assistant"):
            st.markdown(final.content or "")
        st.session_state.history.append(AIMessage(content=final.content or ""))