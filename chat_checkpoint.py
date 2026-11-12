"""
Example 3: LangGraph State Persistence with Checkpointers (MongoDB / Memory)
Demonstrates:
- Preserving multi-turn chat memory across invokes using checkpointer threads
- Streaming graph state updates
- Automatic fallback from MongoDB to MemorySaver when Docker container is offline
"""
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain.chat_models import init_chat_model
from pymongo import MongoClient

load_dotenv()

# Initialize Chat Model
llm = init_chat_model(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    model_provider="groq"
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


def get_checkpointer():
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://admin:admin@localhost:27017/?authSource=admin")
    try:
        mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=1500)
        mongo_client.admin.command('ping')
        print("💾 Connected to MongoDB checkpointer.")
        return MongoDBSaver(mongo_client, db_name="LangGraph")
    except Exception:
        print("⚠️  MongoDB offline. Using in-memory checkpointer (MemorySaver).")
        print("   (Tip: Run 'docker compose up -d' to start MongoDB)")
        return MemorySaver()


checkpointer = get_checkpointer()
graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Running LangGraph Multi-Turn Checkpoint Chat")
    print("=" * 60)

    config = {"configurable": {"thread_id": "user_rohit"}}

    print("\n--- Turn 1: Introducing Name ---")
    turn1_input = State({"messages": ["Hi! My name is Rohit Suryawanshi."]})
    for chunk in graph_with_checkpointer.stream(turn1_input, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    print("\n--- Turn 2: Asking LLM to Remember Name ---")
    turn2_input = State({"messages": ["What is my name?"]})
    for chunk in graph_with_checkpointer.stream(turn2_input, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

