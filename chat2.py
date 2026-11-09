"""
Example 2: Conditional Edges & Dynamic Branching with State Checkpointing in LangGraph
Demonstrates:
- Custom state schema using TypedDict with optional fields
- Using Groq client directly inside graph nodes
- Implementing conditional edges with `add_conditional_edges` and `Literal` return types
- Dynamically routing between alternative nodes or skipping to end
- Checkpointing graph state with MongoDB (or falling back to MemorySaver)
"""
import os
from typing import Optional, Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from groq import Groq

load_dotenv()

client = Groq()
MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


class State(TypedDict):
    user_query: str
    llm_output: Optional[str]
    is_good: Optional[bool]


def chatbot(state: State):
    """First LLM Node: Generates initial response to user query."""
    print("🤖 [chatbot Node] Generating response...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": state.get("user_query")}
        ]
    )
    state["llm_output"] = response.choices[0].message.content
    return state


def evaluate_response(state: State) -> Literal["Groq_chatbot", "endnode"]:
    """Conditional Edge function: Decides next node based on state conditions."""
    print("⚖️  [evaluate_response] Routing decision...")
    # Branching logic: routes to alternative model or finishes
    if False:
        return "endnode"
    return "Groq_chatbot"


def Groq_chatbot(state: State):
    """Groq Chatbot Node: Triggered conditionally."""
    print("✨ [Groq_chatbot Node] Refining response...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": state.get("user_query")}
        ]
    )
    state["llm_output"] = response.choices[0].message.content
    return state


def endnode(state: State):
    """Final sink node before ending graph execution."""
    print("🏁 [endnode Node] Finalizing state...")
    return state


# Build Graph with Conditional Routing
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("Groq_chatbot", Groq_chatbot)
graph_builder.add_node("endnode", endnode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluate_response)
graph_builder.add_edge("Groq_chatbot", "endnode")
graph_builder.add_edge("endnode", END)


def get_checkpointer():
    """
    Attempts to connect to MongoDB for persistent checkpointing.
    If MongoDB is offline, gracefully falls back to MemorySaver.
    """
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://admin:admin@localhost:27017/?authSource=admin")
    try:
        mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=1500)
        mongo_client.admin.command('ping')
        print("💾 Connected to MongoDB checkpointer.")
        return MongoDBSaver(mongo_client, db_name="LangGraph")
    except Exception:
        print("⚠️  MongoDB offline. Using in-memory checkpointer (MemorySaver).")
        print("   (Tip: Run 'docker compose up -d' in LangGraph_learning to enable persistent MongoDB checkpoints)")
        return MemorySaver()


# Compile Graph with checkpointer
checkpointer = get_checkpointer()
graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)

# Configuration with proper thread_id for state isolation
config = {
    "configurable": {
        "thread_id": "Rohit"
    }
}

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LangGraph Interactive Chatbot")
    print("Type your query below. Type 'exit' or 'q' to quit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        current_input = State({
            "user_query": user_input,
            "llm_output": None,
            "is_good": None
        })

        updated_state = graph_with_checkpointer.invoke(current_input, config=config)

        print(f"\nAI > {updated_state.get('llm_output')}")
