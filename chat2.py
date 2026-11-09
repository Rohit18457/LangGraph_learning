"""
Example 2: Conditional Edges & Multi-Turn Chatbot with State Checkpointing in LangGraph
Demonstrates:
- Message history persistence using `Annotated[list, add_messages]` reducer
- Conditional edges with `add_conditional_edges` and `Literal` routing
- Checkpointing graph state with MongoDB (or falling back to MemorySaver)
- Interactive multi-turn conversation loop that remembers context across turns
"""
import os
from typing import Annotated, Optional, Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pymongo import MongoClient

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# Initialize Chat Model with Groq (capping max_tokens to prevent OTPM rate limits)
llm = init_chat_model(
    model=MODEL_NAME,
    model_provider="groq",
    max_tokens=300
)


class State(TypedDict):
    # `add_messages` ensures all user and AI turns are preserved in sequence
    messages: Annotated[list, add_messages]
    is_good: Optional[bool]


def chatbot(state: State):
    """First LLM Node: Generates response using full conversation history."""
    print("🤖 [chatbot Node] Generating response...")
    try:
        response = llm.invoke(state["messages"])
        return {
            "messages": [response],
            "is_good": True
        }
    except Exception as e:
        print(f"⚠️ Chatbot Error: {e}")
        return {
            "messages": [{"role": "assistant", "content": f"⚠️ Error generating response: {e}"}],
            "is_good": False
        }


def evaluate_response(state: State) -> Literal["Groq_chatbot", "endnode"]:
    """Conditional Edge function: Decides next node based on state conditions."""
    print("⚖️  [evaluate_response] Routing decision...")
    # If the response was successful, finish directly. Refine only if failed.
    if state.get("is_good"):
        return "endnode"
    return "Groq_chatbot"


def Groq_chatbot(state: State):
    """Fallback / Refinement LLM Node: Triggered conditionally if primary node needed retry."""
    print("✨ [Groq_chatbot Node] Refining response...")
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"⚠️ Refinement Error: {e}"}]}


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

# Configuration with thread_id for conversation state isolation
config = {
    "configurable": {
        "thread_id": "user_session_rohit"
    }
}

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LangGraph Interactive Multi-Turn Chatbot (with Memory)")
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

        current_input = {
            "messages": [HumanMessage(content=user_input)]
        }

        updated_state = graph_with_checkpointer.invoke(current_input, config=config)

        # Get the latest assistant response
        latest_message = updated_state["messages"][-1]
        content = getattr(latest_message, "content", latest_message)
        print(f"\nAI > {content}")
