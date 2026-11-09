"""
Example 1: Basic Linear StateGraph in LangGraph
Demonstrates:
- Defining graph State using TypedDict with message reducer `add_messages`
- Initializing Chat Model via LangChain init_chat_model with Groq
- Adding Nodes and linear Edges (START -> chatbot -> sampleNode -> END)
- Compiling and invoking the graph
"""
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

load_dotenv()

# Initialize Chat Model with Groq
llm = init_chat_model(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    model_provider="groq"
)


class State(TypedDict):
    # `add_messages` ensures new messages are appended rather than overwriting
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    """Chatbot Node: Calls the LLM with current conversation history."""
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]}


def sampleNode(state: State):
    """Sample Node: Demonstrates downstream node receiving updated state."""
    print("\nInside sampleNode, current state:", state)
    return {"messages": ["Hi, Sample message Appended."]}


# Build the Graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("sampleNode", sampleNode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "sampleNode")
graph_builder.add_edge("sampleNode", END)

# Compile Graph
graph = graph_builder.compile()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Running LangGraph Linear Chatbot Pipeline")
    print("=" * 60)

    initial_input = State({"messages": ["Hi, My name is Rohit Suryawanshi"]})
    updated_state = graph.invoke(initial_input)

    print("\nFinal State Result:")
    for msg in updated_state["messages"]:
        content = getattr(msg, "content", msg)
        msg_type = type(msg).__name__
        print(f" - [{msg_type}]: {content}")
