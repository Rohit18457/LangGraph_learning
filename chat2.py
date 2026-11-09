"""
Example 2: Conditional Edges & Dynamic Branching in LangGraph
Demonstrates:
- Custom state schema using TypedDict with optional fields
- Using Groq client directly inside graph nodes
- Implementing conditional edges with `add_conditional_edges` and `Literal` return types
- Dynamically routing between alternative nodes or skipping to end
"""
import os
from typing import Optional, Literal
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
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


def evaluate_response(state: State) -> Literal["chatbot_gemini", "endnode"]:
    """Conditional Edge function: Decides next node based on state conditions."""
    print("⚖️  [evaluate_response] Routing decision...")
    # Route to alternative model or finish
    if False:
        return "endnode"
    return "chatbot_gemini"


def chatbot_gemini(state: State):
    """Alternative / Refinement LLM Node: Triggered conditionally."""
    print("✨ [chatbot_gemini Node] Refining response...")
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
graph_builder.add_node("chatbot_gemini", chatbot_gemini)
graph_builder.add_node("endnode", endnode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluate_response)
graph_builder.add_edge("chatbot_gemini", "endnode")
graph_builder.add_edge("endnode", END)

# Compile Graph
graph = graph_builder.compile()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Running LangGraph Conditional Branching Pipeline")
    print("=" * 60)

    initial_input = State({"user_query": "hey, give me 3+3"})
    updated_state = graph.invoke(initial_input)

    print("\nUpdated Result:")
    print("Query :", updated_state["user_query"])
    print("Answer:", updated_state["llm_output"])
