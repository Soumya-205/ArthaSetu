"""
LangGraph Hello World

A minimal 2-node graph, just to see how State, nodes, and edges
actually work before building the real ArthaSetu graph.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


# Step 1: Define what the "state" looks like.

class GreetingState(TypedDict):
    name: str
    greeting: str


# Step 2: Define node functions.
# Every node takes the current state in, and returns a dict of updates.
def node_get_name(state: GreetingState) -> dict:
    print("  [node_get_name] running...")
    # In a real app this might come from user input.
    # For this test, we just hardcode it.
    return {"name": "Soumya"}


def node_make_greeting(state: GreetingState) -> dict:
    print("  [node_make_greeting] running...")
    # This node can READ what the previous node wrote into state.
    name = state["name"]
    return {"greeting": f"Hello, {name}! Welcome to LangGraph."}


# Step 3: Build the graph.
graph_builder = StateGraph(GreetingState)

graph_builder.add_node("get_name", node_get_name)
graph_builder.add_node("make_greeting", node_make_greeting)

graph_builder.set_entry_point("get_name")
graph_builder.add_edge("get_name", "make_greeting")
graph_builder.add_edge("make_greeting", END)

graph = graph_builder.compile()


# Step 4: Run it.
if __name__ == "__main__":
    print("Running the graph...\n")
    final_state = graph.invoke({})
    print("\nFinal state:", final_state)