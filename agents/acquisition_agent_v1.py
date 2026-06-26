"""
ArthaSetu - Acquisition Agent (early version)

A 3-node LangGraph chain that conversationally collects profession,
income, and education from a brand-new customer - exactly the
"Acquisition path" from the architecture diagram.

This does NOT yet include classification (node_classify) 
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


class AcquisitionState(TypedDict):
    profession: str
    income: int
    education: str


def node_profession(state: AcquisitionState) -> dict:
    answer = input("Agent: What's your profession? > ")
    return {"profession": answer.strip()}


def node_income(state: AcquisitionState) -> dict:
    answer = input("Agent: What's your approximate monthly income (in rupees)? > ")
    # Basic safety: if they type something non-numeric, default to 0
    # rather than crashing. 
    try:
        income_value = int(answer.strip())
    except ValueError:
        income_value = 0
    return {"income": income_value}


def node_education(state: AcquisitionState) -> dict:
    answer = input("Agent: What's your highest education level? > ")
    return {"education": answer.strip()}


# Build the graph
graph_builder = StateGraph(AcquisitionState)

graph_builder.add_node("profession", node_profession)
graph_builder.add_node("income", node_income)
graph_builder.add_node("education", node_education)

graph_builder.set_entry_point("profession")
graph_builder.add_edge("profession", "income")
graph_builder.add_edge("income", "education")
graph_builder.add_edge("education", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    print("--- ArthaSetu Acquisition Agent (early test) ---\n")
    final_state = graph.invoke({})
    print("\nCollected profile:")
    print(f"  Profession: {final_state['profession']}")
    print(f"  Income: {final_state['income']}")
    print(f"  Education: {final_state['education']}")