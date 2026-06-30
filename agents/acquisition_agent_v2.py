"""
ArthaSetu- Acquisition Agent version 2 with classification

It extends the acquisition_agent_v1.py by adding node_classify- the hybrid
rule+LLM based classification step- using the scoring/voting LLM functions
already built and tested in classify_logic.py

Flow: profession->income->education->classify->END
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from classify_logic import(
    score_income,
    score_education,
    score_profession,
    score_profession_with_llm_fallback,
    tally_votes,
    resolve_conflict_with_llm,
)

class AcquisitionState(TypedDict):
    profession:str
    income:int
    education:str
    customer_type:str

llm=ChatOllama(model="llama3")

def node_profession(state: AcquisitionState)->dict:
    answer=input("Agent: What's your profession? > ")
    return {"profession": answer.strip()}

def node_income(state: AcquisitionState)->dict:
    answer=input("Agent: What's your approximate monthly income (in rupees)? > ")
    try:
        income_value=int(answer.strip())
    except ValueError:
        income_value=0
    return {"income": income_value}

def node_education(state: AcquisitionState)->dict:
    answer=input("Agent: What's your highest education level? > ")
    return {"education": answer.strip()}

def node_classify(state: AcquisitionState)->dict:
    """
    The hybrid rule + LLM classification node.
    
    1.Score all 3 signals.
    2.If profession is unknown, ask the LLM to classify it specifically.
    3.Tally the votes.
    4.If clear, use the rule's decision directly.
    5.If conflicting, ask the LLM to reason over full context.
    """
    print(" [node_classify] scoring signals...")

    income_vote=score_income(state["income"])
    education_vote=score_education(state["education"])
    profession_vote=score_profession(state["profession"])

    if profession_vote=="unknown":
        print(f" [node_classify] profession '{state['profession']}' not in lookup table, asking LLM...")
        profession_vote=score_profession_with_llm_fallback(state["profession"], llm)

    print(f" [node_classify] votes -> income: {income_vote}, education: {education_vote}, profession: {profession_vote}")

    tally=tally_votes(income_vote, education_vote, profession_vote)
    print(f" [node_classify] tally outcome: {tally['outcome']}")

    if tally["outcome"]=="clear":
        customer_type=tally["decision"]
        print(f" [node_classify] clear decision via rule: Type {customer_type}")

    else:
        print(" [node_classify] conflicating signals, asking LLM to reason...")
        customer_type, reasoning=resolve_conflict_with_llm(
            state["profession"], state["income"], state["education"], llm
        )
        print(f" [node_classify]LLM's fully reasoning:\n{reasoning}\n")
        print(f" [node_classify] LLM resolved condflict: Type {customer_type}")
    return {"customer_type": customer_type}

#Build the graph
graph_builder=StateGraph(AcquisitionState)

graph_builder.add_node("profession", node_profession)
graph_builder.add_node("income", node_income)
graph_builder.add_node("education", node_education)
graph_builder.add_node("classify", node_classify)

graph_builder.set_entry_point("profession")
graph_builder.add_edge("profession", "income")
graph_builder.add_edge("income", "education")
graph_builder.add_edge("education", "classify")
graph_builder.add_edge("classify", END)

graph = graph_builder.compile()

if __name__=="__main__":
    print("--- ArthaSetu Acquisition Agent (v2, with classification) ---\n")
    final_state=graph.invoke({})
    print("\nFinal profile:")
    print(f"  Profession: {final_state['profession']}")
    print(f"  Income: {final_state['income']}")
    print(f"  Education: {final_state['education']}")
    print(f"  Customer Type: {final_state['customer_type']}")
