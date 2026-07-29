"""
ArthaSetu - Acquisition Agent (v3, with RAG response)

Extends v2 by adding:
- Conditional edge after node_classify routing to Type A or Type B retrieval
- node_retrieve_type_a: broad retrieval, agent picks most relevant products
- node_retrieve_type_b: customer states interest, targeted retrieval
- node_respond: LLM generates personalized response using retrieved chunks

Full flow:
profession -> income -> education -> classify
    -> [Type A] retrieve_type_a -> respond -> END
    -> [Type B] ask_interest -> retrieve_type_b -> respond -> END
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
import chromadb

from classify_logic import (
    score_income,
    score_education,
    score_profession,
    score_profession_with_llm_fallback,
    tally_votes,
    resolve_conflict_with_llm,
)

# Constants matching your ingestion setup
CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "arthasetu_products"


class AcquisitionState(TypedDict):
    profession: str
    income: int
    education: str
    customer_type: str
    customer_query: Optional[str]
    retrieved_context: Optional[str]
    final_response: Optional[str]


# Shared connections
llm = ChatOllama(model="mistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)


# ── Conversation nodes ────────────────────────────────────────────────────────

def node_profession(state: AcquisitionState) -> dict:
    answer = input("Agent: What's your profession? > ")
    return {"profession": answer.strip()}


def node_income(state: AcquisitionState) -> dict:
    answer = input("Agent: What's your approximate monthly income (in rupees)? > ")
    try:
        income_value = int(answer.strip())
    except ValueError:
        income_value = 0
    return {"income": income_value}


def node_education(state: AcquisitionState) -> dict:
    answer = input("Agent: What's your highest education level? > ")
    return {"education": answer.strip()}


# ── Classification node ───────────────────────────────────────────────────────

def node_classify(state: AcquisitionState) -> dict:
    print("  [node_classify] scoring signals...")

    income_vote = score_income(state["income"])
    education_vote = score_education(state["education"])
    profession_vote = score_profession(state["profession"])

    if profession_vote == "unknown":
        print(f"  [node_classify] '{state['profession']}' not in lookup, asking LLM...")
        profession_vote = score_profession_with_llm_fallback(state["profession"], llm)

    print(f"  [node_classify] votes -> income: {income_vote}, education: {education_vote}, profession: {profession_vote}")

    tally = tally_votes(income_vote, education_vote, profession_vote)
    print(f"  [node_classify] tally: {tally['outcome']}")

    if tally["outcome"] == "clear":
        customer_type = tally["decision"]
        print(f"  [node_classify] rule decision: Type {customer_type}")
    else:
        print("  [node_classify] conflict — asking LLM to reason...")
        customer_type, reasoning = resolve_conflict_with_llm(
            state["profession"], state["income"], state["education"], llm
        )
        print(f"  [node_classify] LLM reasoning:\n{reasoning}\n")
        print(f"  [node_classify] LLM decision: Type {customer_type}")

    return {"customer_type": customer_type}


# ── Conditional edge router ───────────────────────────────────────────────────

def route_by_customer_type(state: AcquisitionState) -> str:
    if state["customer_type"] == "A":
        return "retrieve_type_a"
    return "ask_interest"


# ── Retrieval nodes ───────────────────────────────────────────────────────────

def node_retrieve_type_a(state: AcquisitionState) -> dict:
    """
    Type A: broad retrieval across all products.
    Query is constructed from the customer's own profile since they
    haven't expressed a specific interest yet.
    """
    print("  [node_retrieve_type_a] retrieving broadly for exposure-gap customer...")
    query = f"banking products savings deposits insurance for {state['profession']} income {state['income']}"
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=4)
    context = "\n\n---\n\n".join(results["documents"][0])
    return {"retrieved_context": context, "customer_query": query}


def node_ask_interest(state: AcquisitionState) -> dict:
    """
    Type B: ask the customer what they want to know before retrieving.
    Their answer becomes the ChromaDB query directly.
    """
    answer = input("\nAgent: What would you like to know more about? > ")
    return {"customer_query": answer.strip()}


def node_retrieve_type_b(state: AcquisitionState) -> dict:
    """
    Type B: targeted retrieval based on what the customer actually asked.
    """
    print("  [node_retrieve_type_b] retrieving targeted chunks for convenience-gap customer...")
    query_vector = embeddings.embed_query(state["customer_query"])
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    context = "\n\n---\n\n".join(results["documents"][0])
    return {"retrieved_context": context}


# ── Response node ─────────────────────────────────────────────────────────────

def node_respond(state: AcquisitionState) -> dict:
    """
    Generates a personalized response using the retrieved context.
    Tone adapts based on customer_type:
    - Type A: simple language, benefits-first, branch fallback mentioned
    - Type B: comparative, analytical, lets customer decide
    """
    print("  [node_respond] generating personalized response...")

    if state["customer_type"] == "A":
        tone_instruction = (
            "Use simple, friendly language. Focus on the benefits and how the product "
            "helps in daily life. Avoid jargon. At the end, always mention that the "
            "customer can visit their nearest branch for help with any next steps."
        )
    else:
        tone_instruction = (
            "Use a clear, analytical tone. Present the pros and cons so the customer "
            "can make an informed decision. Respect that they already understand banking "
            "basics — don't over-explain. End by asking if they'd like to proceed."
        )

    prompt = (
        f"You are a helpful bank assistant. A customer has approached the bank.\n\n"
        f"Customer profile:\n"
        f"- Profession: {state['profession']}\n"
        f"- Monthly income: Rs.{state['income']}\n"
        f"- Education: {state['education']}\n"
        f"- Classification: Type {'A (exposure gap)' if state['customer_type'] == 'A' else 'B (convenience gap)'}\n\n"
        f"Relevant banking information retrieved for this customer:\n"
        f"{state['retrieved_context']}\n\n"
        f"Respond to the customer based on the above information.\n"
        f"Tone guidance: {tone_instruction}\n\n"
        f"Keep your response concise — 3 to 5 sentences maximum."
    )

    response = llm.invoke(prompt)
    return {"final_response": response.content}


# ── Build the graph ───────────────────────────────────────────────────────────

graph_builder = StateGraph(AcquisitionState)

# Add all nodes
graph_builder.add_node("profession", node_profession)
graph_builder.add_node("income", node_income)
graph_builder.add_node("education", node_education)
graph_builder.add_node("classify", node_classify)
graph_builder.add_node("retrieve_type_a", node_retrieve_type_a)
graph_builder.add_node("ask_interest", node_ask_interest)
graph_builder.add_node("retrieve_type_b", node_retrieve_type_b)
graph_builder.add_node("respond", node_respond)

# Linear edges for the first half
graph_builder.set_entry_point("profession")
graph_builder.add_edge("profession", "income")
graph_builder.add_edge("income", "education")
graph_builder.add_edge("education", "classify")

# Conditional edge after classify
graph_builder.add_conditional_edges(
    "classify",
    route_by_customer_type,
    {
        "retrieve_type_a": "retrieve_type_a",
        "ask_interest": "ask_interest",
    }
)

# Type A path
graph_builder.add_edge("retrieve_type_a", "respond")

# Type B path
graph_builder.add_edge("ask_interest", "retrieve_type_b")
graph_builder.add_edge("retrieve_type_b", "respond")

# Both paths end here
graph_builder.add_edge("respond", END)

graph = graph_builder.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("--- ArthaSetu Acquisition Agent (v3, with RAG response) ---\n")

    final_state = {}
    for chunk in graph.stream({}, stream_mode="updates"):
        for node_name, node_update in chunk.itens():
            print(f"\n-> Node fired: {node_name}")
            final_state.update(node_update)

    print("\n" + "="*60)
    print("AGENT RESPONSE TO CUSTOMER:")
    print("="*60)
    print(final_state["final_response"])
    print("="*60)
    print(f"\n[Debug] Customer Type: {final_state['customer_type']}")
    print(f"[Debug] Query used: {final_state['customer_query']}")