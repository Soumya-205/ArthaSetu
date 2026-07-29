"""
ArthaSetu - Adoption Agent

For existing customers. Silently fetches customer data from SQLite,
scores 5 signals (income, education, profession, lofin frequency,
digital transaction ratio), classifies as Type A or Type B, then
generates a personalized product adoption recommendation using RAG.

Flow:
fetch_customer -> classify -> [Type A] retrieve_type_a -> respond -> END
                           -> [Type B] ask_interest -> retrieve_type_b -> respond ->END
Key differences from Acquisition Agent: no conversational signal collection.
All signals come from the database silently before any response is shown.
"""

import sqlite3
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
import chromadb

from classify_logic import (
    score_income,
    score_education,
    score_profession,
    score_profession_with_llm_fallback,
    score_login_frequency,
    score_digital_ratio,
    tally_votes_adoption,
    resolve_conflict_with_llm,
    resolve_conflict_with_llm_adoption,  
)

# Constants
DB_PATH = "data/customers.db"
CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "arthasetu_products"


class AdoptionState(TypedDict):
    customer_id: str
    name: str
    profession: str
    education: str
    monthly_income: int
    weekly_login_frequency: int
    digital_transaction_ratio: float
    account_type: str
    customer_found: bool
    customer_type: Optional[str]
    customer_query: Optional[str]
    retrieved_context: Optional[str]
    final_response: Optional[str]


# Shared connections
llm = ChatOllama(model="mistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)


# ── Entry node ────────────────────────────────────────────────────────────────

def node_fetch_customer(state: AdoptionState) -> dict:
    """
    Asks for a customer ID, fetches their full record from SQLite.
    Sets customer_found=False if the ID doesn't exist, which
    route_after_fetch uses to exit gracefully instead of crashing.
    """
    customer_id = input("Agent: Welcome! Please enter your customer ID > ").strip().upper()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        print(f"\n  Sorry, we couldn't find an account with ID '{customer_id}'.")
        print("  Please check your customer ID and try again, or visit your nearest branch.\n")
        return {
            "customer_id": customer_id,
            "customer_found": False,
            "name": "", "profession": "", "education": "",
            "monthly_income": 0, "weekly_login_frequency": 0,
            "digital_transaction_ratio": 0.0, "account_type": "",
        }

    print(f"\n  Welcome back, {row['name']}! Let me look at your account...\n")
    return {
        "customer_id": row["customer_id"],
        "name": row["name"],
        "profession": row["profession"],
        "education": row["education"],
        "monthly_income": row["monthly_income"],
        "weekly_login_frequency": row["weekly_login_frequency"],
        "digital_transaction_ratio": row["digital_transaction_ratio"],
        "account_type": row["account_type"],
        "customer_found": True,
    }


# ── Router after fetch ────────────────────────────────────────────────────────

def route_after_fetch(state: AdoptionState) -> str:
    """If customer wasn't found, skip everything and go straight to END."""
    if not state["customer_found"]:
        return "not_found"
    return "classify"


def node_not_found(state: AdoptionState) -> dict:
    """Placeholder node for the not-found path — message already printed in fetch."""
    return {}


# ── Classification node ───────────────────────────────────────────────────────

def node_classify(state: AdoptionState) -> dict:
    """
    Scores all 5 signals silently from the fetched customer data.
    Uses tally_votes_adoption (4-of-5 threshold) instead of the
    Acquisition Agent's 3-signal tally.
    """
    print("  [node_classify] scoring 5 signals silently...")

    income_vote = score_income(state["monthly_income"])
    education_vote = score_education(state["education"])
    profession_vote = score_profession(state["profession"])
    login_vote = score_login_frequency(state["weekly_login_frequency"])
    digital_vote = score_digital_ratio(state["digital_transaction_ratio"])

    if profession_vote == "unknown":
        print(f"  [node_classify] '{state['profession']}' not in lookup, asking LLM...")
        profession_vote = score_profession_with_llm_fallback(state["profession"], llm)

    print(f"  [node_classify] votes -> income: {income_vote}, education: {education_vote}, "
          f"profession: {profession_vote}, login: {login_vote}, digital: {digital_vote}")

    tally = tally_votes_adoption([income_vote, education_vote, profession_vote, login_vote, digital_vote])
    print(f"  [node_classify] tally: {tally['outcome']}")

    if tally["outcome"] == "clear":
        customer_type = tally["decision"]
        print(f"  [node_classify] rule decision: Type {customer_type}")
    else:
        print("  [node_classify] conflict — asking LLM to reason...")
        customer_type, reasoning = resolve_conflict_with_llm_adoption(
            state["profession"], state["monthly_income"], state["education"],
            state["weekly_login_frequency"], state["digital_transaction_ratio"], llm
        )
        print(f"  [node_classify] LLM reasoning:\n{reasoning}\n")
        print(f"  [node_classify] LLM decision: Type {customer_type}")

    return {"customer_type": customer_type}


# ── Conditional edge router ───────────────────────────────────────────────────

def route_by_customer_type(state: AdoptionState) -> str:
    if state["customer_type"] == "A":
        return "retrieve_type_a"
    return "ask_interest"


# ── Retrieval nodes ───────────────────────────────────────────────────────────

def node_retrieve_type_a(state: AdoptionState) -> dict:
    """Broad retrieval for exposure-gap customers."""
    print("  [node_retrieve_type_a] broad retrieval for exposure-gap customer...")
    query = (f"banking products for {state['profession']} "
             f"income {state['monthly_income']} account {state['account_type']}")
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=4)
    context = "\n\n---\n\n".join(results["documents"][0])
    return {"retrieved_context": context, "customer_query": query}


def node_ask_interest(state: AdoptionState) -> dict:
    """Ask Type B customer what they want to know about."""
    answer = input(f"\nAgent: Hi {state['name']}, what banking product or service "
                   f"would you like to know more about? > ")
    return {"customer_query": answer.strip()}


def node_retrieve_type_b(state: AdoptionState) -> dict:
    """Targeted retrieval based on what the customer asked."""
    print("  [node_retrieve_type_b] targeted retrieval for convenience-gap customer...")
    query_vector = embeddings.embed_query(state["customer_query"])
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    context = "\n\n---\n\n".join(results["documents"][0])
    return {"retrieved_context": context}


# ── Response node ─────────────────────────────────────────────────────────────

def node_respond(state: AdoptionState) -> dict:
    """Generates a personalized response using retrieved context and customer profile."""
    print("  [node_respond] generating personalized response...")

    if state["customer_type"] == "A":
        tone_instruction = (
            "Use simple, friendly language. Focus on benefits and how the product "
            "helps in daily life. Avoid jargon. Mention that they can visit their "
            "nearest branch for help with next steps."
        )
    else:
        tone_instruction = (
            "Use a clear, analytical tone. Present pros and cons so the customer "
            "can make an informed decision. They already understand banking basics — "
            "don't over-explain. End by asking if they'd like to proceed."
        )

    prompt = (
        f"You are a helpful bank assistant. An existing customer has opened the chat.\n\n"
        f"Customer profile:\n"
        f"- Name: {state['name']}\n"
        f"- Profession: {state['profession']}\n"
        f"- Monthly income: Rs.{state['monthly_income']}\n"
        f"- Education: {state['education']}\n"
        f"- Current account type: {state['account_type']}\n"
        f"- Classification: Type {'A (exposure gap)' if state['customer_type'] == 'A' else 'B (convenience gap)'}\n\n"
        f"Relevant banking information:\n"
        f"{state['retrieved_context']}\n\n"
        f"Respond to the customer based on the above.\n"
        f"Tone guidance: {tone_instruction}\n\n"
        f"Keep your response concise — 3 to 5 sentences maximum. "
        f"Address the customer by their first name."
    )

    response = llm.invoke(prompt)
    return {"final_response": response.content}


# ── Build the graph ───────────────────────────────────────────────────────────

graph_builder = StateGraph(AdoptionState)

graph_builder.add_node("fetch_customer", node_fetch_customer)
graph_builder.add_node("not_found", node_not_found)
graph_builder.add_node("classify", node_classify)
graph_builder.add_node("retrieve_type_a", node_retrieve_type_a)
graph_builder.add_node("ask_interest", node_ask_interest)
graph_builder.add_node("retrieve_type_b", node_retrieve_type_b)
graph_builder.add_node("respond", node_respond)

graph_builder.set_entry_point("fetch_customer")

# First conditional edge — did we find the customer?
graph_builder.add_conditional_edges(
    "fetch_customer",
    route_after_fetch,
    {
        "not_found": "not_found",
        "classify": "classify",
    }
)

graph_builder.add_edge("not_found", END)

# Second conditional edge — Type A or Type B?
graph_builder.add_conditional_edges(
    "classify",
    route_by_customer_type,
    {
        "retrieve_type_a": "retrieve_type_a",
        "ask_interest": "ask_interest",
    }
)

graph_builder.add_edge("retrieve_type_a", "respond")
graph_builder.add_edge("ask_interest", "retrieve_type_b")
graph_builder.add_edge("retrieve_type_b", "respond")
graph_builder.add_edge("respond", END)

graph = graph_builder.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("--- ArthaSetu Adoption Agent ---\n")

    final_state = {}
    for chunk in graph.stream({}, stream_mode="updates"):
        for node_name, node_update in chunk.items():
            print(f"\n-> Node fired: {node_name}")
            final_state.update(node_update)

    if final_state.get("customer_found"):
        print("\n" + "="*60)
        print("AGENT RESPONSE TO CUSTOMER:")
        print("="*60)
        print(final_state["final_response"])
        print("="*60)
        print(f"\n[Debug] Customer: {final_state['name']} | Type: {final_state['customer_type']}")
        print(f"[Debug] Query used: {final_state.get('customer_query', 'N/A')}")