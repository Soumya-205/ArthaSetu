# ArthaSetu

**Agentic AI for Banking Customer Acquisition & Digital Adoption**

> 🚧 Work in progress — an ongoing personal project exploring agentic AI in banking.

---

## Problem Statement

Banks face increasing challenges in acquiring customers at scale, driving adoption of digital products, and creating meaningful long-term engagement. Branch managers in particular juggle two competing responsibilities at once: bringing in brand-new customers, and getting existing customers to adopt digital products they haven't tried yet.

ArthaSetu is an agentic AI system designed to assist with both — acquisition of new prospects, and adoption of digital products among existing customers — using a single underlying reasoning core.

## Core Insight

Low digital adoption isn't one problem — it's two different problems that look the same on the surface:

- **Exposure gap** — the customer hasn't had enough exposure to digital banking tools to feel comfortable using them (often, but not exclusively, more common in rural contexts with less digital infrastructure).
- **Convenience gap** — the customer is aware of digital products but hasn't been sufficiently nudged or motivated to adopt them (often, but not exclusively, more common in urban contexts).

Treating both groups the same way wastes effort. ArthaSetu's agent reasons about *which* gap a customer has before deciding *how* to engage them — adjusting tone, depth, and channel recommendations accordingly, rather than applying a one-size-fits-all script.

Importantly, this classification is based on **behavioral and contextual signals** (digital engagement patterns, education, occupation), never on geography or demographic labels directly — the system never tags anyone as "rural" or "urban."

## Architecture Overview

ArthaSetu has two entry points that converge into a shared pipeline:

```mermaid
flowchart TD
    A[Customer opens chat] --> B{Record found in DB?}
    B -- No --> C[Acquisition Agent<br/>New prospect, no history]
    B -- Yes --> D[Adoption Agent<br/>Existing customer, has history]

    C --> C1[node_profession]
    C1 --> C2[node_income]
    C2 --> C3[node_education]

    D --> D1[node_fetch_signals<br/>Login freq, UPI ratio, KYC]
    D1 --> D2{Signals clear?}
    D2 -- Mixed --> D3[Ask one indirect question]
    D3 --> D2

    C3 --> E[node_classify]
    D2 -- Clear --> E

    E --> E1{Vote on 3 signals}
    E1 -- "3-0 agreement" --> E2[Hardcoded rule<br/>fast, no LLM cost]
    E1 -- "2-1 conflict" --> E3[LLM reasoning call<br/>Llama3 weighs context]

    E2 --> F{customer_type}
    E3 --> F

    F -- Type A: exposure gap --> G1[RAG query]
    F -- Type B: convenience gap --> G2[RAG query]

    G1 --> H1[LLM response<br/>Simple language, benefits + risks]
    G2 --> H2[LLM response<br/>Pros/cons, let customer decide]

    H1 --> I1[Guide next steps<br/>App setup, branch fallback]
    H2 --> I2[Guide next steps<br/>Application steps, branch fallback]

    I1 --> J[Goal achieved]
    I2 --> J
```

Both entry points only differ in **how** the state gets filled — asking conversationally vs. fetching from existing records. Everything from `node_classify` onward is identical logic shared by both paths.

- **Acquisition Agent** — for brand-new prospects with no existing bank relationship. Builds a profile conversationally (profession, income, education) since no account data exists yet.
- **Adoption Agent** — for existing customers. Fetches behavioral signals silently from existing data (login frequency, digital vs. branch transaction ratio, KYC-linked education/occupation).

Both paths feed into the same **classification core**:

1. Each available signal is scored as leaning toward **Type A (exposure gap)** or **Type B (convenience gap)**.
2. If the signals agree (e.g. 3-0), a fast hardcoded rule decides instantly.
3. If the signals conflict (e.g. 2-1), an LLM call reasons over the context to make the call — handling edge cases a fixed rule would get wrong (e.g. a farmer with high income and strong education shouldn't be auto-classified as exposure-gap).

From there, both customer types go through the same RAG-grounded response pipeline — facts retrieved are identical for everyone; only the LLM's phrasing, tone, and emphasis differ based on customer type.

## Tech Stack (so far)

- **LLM:** Llama3 via Ollama (local)
- **Embeddings:** nomic-embed-text via Ollama (local)
- **Vector store:** ChromaDB (persistent, local)
- **Agent orchestration:** LangGraph (planned)
- **Language:** Python 3.12

## Current Progress

- [x] Problem framing and architecture design
- [x] RAG knowledge base — product documents (`data/products/`) for Fixed Deposit, Recurring Deposit, Savings Account
- [x] Section-wise chunking strategy (chunks split by `##` heading: overview, eligibility, rates, risks, how_to_apply)
- [x] Ingestion pipeline (`agents/ingest.py`) — embeds and stores chunks in ChromaDB with `product_name` and `section_type` metadata
- [x] Retrieval test script (`agents/test_retrieval.py`)
- [ ] More product documents (insurance, loans)
- [ ] LangGraph state design and node implementation
- [ ] Acquisition agent conversational flow
- [ ] Adoption agent signal-fetching logic
- [ ] Hybrid rule + LLM classification logic
- [ ] End-to-end demo

## Project Structure

```
ArthaSetu/
├── agents/              # Agent logic, graph nodes, ingestion & test scripts
├── data/
│   ├── products/        # RAG knowledge base - one .md file per banking product
│   └── chroma_db/       # Persistent ChromaDB store (generated, not committed)
├── notebooks/           # Exploration and experiments
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone and enter the project
git clone https://github.com/Soumya-205/ArthaSetu.git
cd ArthaSetu

# Create and activate a virtual environment (Python 3.12 recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull llama3
ollama pull nomic-embed-text

# Build the knowledge base
python agents/ingest.py

# Try retrieval
python agents/test_retrieval.py
```

## Note on Data

Product details (interest rates, eligibility criteria) used in this project are **illustrative and synthetic**, modeled on real banking product categories but not scraped from any live source. They are not accurate, current rates for any real bank and should not be used for actual financial decisions.

## Author

Built by [Soumya](https://github.com/Soumya-205) — BTech CSE (Data Science), Manipal University Jaipur.# ArthaSetu
