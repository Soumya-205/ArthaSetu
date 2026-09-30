"""Drive both ArthaSetu agents non-interactively and log node traces.

Run from the repo root (the agents use relative paths like data/chroma_db):

    python viz/run_batch.py --fast --cpu -n 40

--fast stubs ONLY the final customer-facing response (the slowest LLM call, and it
doesn't affect routing). Classification conflicts still call the real LLM.
"""
import argparse
import builtins
import os
import pathlib
import random
import sqlite3
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "agents"))

import trace_logger  # noqa: E402
import adoption_agent as adoption  # noqa: E402
import acquisition_agent_v3 as acq  # noqa: E402

# (profession, monthly income, education) -- edit education wording to match score_education
PERSONAS = [
    ("laborer", 8000, "primary education"), ("farmer", 8000, "primary education"),
    ("farmer", 50000, "postgraduate"), ("driver", 15000, "secondary education"),
    ("shopkeeper", 30000, "secondary education"), ("teacher", 45000, "graduate"),
    ("nurse", 35000, "graduate"), ("clerk", 25000, "graduate"),
    ("accountant", 60000, "postgraduate"), ("doctor", 120000, "postgraduate"),
    ("software engineer", 85000, "postgraduate"), ("student", 0, "graduate"),
]
INTERESTS = ["fixed deposit interest rates", "how does a personal loan work",
             "life insurance eligibility", "recurring deposit vs savings account",
             "savings account benefits"]


class FastLLM:
    """Proxy: stub only the final response prompt, pass everything else through."""
    def __init__(self, real):
        self.real = real

    def invoke(self, prompt, *a, **k):
        if isinstance(prompt, str) and prompt.startswith("You are a helpful bank assistant"):
            return types.SimpleNamespace(content="[stubbed response]")
        return self.real.invoke(prompt, *a, **k)

    def __getattr__(self, name):
        return getattr(self.real, name)


def run(mod, agent, answers):
    it = iter(answers)
    builtins.input = lambda prompt="": next(it)
    rid = trace_logger.new_run()
    events = []
    for chunk in mod.graph.stream({}, stream_mode="updates"):
        for node, update in chunk.items():
            events.append((node, update))
    for node, update in events:  # written only if the whole run succeeded
        trace_logger.log_node(rid, agent, node, update)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="run the LLM on CPU (avoids CUDA crashes on 4 GB GPUs)")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    if a.cpu:
        from langchain_ollama import ChatOllama
        adoption.llm = ChatOllama(model="mistral", num_gpu=0)
        acq.llm = ChatOllama(model="mistral", num_gpu=0)
    if a.fast:
        adoption.llm, acq.llm = FastLLM(adoption.llm), FastLLM(acq.llm)
    ids = [r[0] for r in sqlite3.connect(adoption.DB_PATH).execute("SELECT customer_id FROM customers")]
    fails = 0
    for i in range(a.n):
        q = rnd.choice(INTERESTS)
        try:
            if i % 2:
                cid = rnd.choice(ids) if rnd.random() > .08 else "X000"  # occasional bad ID -> not_found path
                run(adoption, "adoption", [cid, q])
            else:
                prof, inc, edu = rnd.choice(PERSONAS)
                run(acq, "acquisition", [prof, str(inc), edu, q])
            fails = 0
            print(f"run {i + 1}/{a.n} done")
        except Exception as e:  # skip a failed run; abort only if it keeps failing
            fails += 1
            print(f"run {i + 1}/{a.n} FAILED ({type(e).__name__}): {str(e)[:150]}")
            if fails >= 3:
                sys.exit("3 failures in a row - fix the error above, then rerun.")