"""Append one JSON line per fired LangGraph node. Lives in agents/ so the agents can
`from trace_logger import ...` exactly like they import classify_logic.
Writes to <repo>/viz/traces.jsonl.
"""
import json
import pathlib
import time
import uuid

LOG = pathlib.Path(__file__).resolve().parents[1] / "viz" / "traces.jsonl"


def new_run():
    return uuid.uuid4().hex[:8]


def log_node(run_id, agent, node, delta=None, path=LOG):
    """`delta` = dict returned by the node; keep only tiny routing metadata."""
    meta = {k: str(delta[k]) for k in ("customer_type", "classification_route")
            if isinstance(delta, dict) and k in delta}
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(run=run_id, agent=agent, node=node, t=time.time(), meta=meta)) + "\n")