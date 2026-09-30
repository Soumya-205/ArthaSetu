"""Aggregate logged node traces into a weighted directed graph and draw it.

    python viz/exec_graph.py --demo     # synthetic traces with the REAL node names (dev only)
    python viz/exec_graph.py            # real traces from viz/traces.jsonl

Encodings: position = fixed layered layout (acquisition lane top, adoption lane bottom,
shared nodes middle), node size = visits, edge width = transition count,
colour at classify = resolved by rule (teal) vs LLM (coral).
"""
import argparse
import collections
import json
import pathlib
import random
import textwrap
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
CORAL, TEAL, NAVY, GREY = "#FF6B5B", "#1C9C9C", "#1f3a5f", "#9aa5b1"
FIXED = {  # topology is known, so a fixed layout reads better than a force layout
    "START": (0, 0), "profession": (1.2, 2.3), "income": (2.5, 2.3), "education": (3.8, 2.3),
    "fetch_customer": (1.2, -2.3), "not_found": (2.5, -3.3),
    "classify": (5.2, 0), "classify\n(rule)": (5.2, .95), "classify\n(LLM)": (5.2, -.95),
    "retrieve_type_a": (7.3, 1.9), "ask_interest": (7.3, -1.0), "retrieve_type_b": (9.0, -1.0),
    "respond": (10.6, 0), "END": (12.0, 0),
}


def demo_traces(n=120, seed=1):
    rnd, out = random.Random(seed), []
    for r in range(n):
        acq = rnd.random() < .5
        agent = "acquisition" if acq else "adoption"
        steps = ["profession", "income", "education"] if acq else ["fetch_customer"]
        if not acq and rnd.random() < .08:
            steps.append("not_found")
        else:
            via = "rule" if rnd.random() < (.6 if acq else .7) else "llm"
            typ = "A" if rnd.random() < .4 else "B"
            steps.append(("classify", {"classification_route": via, "customer_type": typ}))
            steps += ["retrieve_type_a"] if typ == "A" else ["ask_interest", "retrieve_type_b"]
            steps.append("respond")
        for s in steps:
            node, meta = s if isinstance(s, tuple) else (s, {})
            out.append(dict(run=f"d{r}", agent=agent, node=node, meta=meta))
    return out


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build(events):
    runs = collections.defaultdict(list)
    for e in events:
        runs[e["run"]].append(e)
    G = nx.DiGraph()
    def bump(u, v):
        G.add_edge(u, v, w=G[u][v]["w"] + 1 if G.has_edge(u, v) else 1)
    seen = collections.Counter()
    for evs in runs.values():
        prev = "START"
        for e in evs:
            label = e["node"]
            route = e["meta"].get("classification_route") or e["meta"].get("route")
            if label == "classify" and route:
                label += "\n(LLM)" if "llm" in route.lower() else "\n(rule)"
            seen[label] += 1
            bump(prev, label)
            prev = label
        bump(prev, "END")
    for n in G:
        G.nodes[n]["visits"] = len(runs) if n in ("START", "END") else seen[n]
    step = {n: i for i, n in enumerate(G)}  # fallback position for unknown node names
    pos = {n: FIXED.get(n, (2 + step[n] * .8, 0)) for n in G}
    return G, pos, len(runs)


def draw(G, pos, nruns, out, title):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor("#FBF7F0"); fig.patch.set_facecolor("#FBF7F0")
    mx = max(d["w"] for *_, d in G.edges(data=True))
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, [(u, v)], ax=ax, width=0.8 + 9 * d["w"] / mx, edge_color=NAVY,
                               alpha=.35, arrows=True, arrowsize=14, node_size=3600,
                               connectionstyle="arc3,rad=0.12")
    col = lambda n: (CORAL if "(LLM)" in n else TEAL if "(rule)" in n else
                     GREY if n in ("START", "END") else NAVY)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[col(n) for n in G], node_shape="s",
                           node_size=[2400 + 2600 * G.nodes[n]["visits"] / nruns for n in G],
                           edgecolors="white")
    def lab(n):
        base, *rest = n.split("\n")
        return textwrap.fill(base.replace("_", " "), 10, break_long_words=False) + ("\n" + rest[0] if rest else "")
    nx.draw_networkx_labels(G, pos, {n: lab(n) for n in G}, ax=ax, font_size=8, font_color="white")
    llm = sum(G.nodes[n]["visits"] for n in G if "(LLM)" in n)
    rule = sum(G.nodes[n]["visits"] for n in G if "(rule)" in n)
    if llm + rule:
        print(f"classify resolved by LLM in {llm}/{llm + rule} runs ({llm / (llm + rule):.0%})")
    for u, v, d in sorted(G.edges(data=True), key=lambda x: -x[2]["w"])[:5]:
        print(f"  {u!r:>22} -> {v!r:<22} {d['w']}")
    ax.set_title(title + f"  |  {nruns} runs  |  edge width = transitions, node size = visits", fontsize=11)
    ax.axis("off"); fig.tight_layout(); fig.savefig(out, dpi=170); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--log", default=str(HERE / "traces.jsonl"))
    ap.add_argument("-o", default="exec_graph.png")
    a = ap.parse_args()
    G, pos, n = build(demo_traces() if a.demo else load(a.log))
    draw(G, pos, n, a.o, "ArthaSetu execution paths" + (" (DEMO DATA)" if a.demo else ""))