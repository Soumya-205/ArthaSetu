"""Customer similarity graph: kNN graph on signals + Louvain communities.

Visual encodings: colour = Type A/B (toggle: community), shape = decided by rule
(circle) vs LLM conflict resolution (diamond), size = digital transaction ratio.
"""
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

EDU = ["primary", "secondary", "graduate", "postgraduate"]
PROF = {"laborer": .10, "farmer": .15, "driver": .30, "shopkeeper": .40, "clerk": .55,
        "teacher": .60, "nurse": .55, "accountant": .70, "doctor": .80,
        "software engineer": .95}
CORAL, TEAL = "#FF6B5B", "#1C9C9C"
PALETTE = ["#1f3a5f", "#e9a23b", "#7b5ea7", "#3c9d5d", "#c94f7c", "#5b8def", "#8c6d46", "#999999"]


def build(df, k):
    X = np.c_[np.log(df.income), df.education.map(EDU.index), df.profession.map(PROF),
              df.logins_per_week, df.digital_ratio]
    X = StandardScaler().fit_transform(X)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    G = nx.Graph()
    for i, r in df.iterrows():
        G.add_node(i, **r.to_dict())
    for i in range(len(df)):
        for d, j in zip(dist[i][1:], idx[i][1:]):
            G.add_edge(i, int(j), weight=float(1 / (1 + d)))
    return G


def analyse(G, df):
    comms = nx.community.louvain_communities(G, weight="weight", seed=42)
    cid = {n: c for c, s in enumerate(comms) for n in s}
    nx.set_node_attributes(G, cid, "community")
    mod = nx.community.modularity(G, comms, weight="weight")
    purity = sum(df.loc[list(s)].customer_type.value_counts().iloc[0] for s in comms) / len(df)
    diff = {n: np.mean([G.nodes[m]["customer_type"] != G.nodes[n]["customer_type"]
                        for m in G[n]]) for n in G}  # boundary score
    nx.set_node_attributes(G, diff, "boundary")
    b = pd.Series(diff).groupby(df.route).mean()
    print(f"communities={len(comms)}  modularity={mod:.3f}  type purity={purity:.1%}")
    print("mean boundary score (share of neighbours with other type):", b.round(3).to_dict())
    return mod, purity, b


def draw(G, out, seed=7):
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=0.35, iterations=200)
    ex, ey = [], []
    for u, v in G.edges:
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    N = list(G.nodes)
    a = lambda key: [G.nodes[n][key] for n in N]
    by_type = [CORAL if t == "A" else TEAL for t in a("customer_type")]
    by_comm = [PALETTE[c % len(PALETTE)] for c in a("community")]
    hover = [f"{G.nodes[n]['customer_id']}<br>{G.nodes[n]['profession']}, {G.nodes[n]['education']}"
             f"<br>income {G.nodes[n]['income']:,}<br>logins/wk {G.nodes[n]['logins_per_week']}, "
             f"digital {G.nodes[n]['digital_ratio']}<br>Type {G.nodes[n]['customer_type']} "
             f"via {G.nodes[n]['route']}<br>community {G.nodes[n]['community']}<br>"
             f"boundary {G.nodes[n]['boundary']:.2f}" for n in N]
    fig = go.Figure([
        go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=.4, color="rgba(120,120,120,.35)"),
                   hoverinfo="none", showlegend=False),
        go.Scatter(x=[pos[n][0] for n in N], y=[pos[n][1] for n in N], mode="markers",
                   text=hover, hoverinfo="text", showlegend=False,
                   marker=dict(size=[8 + 16 * d for d in a("digital_ratio")], color=by_type,
                               symbol=["diamond" if r == "llm" else "circle" for r in a("route")],
                               line=dict(width=1, color="white")))])
    T_TYPE = ("Customer similarity graph — coral = Type A (exposure gap), teal = Type B (convenience gap); "
              "diamond = LLM-resolved conflict; size = digital ratio")
    T_COMM = ("Same graph coloured by Louvain community (each colour = one community, "
              f"{len(set(a('community')))} found); diamond = LLM-resolved; size = digital ratio")
    btn = lambda label, col, t: dict(label=label, method="update",
                                     args=[{"marker.color": [col]}, {"title.text": t}, [1]])
    fig.update_layout(
        title=T_TYPE,
        title_font_size=13, plot_bgcolor="#FBF7F0", height=760,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.08,
                          buttons=[btn("Colour by type", by_type, T_TYPE),
                                   btn("Colour by community", by_comm, T_COMM)])])
    fig.write_html(out, include_plotlyjs="cdn")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="customers_synth.csv")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("-o", default="similarity_graph.html")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    G = build(df, a.k)
    analyse(G, df)
    draw(G, a.o)