"""Generate N synthetic customers with ArthaSetu-style signals.

The 5-signal vote below is a STAND-IN mirroring the README (4-of-5 clear -> rule,
3-2 or worse -> LLM). Replace `standin_classify` with a call into your real
agents/classify_logic.py so the labels come from your actual system.
"""
import argparse
import pathlib
import sys
import numpy as np
import pandas as pd

PROF = {  # profession -> digital exposure prior (0..1)
    "laborer": .10, "farmer": .15, "driver": .30, "shopkeeper": .40,
    "clerk": .55, "teacher": .60, "nurse": .55, "accountant": .70,
    "doctor": .80, "software engineer": .95,
}
EDU = ["primary", "secondary", "graduate", "postgraduate"]


def standin_classify(r):
    """Return (customer_type, route, votes_for_B). Swap for classify_logic."""
    v = [PROF[r.profession] >= .5, r.income >= 30000, EDU.index(r.education) >= 2,
         r.logins_per_week >= 3, r.digital_ratio >= .4]
    b = int(sum(v))
    if b in (0, 1, 4, 5):
        return ("B" if b >= 4 else "A"), "rule", b
    behav = int(v[3]) + int(v[4])  # conflict: behavioural signals win (as in README)
    t = "B" if behav == 2 else "A" if behav == 0 else ("B" if b >= 3 else "A")
    return t, "llm", b


EDU_TEXT = {"primary": "primary education", "secondary": "secondary education",
            "graduate": "graduate", "postgraduate": "postgraduate"}  # match score_education's wording
_cache = {}


def real_classify(r, use_llm=False):
    """Label with the project's real classify_logic (5-signal adoption tally)."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "agents"))
    import classify_logic as cl
    prof = cl.score_profession(r.profession)
    if prof == "unknown":  # avoid an LLM call for lookup misses
        prof = "B" if PROF[r.profession] >= .5 else "A"
    votes = [cl.score_income(r.income), cl.score_education(EDU_TEXT[r.education]), prof,
             cl.score_login_frequency(r.logins_per_week), cl.score_digital_ratio(r.digital_ratio)]
    t = cl.tally_votes_adoption(votes)
    b = sum(v == "B" for v in votes)
    if t["outcome"] == "clear":
        return t["decision"], "rule", b
    if use_llm:
        from langchain_ollama import ChatOllama
        llm = _cache.setdefault("llm", ChatOllama(model="mistral", num_gpu=0))
        ctype, _ = cl.resolve_conflict_with_llm_adoption(
            r.profession, r.income, EDU_TEXT[r.education], r.logins_per_week, r.digital_ratio, llm)
        return ctype, "llm", b
    return standin_classify(r)[0], "llm", b  # approximate conflict resolution


def generate(n, seed=42, real=False, use_llm=False):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        prof = rng.choice(list(PROF))
        z = np.clip(PROF[prof] + rng.normal(0, .18), 0, 1)
        edu = EDU[int(np.clip(round(z * 3 + rng.normal(0, .8)), 0, 3))]
        income = int(np.clip(np.exp(8.6 + 2.2 * z + rng.normal(0, .45)), 5000, 150000))
        contrarian = rng.random() < .15  # e.g. rich/educated but rarely uses digital
        zb = np.clip((1 - z if contrarian else z) + rng.normal(0, .12), 0, 1)
        rows.append(dict(customer_id=f"C{i:03d}", profession=prof, education=edu,
                         income=income, logins_per_week=int(rng.poisson(0.5 + 7 * zb)),
                         digital_ratio=round(float(np.clip(zb + rng.normal(0, .1), 0, 1)), 2)))
    df = pd.DataFrame(rows)
    f = (lambda r: real_classify(r, use_llm)) if real else standin_classify
    df[["customer_type", "route", "votes_B"]] = df.apply(lambda r: pd.Series(f(r)), axis=1)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=250)
    ap.add_argument("-o", default="customers_synth.csv")
    ap.add_argument("--real", action="store_true", help="label with agents/classify_logic.py")
    ap.add_argument("--real-llm", action="store_true", help="also call the real LLM on conflicts (slow)")
    a = ap.parse_args()
    d = generate(a.n, real=a.real or a.real_llm, use_llm=a.real_llm)
    d.to_csv(a.o, index=False)
    print(d.customer_type.value_counts().to_dict(), d.route.value_counts().to_dict())