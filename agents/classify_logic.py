"""
ArthaSetu - Classification scoring logic

Implements the three signal-scoring rules designed for node_classify:
  - income:     below 15,000 -> Type A, else -> Type B
  - education:  below undergraduate -> Type A, undergraduate+ -> Type B
  - profession: daily work doesn't need digital tools -> Type A, does -> Type B

Each scorer returns "A" or "B". A simple voting tally then decides
whether this is a clean 3-0 case (fast rule) or a 2-1 conflict
(needs the LLM to reason about it).
"""

import re

# A small lookup table for common professions.
# This is necessarily incomplete - anything not listed falls back
# to "unknown", which the LLM step will need to reason about directly.
DIGITAL_TOOL_PROFESSIONS = {
    # Type B - daily work typically involves digital/computer tools
    "teacher": "B",
    "software engineer": "B",
    "accountant": "B",
    "bank employee": "B",
    "sales officer": "B",
    "doctor": "B",
    "nurse": "B",
    "office worker": "B",
    "government employee": "B",
    "clerk": "B",
    "manager": "B",
    "consultant": "B",
    "lawyer": "B",
    "engineer": "B",
    "designer": "B",
    "shop owner": "B",
    "businessman": "B",
    "student": "B",

    # Type A - daily work typically doesn't require digital tools
    "farmer": "A",
    "daily wage laborer": "A",
    "laborer": "A",
    "mason": "A",
    "fisherman": "A",
    "domestic worker": "A",
    "driver": "A",
    "carpenter": "A",
    "electrician": "A",
    "plumber": "A",
    "tailor": "A",
    "vendor": "A",
    "street vendor": "A",
    "agricultural worker": "A",
    "construction worker": "A",
    "housewife": "A",
}


def score_income(income: int) -> str:
    """Below 15,000 leans exposure gap (A); 15,000+ leans convenience gap (B)."""
    if income < 15000:
        return "A"
    return "B"


def score_education(education: str) -> str:
    """Below undergraduate leans A; undergraduate or above leans B."""
    education = education.strip().lower()
    below_undergrad = ["none", "no formal education", "primary", "school", "high school", "secondary"]
    if education in below_undergrad:
        return "A"
    # Anything else (undergraduate, postgraduate, phd, etc.) leans B
    return "B"


def _extract_letter_answer(response_text: str) -> str:
    """
    Pulls out the LLM's intended A/B answer from its response text.

    Requires the LLM's prompt to explicitly end its response with
    "FINAL ANSWER: A" or "FINAL ANSWER: B". This is deliberate: an
    earlier version of this function searched for the LAST A/B letter
    anywhere in the response, but that breaks when the LLM reasons
    signal-by-signal (e.g. "income and profession suggest B, but
    education suggests A") - the last letter mentioned isn't
    necessarily the model's actual conclusion. Forcing a fixed,
    unambiguous marker removes this guesswork entirely.
    """
    match = re.search(r"FINAL ANSWER:\s*([AB])", response_text.upper())
    if match:
        return match.group(1)
    return "unknown"


def score_profession(profession: str) -> str:
    """Looks up the profession in a known list. Returns 'unknown' if not found."""
    profession = profession.strip().lower()
    return DIGITAL_TOOL_PROFESSIONS.get(profession, "unknown")


def score_profession_with_llm_fallback(profession: str, llm) -> str:
    """
    Same as score_profession, but if the profession isn't in the lookup
    table, asks the LLM to make a judgment call instead of giving up.

    'llm' is expected to be a LangChain-style chat model with an
    .invoke() method - wired in once we build the actual graph.
    """
    known_vote = score_profession(profession)
    if known_vote != "unknown":
        return known_vote

    prompt = (
        f"Does the profession '{profession}' typically require regular use of "
        f"digital tools (computer, smartphone apps, software) as part of daily work?\n\n"
        f"You may briefly explain your reasoning if needed, but you MUST end your "
        f"response with exactly this format on its own line:\n"
        f"FINAL ANSWER: A\n"
        f"or\n"
        f"FINAL ANSWER: B\n\n"
        f"where A means low digital tool usage and B means regular digital tool usage."
    )
    response = llm.invoke(prompt)
    answer = _extract_letter_answer(response.content)

    return answer


def resolve_conflict_with_llm(profession: str, income: int, education: str, llm) -> str:
    """
    Called when income, education, and profession signals genuinely
    disagree (a real 2-1 split). Unlike the profession fallback, this
    function gives the LLM the full real context - not just A/B votes -
    so it can reason about cases like a high-income, educated farmer.

    'llm' is expected to be a LangChain-style chat model with an
    .invoke() method.
    """
    prompt = (
        f"A bank is classifying a customer as either:\n"
        f"Type A (exposure gap - limited exposure to digital banking tools), or\n"
        f"Type B (convenience gap - aware of digital tools, just not motivated to use them).\n\n"
        f"Customer details:\n"
        f"- Profession: {profession}\n"
        f"- Monthly income: Rs.{income}\n"
        f"- Education level: {education}\n\n"
        f"Some signals about this customer may point in different directions. "
        f"Reason about which classification fits better overall, considering "
        f"that income and education often correlate with digital comfort, but "
        f"profession-specific context (e.g. a farmer's daily routine) matters too.\n\n"
        f"You may briefly explain your reasoning about each signal if needed, but "
        f"you MUST end your response with exactly this format on its own line, "
        f"stating your overall conclusion - not a per-signal breakdown:\n"
        f"FINAL ANSWER: A\n"
        f"or\n"
        f"FINAL ANSWER: B"
    )
    response = llm.invoke(prompt)
    answer = _extract_letter_answer(response.content)

    if answer == "unknown":
        # Only reached if the LLM's response contained neither letter at all -
        # a genuinely rare failure, unlike the old exact-match bug which
        # misfired on almost any normally-phrased response.
        return "B"
    return answer


def tally_votes(income_vote: str, education_vote: str, profession_vote: str) -> dict:
    """
    Counts votes for A and B, ignoring any 'unknown' profession vote.
    Returns a dict describing the outcome, used by node_classify to
    decide whether to use the fast rule or call the LLM.
    """
    votes = [v for v in [income_vote, education_vote, profession_vote] if v != "unknown"]

    count_a = votes.count("A")
    count_b = votes.count("B")

    if count_a == 3 or count_b == 3:
        # Clean 3-0 agreement (only possible if profession wasn't "unknown")
        decision = "A" if count_a == 3 else "B"
        return {"outcome": "clear", "decision": decision, "votes": votes}

    if count_a == 2 or count_b == 2:
        # Either a real 2-1 conflict, or profession was unknown and only 2 votes exist
        if len(votes) == 2:
            # Profession was unknown - only income + education voted.
            # If they agree, treat it as clear; if not, it's genuinely conflicting.
            if count_a == 2 or count_b == 2:
                decision = "A" if count_a == 2 else "B"
                return {"outcome": "clear", "decision": decision, "votes": votes}
        return {"outcome": "conflict", "decision": None, "votes": votes}

    # Only 1 vote available (income + education disagree, profession unknown)
    return {"outcome": "conflict", "decision": None, "votes": votes}


if __name__ == "__main__":
    # Quick manual tests, using the cases discussed in conversation
    print("--- Test 1: Government school teacher ---")
    i, e, p = score_income(18000), score_education("postgraduate"), score_profession("teacher")
    print(f"  income={i}, education={e}, profession={p}")
    print(f"  Tally: {tally_votes(i, e, p)}\n")

    print("--- Test 2: High-income, educated farmer ---")
    i, e, p = score_income(50000), score_education("postgraduate"), score_profession("farmer")
    print(f"  income={i}, education={e}, profession={p}")
    print(f"  Tally: {tally_votes(i, e, p)}\n")

    print("--- Test 3: Daily wage laborer ---")
    i, e, p = score_income(12000), score_education("primary"), score_profession("daily wage laborer")
    print(f"  income={i}, education={e}, profession={p}")
    print(f"  Tally: {tally_votes(i, e, p)}\n")

    print("--- Test 4: Unknown profession ---")
    i, e, p = score_income(40000), score_education("undergraduate"), score_profession("astronaut")
    print(f"  income={i}, education={e}, profession={p}")
    print(f"  Tally: {tally_votes(i, e, p)}\n")