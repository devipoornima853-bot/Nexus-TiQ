"""
LLM layer. This module's only job is language: understanding what the
manager is asking, deciding which deterministic query in analytics.py
answers it, and phrasing the result in plain language. It never computes a
number itself and never states a figure that didn't come from
analytics.py — the prompt below says so explicitly, and the calling code
in app.py only ever forwards the structured `grounding` block that was
actually computed.

If GEMINI_API_KEY is missing or the API call fails, we fall back to a
plain templated rendering of the grounding data rather than failing the
request — a store manager should never see a blank screen because a
network call timed out.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from src.analytics import Catalogue

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTIONS = """You are a retail operations copilot for a store manager.

Rules you must never break:
1. You may only state numbers that appear in the JSON "grounding data" you are given below. Never estimate, round creatively, or invent a figure that isn't there.
2. Every claim you make must be traceable to a specific field in the grounding data. Refer to the numbers directly (e.g. "12 units left, selling ~3.4/day").
3. If the grounding data is empty or does not cover what was asked, say plainly that the data can't answer that — do not guess.
4. Keep the answer short: 2-5 sentences, or a short list if multiple items are involved. No filler, no generic advice not tied to a number.
5. When you recommend an action, name the specific product/store it applies to and state the number that justifies it.
"""


def _extract_entities(question: str, cat: Catalogue) -> dict:
    """Very small heuristic entity finder: sees if any known product name
    or store name/city is mentioned in the question, so we can scope the
    deterministic query instead of dumping the whole catalogue as context."""
    q = question.lower()
    product_id = None
    for _, row in cat.products.iterrows():
        if row["name"].lower() in q:
            product_id = row["product_id"]
            break
    store_id = None
    for _, row in cat.stores.iterrows():
        if row["name"].lower() in q or row["city"].lower() in q:
            store_id = row["store_id"]
            break
    return {"product_id": product_id, "store_id": store_id}


def build_grounding(question: str, cat: Catalogue) -> dict:
    """Decide what deterministic data is relevant to this question and
    compute it. This is the only place allowed to call analytics.py for a
    chat answer, so the mapping from question -> data is easy to audit."""
    q = question.lower()
    entities = _extract_entities(question, cat)
    grounding: dict = {"as_of": cat.as_of.date().isoformat(), "matched_entities": entities}

    if entities["product_id"]:
        grounding["product_summary"] = cat.product_summary(entities["product_id"], entities["store_id"])
        grounding["stockout_risk_for_product"] = [
            r for r in cat.stockout_risks(entities["store_id"]) if r["product_id"] == entities["product_id"]
        ]

    if any(w in q for w in ["out of stock", "running out", "run out", "stockout", "stock out", "reorder", "running low", "low on"]):
        grounding["stockout_risks"] = cat.stockout_risks(entities["store_id"])[:10]

    if any(w in q for w in ["overstock", "not moving", "slow", "dead stock", "excess"]):
        grounding["slow_movers"] = cat.slow_movers(entities["store_id"])[:10]

    if any(w in q for w in ["spike", "drop", "trend", "this month", "this week", "doing", "performance", "sold well"]):
        grounding["sales_moves"] = cat.sales_moves(entities["store_id"])[:10]

    # Nothing matched a specific intent and no product was named. Only fall
    # back to the daily briefing if the question actually reads as a
    # general retail-ops question (e.g. "how are we doing", "anything I
    # should look at") — an unrelated question (weather, sports, etc.)
    # should surface as "no data for this" rather than dumping the
    # briefing, which would look like the system is inventing relevance.
    general_ops_words = [
        "need attention", "needs attention", "what should i", "anything i should",
        "how are we doing", "how's the store", "how is the store", "how are sales",
        "store status", "daily brief", "daily briefing", "look at today",
        "check on the store", "overview of the store", "what's going on in the store",
    ]
    if len(grounding) == 2 and any(w in q for w in general_ops_words):
        grounding["daily_briefing"] = cat.daily_briefing(entities["store_id"])

    return grounding


def _template_fallback(question: str, grounding: dict) -> str:
    """Deterministic, LLM-free rendering used if Gemini is unavailable.
    Not as fluent, but every number still traces to grounding directly."""
    parts = []
    if grounding.get("product_summary"):
        s = grounding["product_summary"]
        parts.append(
            f"{s['product_name']}: {s['units_sold']} units sold and \u20b9{s['revenue']} revenue "
            f"in the last {s['window_days']} days, {s['current_stock_total']} units in stock as of {s['as_of']}."
        )
    if "stockout_risks" in grounding:
        items = grounding["stockout_risks"][:5]
        if items:
            lines = [f"{i['product_name']} ({i['store_id']}): {i['quantity_on_hand']} left, "
                      f"~{i['days_left']}d of stock at current sales" for i in items]
            parts.append("Stock-out risk in the next 7 days:\n- " + "\n- ".join(lines))
        else:
            parts.append("No products are projected to run out within 7 days.")
    if "slow_movers" in grounding:
        items = grounding["slow_movers"][:5]
        if items:
            lines = [f"{i['product_name']} ({i['store_id']}): {i['quantity_on_hand']} units on hand, "
                      f"only {i['units_sold_in_window']} sold in {i['window_days']} days" for i in items]
            parts.append("Slow movers:\n- " + "\n- ".join(lines))
        else:
            parts.append("No slow movers found with the current thresholds.")
    if "sales_moves" in grounding:
        items = grounding["sales_moves"][:5]
        if items:
            lines = [f"{i['product_name']} ({i['store_id']}): {i['direction']} of {i['pct_change']}% "
                      f"vs its own baseline" for i in items]
            parts.append("Notable sales moves:\n- " + "\n- ".join(lines))
        else:
            parts.append("No sales spikes or drops beyond the threshold were found.")
    if grounding.get("daily_briefing"):
        db = grounding["daily_briefing"]
        risk, slow, moves = db["stockout_risks"], db["slow_movers"], db["sales_moves"]
        if risk:
            lines = [f"{i['product_name']} ({i['store_id']}): {i['quantity_on_hand']} left, "
                      f"~{i['days_left']}d of stock" for i in risk[:5]]
            parts.append("Stock-out risk in the next 7 days:\n- " + "\n- ".join(lines))
        if slow:
            lines = [f"{i['product_name']} ({i['store_id']}): {i['quantity_on_hand']} on hand, "
                      f"only {i['units_sold_in_window']} sold in {i['window_days']} days" for i in slow[:5]]
            parts.append("Slow movers:\n- " + "\n- ".join(lines))
        if moves:
            lines = [f"{i['product_name']} ({i['store_id']}): {i['direction']} of {i['pct_change']}%" for i in moves[:5]]
            parts.append("Notable sales moves:\n- " + "\n- ".join(lines))
        if not (risk or slow or moves):
            parts.append("Nothing needs attention right now — no stock-out risk, slow movers, "
                          "or unusual sales moves found in the data.")
    if not parts:
        parts.append("I couldn't find grounding data that answers this question — try asking about "
                      "stock-outs, slow movers, sales trends, or a specific product.")
    return "\n\n".join(parts)


def answer_question(question: str, cat: Catalogue) -> dict:
    grounding = build_grounding(question, cat)
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return {
            "answer": _template_fallback(question, grounding) +
                      "\n\n(LLM phrasing unavailable: GEMINI_API_KEY not set — showing raw grounded data instead.)",
            "grounding": grounding,
            "llm_used": False,
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_INSTRUCTIONS)
        prompt = (
            f"Manager's question: {question}\n\n"
            f"Grounding data (JSON):\n{json.dumps(grounding, indent=2)}\n\n"
            "Answer the manager's question using only this data."
        )
        response = model.generate_content(prompt, request_options={"timeout": 30})
        text = (response.text or "").strip()
        if not text:
            raise ValueError("empty response from model")
        return {"answer": text, "grounding": grounding, "llm_used": True}
    except Exception as exc:  # network error, bad key, quota, etc.
        return {
            "answer": _template_fallback(question, grounding) +
                      f"\n\n(LLM call failed ({exc.__class__.__name__}) — showing raw grounded data instead.)",
            "grounding": grounding,
            "llm_used": False,
        }
