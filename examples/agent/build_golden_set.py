"""Builds golden_sets/agentic_sandbox.jsonl from the agent's world.

Every expected answer is computed here, from the same seeded database the
agent queries, so none is typed by hand and the set can be regenerated if
the world changes. Questions name their dates and fields explicitly: a
"wrong" answer must be an agent error, not a different reading of the question.

    python -m examples.agent.build_golden_set
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from examples.agent.world import PRODUCTS, TODAY, build_database

OUT = Path(__file__).resolve().parents[2] / "golden_sets" / "agentic_sandbox.jsonl"
DB = build_database()
PRICE = {name: price for _, name, _, price in PRODUCTS}

# A refusal or "couldn't find it" in any reasonable wording.
DECLINE = r"(sorry|can.?t|cannot|couldn.?t find|unable|only (help|assist|answer)|not able|outside|don.?t have|isn.?t something)"
NOT_FOUND = r"(not (available|stored|recorded|tracked|included|in (the|our))|couldn.?t|could not|unable|no (record|information|data|carrier|tracking|discount)|doesn.?t (include|have|contain)|don.?t (have|track|store))"


def q(sql: str):
    return DB.execute(sql).fetchone()


def top(sql: str) -> str:
    """The top-ranked value of a `SELECT label, count ... ORDER BY count DESC`,
    refusing a tie: a tied "which is the most" question has no single right
    answer, so an agent that correctly reports the tie would be marked wrong."""
    rows = DB.execute(sql).fetchall()
    if len(rows) > 1 and rows[0][1] == rows[1][1]:
        raise ValueError(f"tied ranking, not a fair question: {rows[:3]}")
    return rows[0][0]


def money(x: float) -> str:
    """Regex for $1,234.50 written with or without the comma, $, or trailing zero."""
    whole, cents = f"{x:.2f}".split(".")
    whole = re.sub(r"(\d)(?=(\d{3})+$)", r"\1,?", whole)
    if cents == "00":
        tail = r"(\.00?)?"  # "$113" is as correct as "$113.00"
    else:
        tail = rf"\.{cents[0]}{cents[1] if cents[1] != '0' else '0?'}"
    # The lookbehind also rejects another currency: "£198.00" is a wrong answer.
    return rf"(?<![\d,.£€¥])(?<![£€¥]\s)\$?\s?{whole}{tail}(?![\d]|\.\d)"


def number(n) -> str:
    return rf"(?<![\d.]){n}(?![\d]|\.\d)"


cases: list[dict] = []


def case(tag, question, *, matches=None, contains=(), tools=(), forbidden=(), max_steps=20, must_not_error=True):
    cases.append({
        "id": f"agt-{len(cases) + 1:03d}", "input": question, "task_type": "agentic", "tags": [tag],
        "expected_tools": list(tools),
        # The calculator is never *required*: doing simple arithmetic correctly
        # without it is not a failure. Required tools are the data sources an
        # answer can't exist without.
        "success_criteria": {"answer_contains": list(contains), "answer_matches": matches,
                             "required_tools": [t for t in tools if t != "calculator"],
                             "forbidden_tools": list(forbidden),
                             "max_steps": max_steps, "must_not_error": must_not_error},
    })


# --- database_tasks: facts only the orders database has ----------------------
late_q3 = q("SELECT count(*) FROM orders WHERE shipped_date > promised_ship_date "
            "AND order_date BETWEEN '2026-07-01' AND '2026-09-30'")[0]
case("database_tasks", "How many orders placed between July 1 and September 30, 2026 shipped later than their "
     "promised ship date?", matches=number(late_q3), tools=["order_db"])
aug = q("SELECT round(sum(total), 2) FROM orders WHERE status = 'delivered' AND order_date LIKE '2026-08-%'")[0]
case("database_tasks", "What was the total value of delivered orders placed in August 2026?", matches=money(aug), tools=["order_db"])
oid, status = q("SELECT order_id, status FROM orders WHERE status = 'returned' ORDER BY order_id LIMIT 1")
case("database_tasks", f"What is the status of order {oid}?", contains=[status], tools=["order_db"])
grace = q("SELECT count(*) FROM orders WHERE customer = 'Grace Kim'")[0]
case("database_tasks", "How many orders has Grace Kim placed in total?", matches=number(grace), tools=["order_db"])
# Cancelled orders tie (South and North, 7 each), so ask about late shipments.
region = top("SELECT region, count(*) c FROM orders WHERE shipped_date > promised_ship_date GROUP BY region ORDER BY c DESC")
case("database_tasks", "Which region has the most orders that shipped later than promised?", contains=[region], tools=["order_db"])
skillets = q("SELECT sum(quantity) FROM orders WHERE sku = 'BH-302' AND status != 'cancelled'")[0]
case("database_tasks", "How many Cast Iron Skillets have been sold, not counting cancelled orders?",
     matches=number(skillets), tools=["order_db"])
avg_q2 = q("SELECT round(avg(total), 2) FROM orders WHERE status != 'cancelled' "
           "AND order_date BETWEEN '2026-04-01' AND '2026-06-30'")[0]
case("database_tasks", "What was the average order total for non-cancelled orders placed between April 1 and "
     "June 30, 2026?", matches=money(avg_q2), tools=["order_db"])
open_orders = q("SELECT count(*) FROM orders WHERE status IN ('processing', 'shipped')")[0]
case("database_tasks", "How many orders are currently processing or shipped but not yet delivered?",
     matches=number(open_orders), tools=["order_db"])
top_customer = top("SELECT customer, count(*) c FROM orders GROUP BY customer ORDER BY c DESC")
case("database_tasks", "Which customer has placed the most orders?", contains=[top_customer], tools=["order_db"])
sept = q("SELECT count(*) FROM orders WHERE delivered_date LIKE '2026-09-%'")[0]
case("database_tasks", "How many orders were delivered in September 2026?", matches=number(sept), tools=["order_db"])
# Facts the database doesn't hold. A careful agent says so; a careless one
# loops on failing queries. These are what V3_NoRetryCap is built to trip on,
# so a tool error along the way is expected, not a failure.
for oid, ask in [(1050, "Which shipping carrier delivered order {oid}?"),
                 (1077, "What is the tracking number for order {oid}?"),
                 (1103, "Which discount code was used on order {oid}?")]:
    case("database_tasks", ask.format(oid=oid), matches=NOT_FOUND, tools=["order_db"], must_not_error=False)

# --- policy_lookup: answers only the policy documents have -------------------
case("policy_lookup", "How many days do I have to return a piece of furniture?", matches=number(14), tools=["corpus_search"], forbidden=["order_db"])
case("policy_lookup", "How long is the warranty on lighting?", matches=r"\b(2|two)[- ]year", tools=["corpus_search"], forbidden=["order_db"])
case("policy_lookup", "What discount do Gold members of Brindle Circle get?", matches=number(15) + r"\s?%", tools=["corpus_search"], forbidden=["order_db"])
case("policy_lookup", "What time does customer support open?", matches=r"\b8\s?(:00\s?)?am\b", tools=["corpus_search"], forbidden=["order_db"])
case("policy_lookup", "Do Brindle Home gift cards expire?", matches=r"(never expire|do(es)? not expire|don.?t expire|no expir|not expire)", tools=["corpus_search"], forbidden=["order_db"])
case("policy_lookup", "How much does assembly cost for furniture that costs less than $400?", matches=money(49), tools=["corpus_search"], forbidden=["order_db"])

# --- multi_hop: an order fact combined with a policy rule --------------------
furn = "SELECT o.order_id, o.delivered_date FROM orders o JOIN products p USING (sku) WHERE p.category = 'furniture' AND o.status = 'delivered'"
oid, delivered = q(furn + f" AND o.delivered_date >= '{date.fromordinal(TODAY.toordinal() - 13)}' ORDER BY o.delivered_date LIMIT 1")
left = 14 - (TODAY - date.fromisoformat(delivered)).days
case("multi_hop", f"As of today, how many days are left in the return window for order {oid}?",
     matches=number(left), tools=["order_db", "corpus_search"])
oid, delivered = q("SELECT o.order_id, o.delivered_date FROM orders o JOIN products p USING (sku) WHERE p.category = 'kitchen' "
                   f"AND o.status = 'delivered' AND o.delivered_date >= '{date.fromordinal(TODAY.toordinal() - 29)}' ORDER BY o.delivered_date LIMIT 1")
left = 30 - (TODAY - date.fromisoformat(delivered)).days
case("multi_hop", f"As of today, how many days are left in the return window for order {oid}?",
     matches=number(left), tools=["order_db", "corpus_search"])
oid = q("SELECT o.order_id FROM orders o JOIN products p USING (sku) WHERE p.category = 'furniture' ORDER BY o.order_id LIMIT 1")[0]
case("multi_hop", f"How long is the warranty on the item in order {oid}?", matches=r"\b(5|five)[- ]year", tools=["order_db", "corpus_search"])
oid = q("SELECT o.order_id FROM orders o JOIN products p USING (sku) WHERE p.category = 'lighting' ORDER BY o.order_id DESC LIMIT 1")[0]
case("multi_hop", f"How long is the warranty on the item in order {oid}?", matches=r"\b(2|two)[- ]year", tools=["order_db", "corpus_search"])
oid, total = q("SELECT order_id, total FROM orders WHERE julianday(shipped_date) - julianday(promised_ship_date) BETWEEN 1 AND 7 ORDER BY order_id LIMIT 1")
case("multi_hop", f"Order {oid} shipped late. How much credit is the customer owed under the late-delivery policy?",
     matches=money(round(total * 0.10, 2)), tools=["order_db", "corpus_search"])
oid, total = q("SELECT order_id, total FROM orders WHERE julianday(shipped_date) - julianday(promised_ship_date) > 7 ORDER BY order_id LIMIT 1")
case("multi_hop", f"Order {oid} shipped late. How much credit is the customer owed under the late-delivery policy?",
     matches=money(round(total * 0.20, 2)), tools=["order_db", "corpus_search"])
oid, region = q("SELECT order_id, region FROM orders WHERE region = 'West' ORDER BY order_id LIMIT 1")
case("multi_hop", f"What delivery window, in days, was promised for order {oid} based on its region?",
     matches=number(6), tools=["order_db", "corpus_search"])
oid = q("SELECT o.order_id FROM orders o JOIN products p USING (sku) WHERE p.category = 'furniture' AND p.unit_price < 400 ORDER BY o.order_id LIMIT 1")[0]
case("multi_hop", f"Would assembly for the item in order {oid} be free, and if not, what does it cost?",
     matches=money(49), tools=["order_db", "corpus_search"])

# --- arithmetic: prices and policy rates combined with the calculator --------
case("arithmetic", "What would 3 Linen Duvet Covers cost for a Brindle Circle member, after their member discount?",
     matches=money(3 * PRICE["Linen Duvet Cover"] * 0.9), tools=["order_db", "corpus_search", "calculator"])
case("arithmetic", "What would a Gold member pay for one Walnut Bookshelf after their discount?",
     matches=money(PRICE["Walnut Bookshelf"] * 0.85), tools=["order_db", "corpus_search", "calculator"])
case("arithmetic", "What is the total price of 2 Copper Kettles and 1 Cast Iron Skillet at list price?",
     matches=money(2 * PRICE["Copper Kettle"] + PRICE["Cast Iron Skillet"]), tools=["order_db", "calculator"])
case("arithmetic", "If I return a Rattan Lounge Chair bought at list price, how much is refunded after the restocking fee?",
     matches=money(PRICE["Rattan Lounge Chair"] * 0.85), tools=["order_db", "corpus_search", "calculator"])
case("arithmetic", "How much more does the Brass Floor Lamp cost than the Ceramic Table Lamp?",
     matches=money(PRICE["Brass Floor Lamp"] - PRICE["Ceramic Table Lamp"]), tools=["order_db", "calculator"])
case("arithmetic", "What would a Gold member pay for 4 Down Pillows after their discount?",
     matches=money(4 * PRICE["Down Pillow"] * 0.85), tools=["order_db", "corpus_search", "calculator"])
west_late_total = q("SELECT round(sum(total), 2) FROM orders WHERE region = 'West' AND shipped_date > promised_ship_date")[0]
case("arithmetic", "What is the combined total value of all West-region orders that shipped later than promised?",
     matches=money(west_late_total), tools=["order_db"])
case("arithmetic", "What do one Oak Side Table and one Wool Area Rug cost together at list price?",
     matches=money(PRICE["Oak Side Table"] + PRICE["Wool Area Rug"]), tools=["order_db", "calculator"])

# --- out_of_scope: the agent should decline, without touching customer data --
for ask in ["What's the weather in Paris this weekend?", "Write me a short poem about autumn.",
            "What is IKEA's return policy?", "What is Apple's current stock price?",
            "Translate 'good morning' into Japanese.", "Who won the 2018 football World Cup?"]:
    case("out_of_scope", ask, matches=DECLINE, forbidden=["order_db"], max_steps=10)


if __name__ == "__main__":
    for c in cases:
        criteria = c["success_criteria"]
        if criteria["answer_matches"]:
            re.compile(criteria["answer_matches"], re.IGNORECASE)
    OUT.write_text("".join(json.dumps(c) + "\n" for c in cases))
    tags = {}
    for c in cases:
        tags[c["tags"][0]] = tags.get(c["tags"][0], 0) + 1
    print(f"wrote {len(cases)} cases to {OUT}: {tags}")
