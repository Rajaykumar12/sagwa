"""The agent's world: a fictional homewares retailer, "Brindle Home".

Everything is generated from a fixed seed, so the orders database and the
policy documents are identical on every machine and every run. That is what
lets golden-set answers be exact numbers rather than judgments.

`TODAY` is pinned too: "last quarter" and "within the return window" have to
mean the same thing whenever the golden set is evaluated.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta

TODAY = date(2026, 10, 1)
SEED = 7

PRODUCTS = [
    # sku, name, category, unit price (USD)
    ("BH-101", "Linen Duvet Cover", "bedding", 129.00),
    ("BH-102", "Merino Throw Blanket", "bedding", 89.50),
    ("BH-103", "Down Pillow", "bedding", 45.00),
    ("BH-201", "Oak Side Table", "furniture", 249.00),
    ("BH-202", "Walnut Bookshelf", "furniture", 589.00),
    ("BH-203", "Rattan Lounge Chair", "furniture", 412.00),
    ("BH-301", "Stoneware Dinner Set", "kitchen", 98.00),
    ("BH-302", "Cast Iron Skillet", "kitchen", 54.00),
    ("BH-303", "Copper Kettle", "kitchen", 72.00),
    ("BH-401", "Brass Floor Lamp", "lighting", 199.00),
    ("BH-402", "Ceramic Table Lamp", "lighting", 86.00),
    ("BH-501", "Wool Area Rug", "decor", 340.00),
]

REGIONS = ["North", "South", "East", "West"]
CUSTOMERS = [
    "Asha Rao", "Ben Okafor", "Chloe Martin", "Dev Patel", "Elena Rossi", "Farid Haddad",
    "Grace Kim", "Hiro Tanaka", "Isla Murphy", "Jonas Berg", "Kavya Nair", "Liam Walsh",
    "Maya Cohen", "Noah Smith", "Olivia Brown", "Priya Shah", "Quinn Lee", "Rosa Diaz",
]

# Promised shipping time per region, in days from order. Also stated in the
# shipping policy document, so the two sources agree.
SHIPPING_DAYS = {"North": 3, "South": 5, "East": 4, "West": 6}

# Policy documents: the corpus `corpus_search` indexes. Numbers here are the
# ground truth for policy questions, and some must be combined with order data.
DOCUMENTS = {
    "returns": "Returns policy. Items can be returned within 30 days of delivery for a full refund. "
               "Furniture can be returned within 14 days of delivery and a 15% restocking fee applies. "
               "Items marked final sale cannot be returned. Refunds are issued to the original payment method within 5 business days.",
    "shipping-times": "Shipping times. Orders ship from our warehouse and the promised delivery window depends on region: "
                      "North 3 days, East 4 days, South 5 days, West 6 days, counted from the order date.",
    "late-delivery": "Late delivery compensation. If an order ships later than its promised ship date, the customer is owed "
                     "a credit of 10% of the order total. If it is more than 7 days late, the credit is 20% of the order total.",
    "warranty": "Warranty. Furniture carries a 5-year warranty against manufacturing defects. Lighting carries a 2-year warranty. "
                "Bedding, kitchen and decor items carry a 1-year warranty. Warranty does not cover wear and tear or misuse.",
    "loyalty": "Loyalty program. Members of Brindle Circle receive 10% off every order. Members who spent over $1,000 "
               "in the previous calendar year are Gold members and receive 15% off instead.",
    "price-match": "Price match. We match a lower advertised price from a major retailer within 14 days of purchase. "
                   "Price match does not apply to clearance or final-sale items.",
    "cancellation": "Cancellations. An order can be cancelled free of charge until it ships. After shipping, it must be returned instead.",
    "gift-cards": "Gift cards. Gift cards never expire and can be used online or in store. They cannot be exchanged for cash.",
    "care-linen": "Care guide: linen. Machine wash cold, tumble dry low. Linen softens with every wash.",
    "care-wood": "Care guide: oak and walnut furniture. Dust weekly and oil every six months. Keep out of direct sunlight.",
    "care-cast-iron": "Care guide: cast iron. Hand wash, dry immediately and rub with a thin layer of oil. Never use a dishwasher.",
    "stores": "Stores. Brindle Home has stores in Portland, Austin and Boston, open 10am to 7pm Monday to Saturday.",
    "support-hours": "Customer support is available by chat and phone from 8am to 8pm Eastern, seven days a week.",
    "assembly": "Assembly. Furniture over $400 includes free white-glove assembly. Other furniture can be assembled for a $49 fee.",
    "sustainability": "Sustainability. Our wool and linen are certified organic, and all packaging is recyclable.",
}


def _orders() -> list[tuple]:
    """~240 orders from 2026-01-02 to 2026-09-28, deterministic."""
    rng = random.Random(SEED)
    rows = []
    order_id = 1001
    day = date(2026, 1, 2)
    while day <= date(2026, 9, 28):
        for _ in range(rng.choice([0, 1, 1, 1, 2])):
            sku, _, _, price = rng.choice(PRODUCTS)
            region = rng.choice(REGIONS)
            quantity = rng.choice([1, 1, 1, 2, 3])
            promised_ship = day + timedelta(days=SHIPPING_DAYS[region] - 2)
            roll = rng.random()
            if roll < 0.06:
                status, shipped, delivered = "cancelled", None, None
            else:
                # About one in five ships late; a few by more than a week.
                delay = 0 if rng.random() < 0.8 else rng.choice([1, 2, 3, 5, 9, 12])
                shipped = promised_ship + timedelta(days=delay)
                delivered = shipped + timedelta(days=2)
                if shipped > TODAY:
                    status, shipped, delivered = "processing", None, None
                elif delivered > TODAY:
                    status, delivered = "shipped", None
                elif roll < 0.12:
                    status = "returned"
                else:
                    status = "delivered"
            rows.append((order_id, rng.choice(CUSTOMERS), sku, quantity, round(price * quantity, 2), region,
                         day.isoformat(), promised_ship.isoformat(),
                         shipped.isoformat() if shipped else None,
                         delivered.isoformat() if delivered else None, status))
            order_id += 1
        day += timedelta(days=1)
    return rows


SCHEMA = """
CREATE TABLE products (sku TEXT PRIMARY KEY, name TEXT, category TEXT, unit_price REAL);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY, customer TEXT, sku TEXT REFERENCES products(sku),
    quantity INTEGER, total REAL, region TEXT, order_date TEXT, promised_ship_date TEXT,
    shipped_date TEXT, delivered_date TEXT, status TEXT
);
"""

SCHEMA_DESCRIPTION = (
    "products(sku, name, category, unit_price) and orders(order_id, customer, sku, quantity, total, "
    "region, order_date, promised_ship_date, shipped_date, delivered_date, status). Dates are "
    "'YYYY-MM-DD' text. status is one of delivered, shipped, processing, cancelled, returned. "
    "An order shipped late when shipped_date > promised_ship_date."
)


def build_database() -> sqlite3.Connection:
    """A fresh in-memory database, read-only once built."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", _orders())
    conn.commit()
    conn.execute("PRAGMA query_only = ON")
    return conn
