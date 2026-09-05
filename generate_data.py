"""
Generates a realistic-looking 90-day synthetic dataset for a small retail
chain: stores, products, daily sales, and daily stock snapshots.

Run once (`python -m src.generate_data`) to (re)create the CSVs under data/.
The committed CSVs are what the shipped app actually uses — this script is
here for transparency and reproducibility, not run at app startup.
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

STORES = [
    ("ST01", "Anna Nagar", "Chennai"),
    ("ST02", "T Nagar", "Chennai"),
    ("ST03", "Koramangala", "Bengaluru"),
    ("ST04", "Andheri", "Mumbai"),
    ("ST05", "Salt Lake", "Kolkata"),
]

CATEGORIES = {
    "Beverages": ["Cola 500ml", "Orange Juice 1L", "Green Tea 250ml", "Energy Drink 250ml", "Mineral Water 1L"],
    "Snacks": ["Potato Chips 90g", "Salted Peanuts 200g", "Choco Cookies 150g", "Namkeen Mix 200g", "Popcorn 100g"],
    "Dairy": ["Toned Milk 500ml", "Curd 400g", "Paneer 200g", "Butter 100g", "Cheese Slices 200g"],
    "Personal Care": ["Shampoo 200ml", "Toothpaste 100g", "Soap Bar 125g", "Hand Wash 250ml", "Face Wash 100g"],
    "Household": ["Dish Wash Liquid 500ml", "Detergent Powder 1kg", "Floor Cleaner 500ml", "Toilet Cleaner 500ml", "Air Freshener 250ml"],
    "Staples": ["Rice 5kg", "Wheat Flour 5kg", "Sugar 1kg", "Cooking Oil 1L", "Toor Dal 1kg"],
}

PRODUCTS = []
pid = 1
for cat, items in CATEGORIES.items():
    for name in items:
        PRODUCTS.append({
            "product_id": f"P{pid:03d}",
            "name": name,
            "category": cat,
            "unit_cost": round(random.uniform(20, 250), 2),
            "reorder_point": random.choice([15, 20, 25, 30]),
        })
        pid += 1

# Derive a selling price with a modest margin.
for p in PRODUCTS:
    p["unit_price"] = round(p["unit_cost"] * random.uniform(1.25, 1.6), 2)

START = date(2025, 6, 8)
DAYS = 90

# Give every (store, product) a base daily demand rate so behaviour is
# consistent across the period, then layer in noise, a slow trend for a
# handful of products, and a couple of deliberate anomalies to exercise
# the "spike/drop" and "stock-out risk" logic end to end.
base_demand = {}
for s, _, _ in STORES:
    for p in PRODUCTS:
        base_demand[(s, p["product_id"])] = random.uniform(1.5, 9.0)

# Deliberately engineered scenarios (used by the demo / README):
#  - P006 (Energy Drink) at ST02: strong upward trend -> sales spike, stock-out risk
#  - P013 (Paneer) at ST03: near-zero sales for the last 30 days -> slow mover / overstock
#  - P021 (Soap Bar) at ST01: sudden 10x demand jump in the last 5 days -> spike + likely stock-out
DEMAND_TREND_UP = {("ST02", "P006")}
SLOW_MOVER = {("ST03", "P013")}
SUDDEN_SPIKE = {("ST01", "P021")}

sales_rows = []
stock_rows = []

stock_on_hand = {}
for s, _, _ in STORES:
    for p in PRODUCTS:
        stock_on_hand[(s, p["product_id"])] = random.randint(40, 120)

for day_offset in range(DAYS):
    d = START + timedelta(days=day_offset)
    for store_id, _, _ in STORES:
        for p in PRODUCTS:
            key = (store_id, p["product_id"])
            demand = base_demand[key]

            if key in DEMAND_TREND_UP:
                demand *= 1 + (day_offset / DAYS) * 2.5
            if key in SLOW_MOVER and day_offset > DAYS - 30:
                demand *= 0.05
            if key in SUDDEN_SPIKE and day_offset > DAYS - 5:
                demand *= 10

            qty_sold = max(0, round(random.gauss(demand, max(demand * 0.3, 0.5))))
            qty_sold = min(qty_sold, stock_on_hand[key])  # can't sell what isn't there

            revenue = round(qty_sold * p["unit_price"], 2)
            if qty_sold > 0:
                sales_rows.append({
                    "date": d.isoformat(),
                    "store_id": store_id,
                    "product_id": p["product_id"],
                    "quantity_sold": qty_sold,
                    "revenue": revenue,
                })

            stock_on_hand[key] -= qty_sold
            # Restock roughly weekly, or when running low.
            if stock_on_hand[key] < p["reorder_point"] or day_offset % 7 == 0:
                stock_on_hand[key] += random.randint(30, 80)

            stock_rows.append({
                "date": d.isoformat(),
                "store_id": store_id,
                "product_id": p["product_id"],
                "quantity_on_hand": stock_on_hand[key],
            })


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    write_csv(DATA_DIR / "stores.csv", [
        {"store_id": s, "name": n, "city": c} for s, n, c in STORES
    ], ["store_id", "name", "city"])

    write_csv(DATA_DIR / "products.csv", PRODUCTS,
               ["product_id", "name", "category", "unit_cost", "unit_price", "reorder_point"])

    write_csv(DATA_DIR / "sales.csv", sales_rows,
               ["date", "store_id", "product_id", "quantity_sold", "revenue"])

    write_csv(DATA_DIR / "stock.csv", stock_rows,
               ["date", "store_id", "product_id", "quantity_on_hand"])

    print(f"Wrote {len(PRODUCTS)} products, {len(STORES)} stores, "
          f"{len(sales_rows)} sales rows, {len(stock_rows)} stock rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
