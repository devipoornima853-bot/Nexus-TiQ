"""
Deterministic analytics layer.

Every number the copilot ever states comes from here — plain pandas over
the CSVs, no LLM involved. The LLM layer (src/llm.py) is only allowed to
explain and phrase what this module already computed; it never invents a
figure. This separation is deliberate so every claim in an answer can be
traced back to a specific function and a specific row of data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Catalogue:
    """Loads the CSVs once and exposes cheap, well-named queries over them.

    Nothing here calls an LLM. Every method returns plain data plus enough
    context (dates, row counts, thresholds used) to cite in an answer.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.products = pd.read_csv(data_dir / "products.csv")
        self.stores = pd.read_csv(data_dir / "stores.csv")
        self.sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["date"])
        self.stock = pd.read_csv(data_dir / "stock.csv", parse_dates=["date"])
        self.as_of = self.sales["date"].max()  # "today" for this dataset

    # ---------- lookups -------------------------------------------------

    def find_product(self, text: str) -> Optional[pd.Series]:
        text = text.lower().strip()
        exact = self.products[self.products["name"].str.lower() == text]
        if len(exact):
            return exact.iloc[0]
        contains = self.products[self.products["name"].str.lower().str.contains(text, na=False)]
        if len(contains):
            return contains.iloc[0]
        by_id = self.products[self.products["product_id"].str.lower() == text]
        if len(by_id):
            return by_id.iloc[0]
        return None

    def find_store(self, text: str) -> Optional[pd.Series]:
        text = text.lower().strip()
        match = self.stores[
            self.stores["name"].str.lower().str.contains(text, na=False)
            | self.stores["city"].str.lower().str.contains(text, na=False)
            | (self.stores["store_id"].str.lower() == text)
        ]
        return match.iloc[0] if len(match) else None

    # ---------- core metrics ---------------------------------------------

    def latest_stock(self, store_id: Optional[str] = None) -> pd.DataFrame:
        df = self.stock[self.stock["date"] == self.as_of]
        if store_id:
            df = df[df["store_id"] == store_id]
        return df

    def daily_velocity(self, product_id: str, store_id: Optional[str] = None,
                        window_days: int = 14) -> float:
        """Average units/day sold over the trailing window."""
        cutoff = self.as_of - timedelta(days=window_days)
        df = self.sales[(self.sales["product_id"] == product_id) & (self.sales["date"] > cutoff)]
        if store_id:
            df = df[df["store_id"] == store_id]
        return df["quantity_sold"].sum() / window_days if len(df) or True else 0.0

    def stockout_risks(self, store_id: Optional[str] = None, horizon_days: int = 7,
                        window_days: int = 14) -> list[dict]:
        """Products projected to run out within `horizon_days` at current
        sell-through rate, given current stock on hand."""
        stock_df = self.latest_stock(store_id)
        results = []
        for _, row in stock_df.iterrows():
            velocity = self.daily_velocity(row["product_id"], row["store_id"], window_days)
            if velocity <= 0:
                continue
            days_left = row["quantity_on_hand"] / velocity
            if days_left <= horizon_days:
                prod = self.products[self.products["product_id"] == row["product_id"]].iloc[0]
                results.append({
                    "store_id": row["store_id"],
                    "product_id": row["product_id"],
                    "product_name": prod["name"],
                    "quantity_on_hand": int(row["quantity_on_hand"]),
                    "avg_daily_sales": round(velocity, 2),
                    "days_left": round(days_left, 1),
                    "window_days": window_days,
                    "as_of": self.as_of.date().isoformat(),
                })
        return sorted(results, key=lambda r: r["days_left"])

    def slow_movers(self, store_id: Optional[str] = None, window_days: int = 30,
                     max_units_sold: int = 20) -> list[dict]:
        """Products that have barely sold in the trailing window but still
        carry meaningful stock — candidates for a markdown or a stop-order."""
        cutoff = self.as_of - timedelta(days=window_days)
        sales_window = self.sales[self.sales["date"] > cutoff]
        if store_id:
            sales_window = sales_window[sales_window["store_id"] == store_id]
        sold = sales_window.groupby(["store_id", "product_id"])["quantity_sold"].sum().reset_index()

        stock_df = self.latest_stock(store_id)
        merged = stock_df.merge(sold, on=["store_id", "product_id"], how="left")
        merged["quantity_sold"] = merged["quantity_sold"].fillna(0)

        slow = merged[(merged["quantity_sold"] <= max_units_sold) & (merged["quantity_on_hand"] > 0)]
        results = []
        for _, row in slow.iterrows():
            prod = self.products[self.products["product_id"] == row["product_id"]].iloc[0]
            results.append({
                "store_id": row["store_id"],
                "product_id": row["product_id"],
                "product_name": prod["name"],
                "quantity_on_hand": int(row["quantity_on_hand"]),
                "units_sold_in_window": int(row["quantity_sold"]),
                "window_days": window_days,
                "as_of": self.as_of.date().isoformat(),
            })
        return sorted(results, key=lambda r: -r["quantity_on_hand"])

    def sales_moves(self, store_id: Optional[str] = None, recent_days: int = 7,
                     baseline_days: int = 21, min_baseline_units: int = 5,
                     threshold_pct: float = 50.0) -> list[dict]:
        """Products whose recent sell-through has moved sharply (up or
        down) versus their own prior baseline — i.e. a spike or a drop."""
        recent_cutoff = self.as_of - timedelta(days=recent_days)
        baseline_start = recent_cutoff - timedelta(days=baseline_days)

        recent = self.sales[self.sales["date"] > recent_cutoff]
        baseline = self.sales[(self.sales["date"] > baseline_start) & (self.sales["date"] <= recent_cutoff)]
        if store_id:
            recent = recent[recent["store_id"] == store_id]
            baseline = baseline[baseline["store_id"] == store_id]

        recent_g = recent.groupby(["store_id", "product_id"])["quantity_sold"].sum()
        baseline_g = baseline.groupby(["store_id", "product_id"])["quantity_sold"].sum()

        results = []
        keys = set(recent_g.index) | set(baseline_g.index)
        for key in keys:
            r_total = recent_g.get(key, 0)
            b_total = baseline_g.get(key, 0)
            b_daily = b_total / baseline_days
            r_daily = r_total / recent_days
            baseline_units_scaled = b_daily * recent_days  # baseline scaled to the recent window
            if baseline_units_scaled < min_baseline_units and r_total < min_baseline_units:
                continue
            if baseline_units_scaled == 0:
                pct_change = 100.0 if r_total > 0 else 0.0
            else:
                pct_change = ((r_total - baseline_units_scaled) / baseline_units_scaled) * 100
            if abs(pct_change) >= threshold_pct:
                store_id_, product_id_ = key
                prod = self.products[self.products["product_id"] == product_id_].iloc[0]
                results.append({
                    "store_id": store_id_,
                    "product_id": product_id_,
                    "product_name": prod["name"],
                    "recent_units": int(r_total),
                    "recent_days": recent_days,
                    "baseline_daily_avg": round(b_daily, 2),
                    "baseline_days": baseline_days,
                    "pct_change": round(pct_change, 1),
                    "direction": "spike" if pct_change > 0 else "drop",
                    "as_of": self.as_of.date().isoformat(),
                })
        return sorted(results, key=lambda r: -abs(r["pct_change"]))

    def product_summary(self, product_id: str, store_id: Optional[str] = None,
                         window_days: int = 30) -> dict:
        cutoff = self.as_of - timedelta(days=window_days)
        df = self.sales[(self.sales["product_id"] == product_id) & (self.sales["date"] > cutoff)]
        if store_id:
            df = df[df["store_id"] == store_id]
        prod = self.products[self.products["product_id"] == product_id].iloc[0]
        stock_df = self.latest_stock(store_id)
        stock_df = stock_df[stock_df["product_id"] == product_id]
        return {
            "product_id": product_id,
            "product_name": prod["name"],
            "category": prod["category"],
            "window_days": window_days,
            "units_sold": int(df["quantity_sold"].sum()),
            "revenue": round(float(df["revenue"].sum()), 2),
            "current_stock_total": int(stock_df["quantity_on_hand"].sum()),
            "as_of": self.as_of.date().isoformat(),
        }

    def daily_briefing(self, store_id: Optional[str] = None) -> dict:
        """Everything that needs attention today, in one bundle — used for
        the dashboard view and as grounding context for open-ended chat."""
        return {
            "as_of": self.as_of.date().isoformat(),
            "store_id": store_id,
            "stockout_risks": self.stockout_risks(store_id)[:8],
            "slow_movers": self.slow_movers(store_id)[:8],
            "sales_moves": self.sales_moves(store_id)[:8],
        }
