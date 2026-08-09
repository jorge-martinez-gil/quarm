"""Relational quality-risk surfaces for analytical SQL workloads."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import duckdb
import numpy as np
import pandas as pd

from .attribution import all_coalitions, coalition_key


@dataclass(frozen=True)
class StarSchema:
    name: str
    customers: pd.DataFrame
    products: pd.DataFrame
    orders: pd.DataFrame
    regime: str


@dataclass(frozen=True)
class QuerySpec:
    name: str
    sql: str
    keys: tuple[str, ...]
    values: tuple[str, ...]


@dataclass(frozen=True)
class RelationalStudyResult:
    observations: pd.DataFrame
    channels: tuple[str, ...]
    severity: float
    repeats: int
    master_seed: int
    workload: tuple[str, ...]


class RelationalCorruption(Protocol):
    name: str
    priority: int

    def apply(
        self, schema: StarSchema, severity: float, rng: np.random.Generator
    ) -> tuple[StarSchema, dict[str, float | int | str]]: ...


def _count(severity: float, population: int) -> int:
    if not 0 <= severity <= 1:
        raise ValueError("severity must be in [0,1]")
    if severity == 0 or population == 0:
        return 0
    return min(population, max(1, int(round(severity * population))))


def _copy_schema(schema: StarSchema, *, customers=None, products=None, orders=None) -> StarSchema:
    return StarSchema(
        schema.name,
        schema.customers.copy() if customers is None else customers,
        schema.products.copy() if products is None else products,
        schema.orders.copy() if orders is None else orders,
        schema.regime,
    )


@dataclass(frozen=True)
class MissingMeasures:
    name: str = "missing_measures"
    priority: int = 10

    def apply(self, schema: StarSchema, severity: float, rng: np.random.Generator):
        orders = schema.orders.copy()
        n = _count(severity, len(orders))
        if n:
            rows = rng.choice(len(orders), n, replace=False)
            orders.loc[orders.index[rows], "unit_price"] = np.nan
        return _copy_schema(schema, orders=orders), {"affected_orders": n, "realized_rate": n / max(1, len(orders))}


@dataclass(frozen=True)
class ValueNoise:
    sigma: float = 1.0
    name: str = "value_noise"
    priority: int = 20

    def apply(self, schema: StarSchema, severity: float, rng: np.random.Generator):
        orders = schema.orders.copy()
        n = _count(severity, len(orders))
        if n:
            rows = rng.choice(len(orders), n, replace=False)
            loc = orders.index[rows]
            orders.loc[loc, "unit_price"] *= np.exp(rng.normal(0.0, self.sigma, n))
        return _copy_schema(schema, orders=orders), {
            "affected_orders": n,
            "realized_rate": n / max(1, len(orders)),
            "log_sigma": self.sigma,
        }


@dataclass(frozen=True)
class OrphanForeignKeys:
    name: str = "orphan_foreign_keys"
    priority: int = 30

    def apply(self, schema: StarSchema, severity: float, rng: np.random.Generator):
        orders = schema.orders.copy()
        n = _count(severity, len(orders))
        if n:
            rows = rng.choice(len(orders), n, replace=False)
            orders.loc[orders.index[rows], "customer_id"] = -(rows.astype(np.int64) + 1)
        return _copy_schema(schema, orders=orders), {"affected_orders": n, "realized_rate": n / max(1, len(orders))}


@dataclass(frozen=True)
class StaleTimestamps:
    days: int = 365
    name: str = "stale_timestamps"
    priority: int = 40

    def apply(self, schema: StarSchema, severity: float, rng: np.random.Generator):
        orders = schema.orders.copy()
        n = _count(severity, len(orders))
        if n:
            rows = rng.choice(len(orders), n, replace=False)
            loc = orders.index[rows]
            orders.loc[loc, "order_date"] = orders.loc[loc, "order_date"] - pd.to_timedelta(self.days, unit="D")
        return _copy_schema(schema, orders=orders), {
            "affected_orders": n,
            "realized_rate": n / max(1, len(orders)),
            "days_shifted": self.days,
        }


@dataclass(frozen=True)
class DimensionDrift:
    name: str = "dimension_drift"
    priority: int = 50

    def apply(self, schema: StarSchema, severity: float, rng: np.random.Generator):
        customers = schema.customers.copy()
        n = _count(severity, len(customers))
        impacted_orders = 0
        if n:
            rows = rng.choice(len(customers), n, replace=False)
            customer_ids = customers.loc[customers.index[rows], "customer_id"]
            impacted_orders = int(schema.orders["customer_id"].isin(customer_ids).sum())
            regions = np.array(sorted(customers["region"].unique()))
            mapping = {region: regions[(idx + 1) % len(regions)] for idx, region in enumerate(regions)}
            loc = customers.index[rows]
            customers.loc[loc, "region"] = customers.loc[loc, "region"].map(mapping)
        return _copy_schema(schema, customers=customers), {
            "affected_customers": n,
            "realized_rate": n / max(1, len(customers)),
            "impacted_orders": impacted_orders,
            "fact_amplification": impacted_orders / max(1, n),
        }


@dataclass(frozen=True)
class DuplicateFacts:
    name: str = "duplicate_facts"
    priority: int = 100

    def apply(self, schema: StarSchema, severity: float, rng: np.random.Generator):
        orders = schema.orders.copy()
        n = _count(severity, len(orders))
        if n:
            rows = rng.choice(len(orders), n, replace=True)
            orders = pd.concat([orders, orders.iloc[rows]], ignore_index=True)
        return _copy_schema(schema, orders=orders), {
            "added_orders": n,
            "realized_rate": n / max(1, len(schema.orders)),
        }


def default_relational_corruptions() -> list[RelationalCorruption]:
    return [
        MissingMeasures(),
        ValueNoise(),
        OrphanForeignKeys(),
        StaleTimestamps(),
        DimensionDrift(),
        DuplicateFacts(),
    ]


def generate_star_schema(
    n_orders: int = 100_000,
    *,
    regime: str = "uniform",
    seed: int = 2027,
    name: str | None = None,
) -> StarSchema:
    """Generate a reproducible relational benchmark with controlled skew/seasonality."""
    if n_orders < 1_000:
        raise ValueError("n_orders must be at least 1,000")
    if regime not in {"uniform", "skewed", "seasonal"}:
        raise ValueError("regime must be uniform, skewed, or seasonal")
    rng = np.random.default_rng(seed)
    n_customers = max(1_000, n_orders // 20)
    n_products = max(250, n_orders // 200)
    regions = np.array(["north", "south", "east", "west", "central"])
    segments = np.array(["consumer", "small_business", "enterprise"])
    categories = np.array(["office", "electronics", "home", "outdoor", "health", "apparel"])
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(n_customers, dtype=np.int64),
            "region": rng.choice(regions, n_customers, p=[0.18, 0.20, 0.22, 0.24, 0.16]),
            "segment": rng.choice(segments, n_customers, p=[0.68, 0.23, 0.09]),
        }
    )
    base_prices = rng.lognormal(mean=3.0, sigma=0.8, size=n_products)
    products = pd.DataFrame(
        {
            "product_id": np.arange(n_products, dtype=np.int64),
            "category": rng.choice(categories, n_products),
            "base_price": base_prices,
        }
    )
    if regime == "skewed":
        ranks = np.arange(1, n_customers + 1, dtype=float)
        customer_weights = 1.0 / np.power(ranks, 1.12)
        customer_weights /= customer_weights.sum()
        customer_ids = rng.choice(n_customers, n_orders, p=customer_weights)
    else:
        customer_ids = rng.integers(0, n_customers, n_orders)
    product_ids = rng.integers(0, n_products, n_orders)
    if regime == "seasonal":
        days = np.arange(730)
        weights = 1.2 + np.sin(2 * np.pi * days / 365.0) + 0.7 * (days > 650)
        weights /= weights.sum()
        day_offsets = rng.choice(days, n_orders, p=weights)
    else:
        day_offsets = rng.integers(0, 730, n_orders)
    quantity = np.minimum(rng.geometric(0.42, n_orders), 12).astype(np.int16)
    unit_price = base_prices[product_ids] * rng.lognormal(0.0, 0.12, n_orders)
    discount = rng.choice([0.0, 0.05, 0.10, 0.20, 0.30], n_orders, p=[0.44, 0.16, 0.22, 0.13, 0.05])
    status = rng.choice(["completed", "refunded", "cancelled"], n_orders, p=[0.91, 0.055, 0.035])
    orders = pd.DataFrame(
        {
            "order_id": np.arange(n_orders, dtype=np.int64),
            "customer_id": customer_ids.astype(np.int64),
            "product_id": product_ids.astype(np.int64),
            "order_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(day_offsets, unit="D"),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "status": status,
        }
    )
    return StarSchema(name or f"retail_{regime}_{n_orders}", customers, products, orders, regime)


def load_nyc_taxi_star_schema(
    parquet_path: str,
    zone_lookup_path: str,
    *,
    n_orders: int = 100_000,
) -> StarSchema:
    """Map an official NYC TLC Yellow Taxi snapshot to the workload schema.

    Pickup zones become the location dimension and payment codes become the
    second dimension. A deterministic hash sample avoids order-dependent LIMIT.
    """
    if n_orders < 1_000:
        raise ValueError("n_orders must be at least 1,000")
    zones = pd.read_csv(zone_lookup_path).rename(
        columns={"LocationID": "customer_id", "Borough": "region", "service_zone": "segment"}
    )
    required_zones = {"customer_id", "region", "segment"}
    if not required_zones.issubset(zones.columns):
        raise ValueError(f"zone lookup missing columns: {sorted(required_zones - set(zones.columns))}")
    customers = zones[["customer_id", "region", "segment"]].copy()
    customers["customer_id"] = customers["customer_id"].astype(np.int64)
    customers["region"] = customers["region"].fillna("Unknown").astype(str)
    customers["segment"] = customers["segment"].fillna("Unknown").astype(str)
    payment_labels = {
        0: "unknown",
        1: "credit_card",
        2: "cash",
        3: "no_charge",
        4: "dispute",
        5: "unknown",
        6: "voided",
    }
    products = pd.DataFrame(
        {
            "product_id": np.array(list(payment_labels), dtype=np.int64),
            "category": list(payment_labels.values()),
            "base_price": np.ones(len(payment_labels)),
        }
    )
    connection = duckdb.connect(":memory:")
    try:
        orders = connection.execute(
            """
            SELECT
              row_number() OVER () - 1 AS order_id,
              CAST(PULocationID AS BIGINT) AS customer_id,
              CAST(COALESCE(payment_type, 0) AS BIGINT) AS product_id,
              tpep_pickup_datetime AS order_date,
              CAST(1 AS SMALLINT) AS quantity,
              CAST(total_amount AS DOUBLE) AS unit_price,
              CAST(0.0 AS DOUBLE) AS discount,
              CASE
                WHEN payment_type = 4 THEN 'refunded'
                WHEN payment_type = 6 THEN 'cancelled'
                ELSE 'completed'
              END AS status
            FROM read_parquet(?)
            WHERE tpep_pickup_datetime >= TIMESTAMP '2025-01-01'
              AND tpep_pickup_datetime < TIMESTAMP '2025-02-01'
              AND PULocationID BETWEEN 1 AND 265
              AND total_amount BETWEEN 0 AND 1000
            ORDER BY hash(tpep_pickup_datetime, PULocationID, total_amount, trip_distance)
            LIMIT ?
            """,
            [str(parquet_path), n_orders],
        ).fetchdf()
    finally:
        connection.close()
    if len(orders) < n_orders:
        raise ValueError(f"only {len(orders)} valid TLC rows available; requested {n_orders}")
    return StarSchema("nyc_taxi_2025_01", customers, products, orders, "real_nyc_taxi")


def analytical_workload() -> tuple[QuerySpec, ...]:
    revenue = "o.quantity * o.unit_price * (1.0 - o.discount)"
    return (
        QuerySpec("revenue_region", f"SELECT c.region, SUM({revenue}) AS revenue FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.region ORDER BY c.region", ("region",), ("revenue",)),
        QuerySpec("revenue_month", f"SELECT date_trunc('month', o.order_date) AS month, SUM({revenue}) AS revenue FROM orders o WHERE o.status='completed' GROUP BY month ORDER BY month", ("month",), ("revenue",)),
        QuerySpec("revenue_category", f"SELECT p.category, SUM({revenue}) AS revenue FROM orders o JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY p.category ORDER BY p.category", ("category",), ("revenue",)),
        QuerySpec("aov_segment", f"SELECT c.segment, AVG({revenue}) AS avg_value FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.segment ORDER BY c.segment", ("segment",), ("avg_value",)),
        QuerySpec("refund_region", "SELECT c.region, AVG(CASE WHEN o.status='refunded' THEN 1.0 ELSE 0.0 END) AS refund_rate FROM orders o JOIN customers c USING(customer_id) GROUP BY c.region ORDER BY c.region", ("region",), ("refund_rate",)),
        QuerySpec("orders_status", "SELECT status, COUNT(*)::DOUBLE AS order_count FROM orders GROUP BY status ORDER BY status", ("status",), ("order_count",)),
        QuerySpec("customers_region", "SELECT c.region, COUNT(DISTINCT o.customer_id)::DOUBLE AS customers FROM orders o JOIN customers c USING(customer_id) GROUP BY c.region ORDER BY c.region", ("region",), ("customers",)),
        QuerySpec("daily_revenue", f"SELECT CAST(o.order_date AS DATE) AS day, SUM({revenue}) AS revenue FROM orders o WHERE o.status='completed' GROUP BY day ORDER BY day", ("day",), ("revenue",)),
    )


def execute_workload(schema: StarSchema, workload: tuple[QuerySpec, ...] | None = None) -> dict[str, pd.DataFrame]:
    workload = analytical_workload() if workload is None else workload
    connection = duckdb.connect(":memory:")
    try:
        connection.register("customers", schema.customers)
        connection.register("products", schema.products)
        connection.register("orders", schema.orders)
        return {query.name: connection.execute(query.sql).fetchdf() for query in workload}
    finally:
        connection.close()


def query_answer_loss(reference: pd.DataFrame, observed: pd.DataFrame, query: QuerySpec) -> float:
    """Bounded normalized L1 answer error after outer alignment on group keys."""
    left = reference.copy()
    right = observed.copy()
    for key in query.keys:
        left[key] = left[key].astype(str)
        right[key] = right[key].astype(str)
    if query.keys:
        joined = left.merge(right, on=list(query.keys), how="outer", suffixes=("_ref", "_obs"))
    else:
        joined = pd.concat([left.add_suffix("_ref"), right.add_suffix("_obs")], axis=1)
    errors = []
    for value in query.values:
        truth = joined[f"{value}_ref"].fillna(0.0).to_numpy(dtype=float)
        actual = joined[f"{value}_obs"].fillna(0.0).to_numpy(dtype=float)
        denominator = np.abs(truth).sum() + 1e-12
        errors.append(min(1.0, float(np.abs(actual - truth).sum() / denominator)))
    return float(np.mean(errors))


def workload_risk(reference_answers: dict[str, pd.DataFrame], observed_answers: dict[str, pd.DataFrame], workload=None) -> tuple[float, dict[str, float]]:
    workload = analytical_workload() if workload is None else workload
    losses = {
        query.name: query_answer_loss(reference_answers[query.name], observed_answers[query.name], query)
        for query in workload
    }
    return float(np.mean(list(losses.values()))), losses


def _channel_seed(master_seed: int, repeat: int, channel_name: str) -> int:
    payload = f"quarm-relational|{master_seed}|{repeat}|{channel_name}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "little")


def apply_relational_corruptions(
    schema: StarSchema,
    channels: list[RelationalCorruption] | tuple[RelationalCorruption, ...],
    severity: float,
    master_seed: int,
    repeat: int,
) -> tuple[StarSchema, dict[str, dict[str, float | int | str]]]:
    names = [channel.name for channel in channels]
    if len(names) != len(set(names)):
        raise ValueError("relational corruption names must be unique")
    out = _copy_schema(schema)
    diagnostics = {}
    for channel in sorted(channels, key=lambda item: (item.priority, item.name)):
        rng = np.random.default_rng(_channel_seed(master_seed, repeat, channel.name))
        out, info = channel.apply(out, severity, rng)
        diagnostics[channel.name] = info
    return out, diagnostics


def evaluate_relational_surface(
    schema: StarSchema,
    channels: list[RelationalCorruption] | tuple[RelationalCorruption, ...],
    *,
    severity: float = 0.10,
    repeats: int = 10,
    master_seed: int = 2027,
) -> RelationalStudyResult:
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    if len(channels) > 10:
        raise ValueError("exact evaluation is limited to 10 channels")
    names = tuple(channel.name for channel in channels)
    lookup = {channel.name: channel for channel in channels}
    workload = analytical_workload()
    reference = execute_workload(schema, workload)
    rows = []
    for repeat in range(repeats):
        for coalition in all_coalitions(names):
            selected = [lookup[name] for name in names if name in coalition]
            if selected:
                corrupted, diagnostics = apply_relational_corruptions(
                    schema, selected, severity, master_seed, repeat
                )
                answers = execute_workload(corrupted, workload)
                risk, query_losses = workload_risk(reference, answers, workload)
            else:
                diagnostics = {}
                query_losses = {query.name: 0.0 for query in workload}
                risk = 0.0
            rows.append(
                {
                    "repeat": repeat,
                    "coalition": coalition,
                    "coalition_key": coalition_key(coalition),
                    "coalition_size": len(coalition),
                    "loss": risk,
                    "orders": len(schema.orders),
                    "diagnostics": diagnostics,
                    "query_losses": query_losses,
                }
            )
    return RelationalStudyResult(
        pd.DataFrame(rows), names, severity, repeats, master_seed, tuple(query.name for query in workload)
    )
