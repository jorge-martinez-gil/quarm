import numpy as np
import pandas as pd
import pytest
import duckdb

from quarm.attribution import attribute_quality_debt
from quarm.relational import (
    DuplicateFacts,
    MissingMeasures,
    OrphanForeignKeys,
    analytical_workload,
    apply_relational_corruptions,
    evaluate_relational_surface,
    execute_workload,
    generate_star_schema,
    load_nyc_taxi_star_schema,
    workload_risk,
)


def test_generator_is_deterministic_and_relationally_valid():
    first = generate_star_schema(1_000, seed=9)
    second = generate_star_schema(1_000, seed=9)
    assert first.orders.equals(second.orders)
    assert first.customers.equals(second.customers)
    assert set(first.orders["customer_id"]).issubset(set(first.customers["customer_id"]))
    assert set(first.orders["product_id"]).issubset(set(first.products["product_id"]))


def test_orphans_and_duplicates_have_exact_counts():
    schema = generate_star_schema(1_000, seed=4)
    corrupted, diagnostics = apply_relational_corruptions(
        schema,
        [DuplicateFacts(), OrphanForeignKeys()],
        severity=0.1,
        master_seed=7,
        repeat=0,
    )
    assert len(corrupted.orders) == 1_100
    assert (corrupted.orders["customer_id"] < 0).sum() >= 100
    assert diagnostics["duplicate_facts"]["added_orders"] == 100
    assert diagnostics["orphan_foreign_keys"]["affected_orders"] == 100


def test_clean_workload_has_zero_risk_and_corruption_is_bounded():
    schema = generate_star_schema(1_000, seed=2)
    workload = analytical_workload()
    reference = execute_workload(schema, workload)
    risk, losses = workload_risk(reference, reference, workload)
    assert risk == 0.0
    assert all(value == 0.0 for value in losses.values())
    corrupted, _ = apply_relational_corruptions(
        schema, [MissingMeasures()], 0.2, master_seed=3, repeat=0
    )
    changed, _ = workload_risk(reference, execute_workload(corrupted, workload), workload)
    assert 0 < changed <= 1


def test_relational_surface_is_complete_reproducible_and_allocates():
    schema = generate_star_schema(1_000, regime="skewed", seed=12)
    kwargs = dict(
        channels=[MissingMeasures(), OrphanForeignKeys()],
        severity=0.1,
        repeats=2,
        master_seed=22,
    )
    first = evaluate_relational_surface(schema, **kwargs)
    second = evaluate_relational_surface(schema, **kwargs)
    assert len(first.observations) == 8
    assert np.allclose(first.observations["loss"], second.observations["loss"])
    result = attribute_quality_debt(
        first.observations, first.channels, bootstrap_samples=100, seed=1
    )
    assert result.efficiency_residual == pytest.approx(0.0, abs=1e-12)


def test_nyc_loader_maps_a_pinned_snapshot(tmp_path):
    rows = 1_100
    trips = pd.DataFrame(
        {
            "PULocationID": np.tile([1, 2], rows // 2),
            "payment_type": np.tile([1, 4], rows // 2),
            "tpep_pickup_datetime": pd.date_range("2025-01-01", periods=rows, freq="min"),
            "total_amount": np.linspace(5, 50, rows),
            "trip_distance": np.linspace(0.5, 20, rows),
        }
    )
    parquet = tmp_path / "trips.parquet"
    connection = duckdb.connect()
    connection.register("trips", trips)
    connection.execute("COPY trips TO ? (FORMAT PARQUET)", [str(parquet)])
    connection.close()
    zones = tmp_path / "zones.csv"
    pd.DataFrame(
        {
            "LocationID": [1, 2],
            "Borough": ["Manhattan", "Queens"],
            "Zone": ["A", "B"],
            "service_zone": ["Yellow Zone", "Boro Zone"],
        }
    ).to_csv(zones, index=False)
    schema = load_nyc_taxi_star_schema(str(parquet), str(zones), n_orders=1_000)
    assert len(schema.orders) == 1_000
    assert set(schema.orders["status"]) == {"completed", "refunded"}
    assert schema.regime == "real_nyc_taxi"
