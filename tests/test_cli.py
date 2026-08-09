import json

import numpy as np
import pandas as pd

from quarm.cli import main


def test_cli_writes_self_describing_json(tmp_path):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 3))
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    frame["target"] = (X[:, 0] > 0).astype(int)
    source = tmp_path / "data.csv"
    output = tmp_path / "audit.json"
    frame.to_csv(source, index=False)
    code = main(
        [
            "audit",
            str(source),
            "--target",
            "target",
            "--task",
            "classification",
            "--model",
            "logistic",
            "--repeats",
            "2",
            "--bootstrap",
            "100",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())
    assert code == 0
    assert len(payload["attributions"]) == 4
    assert abs(payload["efficiency_residual"]) < 1e-12
    assert "interpretation" in payload
