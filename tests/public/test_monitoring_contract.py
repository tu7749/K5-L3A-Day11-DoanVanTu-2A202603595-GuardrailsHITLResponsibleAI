"""Cross-check CP3 monitoring against the independent results and audit artifacts."""
import json
from pathlib import Path

import pytest


def test_metrics_include_all_four_checkpoint_groups():
    output = Path(__file__).resolve().parents[2] / "outputs"
    if not (output / "results.json").exists():
        pytest.skip("Run CP3 to generate Blue artifacts")
    results = json.loads((output / "results.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    audit = json.loads((output / "audit_log.json").read_text())

    expected = {
        group: (len(results[group]), sum(row["blocked"] for row in results[group]))
        for group in ("safe_queries", "attack_queries", "edge_cases")
    }
    burst = results["rate_limit"]
    expected["rate_limit"] = (burst["sent"], burst["blocked"])
    for group, (total, blocked) in expected.items():
        assert metrics["by_group"][group] == {
            "total": total, "blocked": blocked, "block_rate": pytest.approx(blocked / total)
        }
    total = sum(n for n, _ in expected.values())
    blocked = sum(b for _, b in expected.values())
    assert metrics["total_requests"] == total == len({row["request_id"] for row in audit}) == len(audit)
    assert metrics["blocked_requests"] == blocked == sum(row["blocked"] for row in audit)
    assert metrics["block_rate"] == pytest.approx(blocked / total)
    assert metrics["rate_limit_hits"] == sum(row["layer"] == "rate_limiter" for row in audit)
