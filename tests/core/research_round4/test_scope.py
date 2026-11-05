"""Backtest-only scope retained after removing external delivery jobs."""
import copy
import json
from pathlib import Path
import pytest
from src.research_round4.run import job_plan

def test_preserved_four_origins_and_two_models():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs/research_round4/protocol.json").read_text())
    jobs = job_plan(config)
    assert jobs == [{"task":"backtest","origin":o,"model":m} for o in ["2023-12","2024-06","2024-09","2024-12"] for m in ["B0","C3"]]
    assert len(jobs)*2 == 16
    assert "final" not in config
    altered = copy.deepcopy(config)
    altered["new_origins"].append("2024-01")
    with pytest.raises(ValueError): job_plan(altered)
