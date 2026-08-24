from __future__ import annotations

import json
from pathlib import Path

import pytest

from otrank.experiments.nhanes_semisynthetic_invariance import (
    load_experiment_config,
    resolve_cqr_r_library,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/real_data_nhanes_semisynthetic_invariance.json"


def test_default_config_loads() -> None:
    config, document, overrides = load_experiment_config(CONFIG, environment={})
    assert (document["n"], document["p"], document["d"]) == (8237, 43, 7)
    assert config.repetitions == 10
    assert config.workers == 4
    assert config.cqr_r_library is None
    assert overrides["r_library"] is None


def test_cli_overrides_and_r_library_precedence() -> None:
    config, _, overrides = load_experiment_config(
        CONFIG,
        repetitions=3,
        workers=2,
        r_library="~/explicit-r-lib",
        environment={"CQR_R_LIB": "/env/cqr", "R_LIBS_USER": "/env/user"},
    )
    assert config.repetitions == 3
    assert config.workers == 2
    assert config.cqr_r_library == str(Path("~/explicit-r-lib").expanduser())
    assert overrides["repetitions"] == 3
    assert resolve_cqr_r_library(None, environment={"CQR_R_LIB": "/env/cqr"}) == "/env/cqr"
    assert resolve_cqr_r_library(None, environment={"R_LIBS_USER": "/env/user"}) == "/env/user"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("otrr"), "missing required config field"),
        (lambda value: value.update({"d": 6}), "requires n=8237, p=43, d=7"),
        (lambda value: value["multiple_testing"].update({"q": 1.5}), r"q in \(0,1\)"),
    ],
)
def test_invalid_config_fails(tmp_path: Path, mutation, message: str) -> None:
    value = json.loads(CONFIG.read_text())
    mutation(value)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path, environment={})


def test_config_contains_no_private_machine_path() -> None:
    assert "/private/tmp" not in CONFIG.read_text()
