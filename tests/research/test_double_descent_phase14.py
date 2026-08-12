from pathlib import Path

from research.double_descent.phase10 import (
    PHASE10_MAP_RATIOS,
    PHASE10_ROBUSTNESS_RATIOS,
    PHASE10_SEEDS,
    Phase10Config,
)
from research.double_descent.phase10 import (
    _phase3_config as _phase10_phase3_config,
)
from research.double_descent.phase12 import (
    PHASE12_BASELINES,
    Phase12Config,
)
from research.double_descent.phase12 import (
    _phase3_config as _phase12_phase3_config,
)
from research.double_descent.phase14 import (
    PHASE14_DATA_TIMERANGE,
    PHASE14_EXPECTED_CASE_COUNT,
    PHASE14_EXPECTED_FIT_COUNT,
    PHASE14_PAIR,
    Phase14Config,
    _data_audit_config,
    _rff_tasks,
    build_download_command,
)


def _config(tmp_path: Path, **overrides) -> Phase14Config:
    values = {
        "data_directory": tmp_path,
        "python_executable": str(tmp_path / "python.exe"),
        "cuda_python_executable": str(tmp_path / "cuda.exe"),
    }
    values.update(overrides)
    return Phase14Config(**values)


def test_phase14_design_is_frozen() -> None:
    assert PHASE14_PAIR == "ETH/USDT:USDT"
    assert PHASE14_DATA_TIMERANGE == "20240924-20260101"
    assert PHASE14_EXPECTED_CASE_COUNT == 29
    assert PHASE14_EXPECTED_FIT_COUNT == 377


def test_phase14_rff_grid_has_21_matched_cases(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tasks = _rff_tasks(config)

    assert tasks == [
        (PHASE10_SEEDS[0], PHASE10_MAP_RATIOS),
        (PHASE10_SEEDS[1], PHASE10_ROBUSTNESS_RATIOS),
        (PHASE10_SEEDS[2], PHASE10_ROBUSTNESS_RATIOS),
    ]
    assert sum(len(ratios) for _, ratios in tasks) == 21
    assert config.baselines == PHASE12_BASELINES


def test_data_audit_uses_eth_not_default_btc(tmp_path: Path) -> None:
    config = _config(tmp_path)
    audit = _data_audit_config(config)

    assert audit.pair == PHASE14_PAIR
    assert audit.train_periods_days == (90,)
    assert audit.timerange == "20250101-20260101"


def test_reused_phase_engines_propagate_the_replication_pair(tmp_path: Path) -> None:
    phase10_audit = _phase10_phase3_config(
        Phase10Config(data_directory=tmp_path, pair=PHASE14_PAIR)
    )
    phase12_audit = _phase12_phase3_config(
        Phase12Config(data_directory=tmp_path, pair=PHASE14_PAIR)
    )

    assert phase10_audit.pair == PHASE14_PAIR
    assert phase12_audit.pair == PHASE14_PAIR


def test_download_command_gets_every_required_futures_candle_type(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = build_download_command(config)

    assert command[:4] == [config.python_executable, "-m", "freqtrade", "download-data"]
    assert command[command.index("--pairs") + 1] == PHASE14_PAIR
    assert command[command.index("--timerange") + 1] == PHASE14_DATA_TIMERANGE
    candle_index = command.index("--candle-types")
    assert command[candle_index + 1 : candle_index + 4] == [
        "futures",
        "funding_rate",
        "mark",
    ]
    assert command[command.index("--trading-mode") + 1] == "futures"


def test_full_run_rejects_eth_parameter_selection(tmp_path: Path) -> None:
    config = _config(tmp_path, gamma=0.2)

    try:
        config._validate_design()
    except ValueError as exc:
        assert "gamma=0.5" in str(exc)
    else:
        raise AssertionError("Phase 14 accepted an ETH-selected gamma")


def test_full_run_rejects_incomplete_confirmatory_map(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        map_ratios=(0.1, 1.0, 5.0),
        robustness_ratios=(0.1, 1.0, 5.0),
    )

    try:
        config._validate_design()
    except ValueError as exc:
        assert "main P/N map" in str(exc)
    else:
        raise AssertionError("Phase 14 accepted an incomplete confirmatory map")


def test_smoke_run_allows_reduced_grid_without_changing_model_parameters(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        map_ratios=(0.1, 1.0, 5.0),
        robustness_ratios=(0.1, 1.0, 5.0),
        seeds=(PHASE10_SEEDS[0],),
        baselines=("zero_return", "market_ols", "market_ridge"),
        minimum_seed_count=1,
        smoke_test=True,
    )

    config._validate_design()
    assert sum(len(ratios) for _, ratios in _rff_tasks(config)) == 3
