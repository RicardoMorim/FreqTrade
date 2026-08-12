from pathlib import Path

import numpy as np

from research.double_descent.freqai.Phase11ShuffledLabelRegressor import (
    deterministic_label_permutation,
    permutation_fingerprint,
)
from research.double_descent.phase11 import (
    PHASE11_EXPECTED_CASE_COUNT,
    PHASE11_FEATURE_SEEDS,
    PHASE11_GAMMA,
    PHASE11_LABEL_SHUFFLE_SEEDS,
    PHASE11_MAP_RATIOS,
    PHASE11_ROBUSTNESS_RATIOS,
    Phase11Config,
    _task_grid,
)


def test_phase11_design_is_frozen() -> None:
    assert PHASE11_GAMMA == 0.5
    assert PHASE11_MAP_RATIOS == (
        0.1,
        0.5,
        0.9,
        0.98,
        1.0,
        1.02,
        1.1,
        2.0,
        5.0,
        10.0,
        50.0,
    )
    assert PHASE11_ROBUSTNESS_RATIOS == (0.1, 1.0, 1.02, 5.0, 50.0)
    assert len(PHASE11_FEATURE_SEEDS) == len(PHASE11_LABEL_SHUFFLE_SEEDS) == 3
    assert not set(PHASE11_FEATURE_SEEDS) & set(PHASE11_LABEL_SHUFFLE_SEEDS)


def test_phase11_task_grid_has_21_matched_cases(tmp_path: Path) -> None:
    config = Phase11Config(data_directory=tmp_path)
    tasks = _task_grid(config)

    assert len(tasks) == 3
    assert sum(len(ratios) for _, _, ratios in tasks) == PHASE11_EXPECTED_CASE_COUNT
    assert tasks[0] == (
        PHASE11_FEATURE_SEEDS[0],
        PHASE11_LABEL_SHUFFLE_SEEDS[0],
        PHASE11_MAP_RATIOS,
    )


def test_label_permutation_is_deterministic_derangement_and_preserves_values() -> None:
    sample_ids = np.arange(100, dtype=np.int64) * 3_600 + 1_700_000_000
    target = np.linspace(-1, 1, sample_ids.size)
    first = deterministic_label_permutation(sample_ids, 42)
    second = deterministic_label_permutation(sample_ids, 42)

    np.testing.assert_array_equal(first, second)
    assert not np.any(first == np.arange(sample_ids.size))
    np.testing.assert_array_equal(np.sort(target[first]), np.sort(target))


def test_label_mapping_is_invariant_to_input_row_order() -> None:
    sample_ids = np.arange(50, dtype=np.int64) * 3_600 + 1_700_000_000
    reordered_ids = sample_ids[::-1]
    original = deterministic_label_permutation(sample_ids, 2026)
    reordered = deterministic_label_permutation(reordered_ids, 2026)
    original_mapping = dict(zip(sample_ids, sample_ids[original], strict=True))
    reordered_mapping = dict(zip(reordered_ids, reordered_ids[reordered], strict=True))

    assert original_mapping == reordered_mapping
    assert permutation_fingerprint(sample_ids, original) == permutation_fingerprint(
        reordered_ids, reordered
    )


def test_label_seed_changes_the_permutation() -> None:
    sample_ids = np.arange(100, dtype=np.int64) + 1_700_000_000
    first = deterministic_label_permutation(sample_ids, 1)
    second = deterministic_label_permutation(sample_ids, 2)

    assert not np.array_equal(first, second)
