"""Determinism and safety tests for internal multiprocessing.

Phase 1 covers the parallel rough-distance loop in
``DeepDiff._get_most_in_common_pairs_in_iterables`` (the ``ignore_order=True``
path). Each parallel run is compared against the equivalent serial run; on
ties or many candidate pairs the merge order must come from the parent's
serial nested loop, not from worker completion order.

We use ``multiprocessing_threshold=0`` to force the parallel path even on
small inputs, then loop the run multiple times to flush out any
non-determinism.
"""

import pytest

from deepdiff import DeepDiff
from deepdiff._multiprocessing import (
    MPConfig,
    normalize_mp_config,
    is_pickleable,
    compute_distances_parallel,
)


REPEATS = 10  # tradeoff between flake-detection and CI time


def _run_parallel(t1, t2, **kwargs):
    return DeepDiff(
        t1, t2,
        multiprocessing=True,
        multiprocessing_workers=4,
        multiprocessing_threshold=0,
        **kwargs,
    )


class TestMPConfig:

    def test_disabled_by_default(self):
        cfg = normalize_mp_config(False, None, None)
        assert cfg.enabled is False
        assert cfg.should_parallelize(10_000) is False

    def test_enabled_default_workers(self):
        cfg = normalize_mp_config(True, None, None)
        assert cfg.enabled is True
        assert cfg.workers >= 1

    def test_explicit_workers(self):
        cfg = normalize_mp_config(True, 3, None)
        assert cfg.workers == 3

    def test_threshold_gates_parallelism(self):
        cfg = normalize_mp_config(True, 4, 100)
        assert cfg.should_parallelize(50) is False
        assert cfg.should_parallelize(100) is True

    def test_invalid_workers(self):
        with pytest.raises(ValueError):
            normalize_mp_config(True, 0, None)
        with pytest.raises(ValueError):
            normalize_mp_config(True, -1, None)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            normalize_mp_config(True, None, -1)

    def test_invalid_multiprocessing_value(self):
        with pytest.raises(ValueError):
            normalize_mp_config("yes", None, None)  # type: ignore[arg-type]

    def test_single_worker_does_not_parallelize(self):
        cfg = MPConfig(enabled=True, workers=1, threshold=0)
        assert cfg.should_parallelize(10_000) is False


class TestParamWiring:

    def test_default_serial_path_unchanged(self):
        t1 = [{"a": 1}, {"a": 2}]
        t2 = [{"a": 2}, {"a": 1}]
        # No multiprocessing parameter at all — must hit the existing path.
        assert DeepDiff(t1, t2, ignore_order=True) == {}

    def test_explicit_multiprocessing_false(self):
        t1 = [1, 2, 3]
        t2 = [3, 2, 1]
        assert DeepDiff(t1, t2, ignore_order=True, multiprocessing=False) == {}

    def test_invalid_workers_surfaces_at_diff_level(self):
        with pytest.raises(ValueError):
            DeepDiff([1], [2], multiprocessing=True, multiprocessing_workers=0)


class TestDeterminism:
    """Each test compares serial vs. parallel many times. Any drift is a bug."""

    def _assert_determinism(self, t1, t2, **kwargs):
        kwargs.setdefault("ignore_order", True)
        kwargs.setdefault("cutoff_intersection_for_pairs", 1)
        serial = DeepDiff(t1, t2, **kwargs)
        for _ in range(REPEATS):
            parallel = _run_parallel(t1, t2, **kwargs)
            assert parallel == serial, (
                "parallel != serial after run; difference: %r vs %r"
                % (parallel, serial)
            )

    def test_nested_lists_of_dicts(self):
        t1 = [{"id": i, "data": {"x": i * 2, "y": [i, i + 1]}} for i in range(20)]
        t2 = [{"id": i, "data": {"x": i * 2 + (1 if i % 5 == 0 else 0), "y": [i, i + 1]}}
              for i in range(20)]
        self._assert_determinism(t1, t2)

    def test_repeated_items_report_repetition_false(self):
        t1 = [1, 1, 1, 2, 3, 3]
        t2 = [3, 1, 2, 2, 4]
        self._assert_determinism(t1, t2, report_repetition=False)

    def test_repeated_items_report_repetition_true(self):
        t1 = [1, 1, 1, 2, 3, 3]
        t2 = [3, 1, 2, 2, 4]
        self._assert_determinism(t1, t2, report_repetition=True)

    def test_tied_distances(self):
        # Multiple candidate pairs with the same rough distance. Worker-order
        # merge would surface here as flapping pairings between runs.
        t1 = [{"k": "a", "v": 1}, {"k": "b", "v": 1}, {"k": "c", "v": 1}]
        t2 = [{"k": "a", "v": 2}, {"k": "b", "v": 2}, {"k": "c", "v": 2}]
        self._assert_determinism(t1, t2)

    def test_sets(self):
        t1 = {frozenset({1, 2}), frozenset({3, 4}), frozenset({5, 6})}
        t2 = {frozenset({1, 2}), frozenset({3, 5}), frozenset({7, 8})}
        self._assert_determinism(t1, t2)

    def test_exclude_paths(self):
        t1 = [{"id": i, "secret": i * 100, "v": i} for i in range(10)]
        t2 = [{"id": i, "secret": i * 999, "v": i + (1 if i == 5 else 0)} for i in range(10)]
        self._assert_determinism(t1, t2, exclude_paths=["root[0]['secret']"])

    def test_ignore_string_case(self):
        t1 = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        t2 = [{"name": "alice"}, {"name": "bob"}, {"name": "DAVE"}]
        self._assert_determinism(t1, t2, ignore_string_case=True)

    def test_custom_pickleable_hasher(self):
        # Module-level callable below is pickleable; lambdas are not.
        self._assert_determinism(
            [{"x": 1}, {"x": 2}, {"x": 3}],
            [{"x": 1}, {"x": 4}, {"x": 5}],
            hasher=_simple_hasher,
        )


class TestSafetyFallback:
    """Unsafe inputs must not crash; they fall back to serial."""

    def test_unpickleable_iterable_compare_func_falls_back(self):
        # A lambda is not pickleable. The parallel section must give up and
        # the result must still match a serial run.
        t1 = [{"k": 1, "v": "a"}, {"k": 2, "v": "b"}]
        t2 = [{"k": 1, "v": "a"}, {"k": 2, "v": "c"}]
        cmp = lambda x, y: x["k"] == y["k"]  # noqa: E731
        serial = DeepDiff(t1, t2, ignore_order=True, iterable_compare_func=cmp)
        parallel = _run_parallel(t1, t2, ignore_order=True, iterable_compare_func=cmp)
        assert parallel == serial

    def test_is_pickleable_helper(self):
        assert is_pickleable({"a": 1}) is True
        assert is_pickleable(lambda x: x) is False

    def test_compute_distances_parallel_returns_none_on_unpickleable_compare_func(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        # Empty params dict pickles fine; the lambda compare func does not.
        result = compute_distances_parallel(
            jobs=[("h1", "h2", {"x": 1}, {"x": 2})],
            parameters={"foo": "bar"},
            original_type=None,
            iterable_compare_func=lambda *args, **kwargs: None,
            config=cfg,
        )
        assert result is None


class TestRecursiveNoNesting:
    """The worker must disable its own multiprocessing so we don't fork-bomb."""

    def test_worker_subdiff_runs_serial(self):
        # The worker invokes DeepDiff(item1, item2, _parameters=sanitized).
        # Sanitization sets _mp_config to disabled; if it didn't, this nested
        # workload would either deadlock or be very slow under spawn. The
        # bound on REPEATS plus pytest's default timeout keeps that visible.
        t1 = [{"deep": {"deeper": {"deepest": [i, i + 1, i + 2]}}} for i in range(8)]
        t2 = [{"deep": {"deeper": {"deepest": [i, i + 1, i + 3]}}} for i in range(8)]
        serial = DeepDiff(t1, t2, ignore_order=True)
        parallel = _run_parallel(t1, t2, ignore_order=True)
        assert parallel == serial


# Module-level helper so it pickles cleanly under the spawn start method.
def _simple_hasher(obj, *args, **kwargs):
    import hashlib
    return hashlib.sha1(repr(obj).encode("utf-8")).hexdigest()
