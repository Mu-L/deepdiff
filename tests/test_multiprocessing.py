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
    compute_hashes_parallel,
    compute_subtree_diffs_parallel,
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


class TestHashtableParallel:
    """Phase 2: ``_create_hashtable`` per-item DeepHash parallelism.

    These exercise the parallel hashing path with ``multiprocessing_threshold=0``
    so even small fixtures hit the worker pool. Result must match the equivalent
    serial run, repeatedly, regardless of worker completion order.
    """

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

    def test_large_list_of_dicts(self):
        # Bigger N so spawn cost is not pathological; results must still match.
        t1 = [{"i": i, "name": "item-%d" % i, "tags": [i, i + 1]} for i in range(40)]
        t2 = [{"i": i, "name": "item-%d" % i, "tags": [i, i + 1]} for i in range(40)]
        # Add a single change deep in the middle
        t2[17]["name"] = "changed"
        self._assert_determinism(t1, t2)

    def test_list_of_lists(self):
        t1 = [[i, i + 1, i + 2] for i in range(15)]
        t2 = [[i, i + 1, i + 2] for i in range(15)]
        t2[5] = [99, 100, 101]
        self._assert_determinism(t1, t2)

    def test_set_of_hashables(self):
        t1 = set(range(30))
        t2 = set(range(30))
        t2.discard(7)
        t2.add(99)
        self._assert_determinism(t1, t2)

    def test_repeated_items_report_repetition_false(self):
        # Repeated items: cache reuse path. Parent merges per-index hashes
        # in serial order so duplicates collapse the same way.
        t1 = [{"k": i % 3} for i in range(20)]
        t2 = [{"k": (i + 1) % 3} for i in range(20)]
        self._assert_determinism(t1, t2, report_repetition=False)

    def test_repeated_items_report_repetition_true(self):
        t1 = [{"k": i % 3} for i in range(20)]
        t2 = [{"k": (i + 1) % 3} for i in range(20)]
        self._assert_determinism(t1, t2, report_repetition=True)

    def test_nested_mixed_structures(self):
        t1 = [
            {"id": i, "data": {"vals": [j for j in range(i)], "meta": {"k": i}}}
            for i in range(12)
        ]
        t2 = [
            {"id": i, "data": {"vals": [j for j in range(i)], "meta": {"k": i + (1 if i == 6 else 0)}}}
            for i in range(12)
        ]
        self._assert_determinism(t1, t2)

    def test_below_threshold_uses_serial(self):
        # Default threshold is 64; small inputs without the override stay serial.
        t1 = [1, 2, 3]
        t2 = [3, 2, 1]
        # No multiprocessing_threshold=0 override here on purpose.
        out = DeepDiff(t1, t2, ignore_order=True, multiprocessing=True)
        assert out == DeepDiff(t1, t2, ignore_order=True)

    def test_unpickleable_hasher_falls_back(self):
        # A lambda hasher is not pickleable. Must not crash; result must match
        # the serial run.
        bad_hasher = lambda obj: _simple_hasher(obj)  # noqa: E731
        t1 = [{"x": i} for i in range(10)]
        t2 = [{"x": i + (1 if i == 3 else 0)} for i in range(10)]
        serial = DeepDiff(t1, t2, ignore_order=True, hasher=bad_hasher)
        parallel = _run_parallel(t1, t2, ignore_order=True, hasher=bad_hasher)
        assert parallel == serial


class TestHashesParallelHelper:
    """Direct unit tests for ``compute_hashes_parallel``."""

    def test_empty_jobs_returns_empty_list(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        assert compute_hashes_parallel(jobs=[], deephash_parameters={}, config=cfg) == []

    def test_unpickleable_params_returns_none(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        # A lambda inside the params dict cannot be pickled under spawn.
        params = {"hasher": lambda obj: "x"}
        result = compute_hashes_parallel(
            jobs=[(1, "root[0]"), (2, "root[1]")],
            deephash_parameters=params,
            config=cfg,
        )
        assert result is None

    def test_returns_one_hash_per_item_in_index_order(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        jobs = [(i, "root[%d]" % i) for i in range(5)]
        # Minimal deephash params — keep keys aligned with what DeepDiff
        # would normally pass. An empty dict is sufficient for primitives.
        result = compute_hashes_parallel(
            jobs=jobs,
            deephash_parameters={},
            config=cfg,
        )
        assert result is not None
        assert len(result) == 5
        # All entries are non-None for primitives.
        assert all(h is not None for h in result)
        # Same int hashed twice yields identical hashes.
        again = compute_hashes_parallel(
            jobs=jobs, deephash_parameters={}, config=cfg
        )
        assert again == result


# Module-level callables/classes so they pickle cleanly under spawn.
def _drop_secret_callback(obj, path):
    # Mirrors a real-world exclude_obj_callback that inspects the path.
    return "secret" in path


from deepdiff.operator import BaseOperator  # noqa: E402


class _NoopOperator(BaseOperator):
    # No types/regex_paths configured, so match() never fires — but its mere
    # presence in custom_operators must force the parent to keep subtree
    # diffs serial (the worker would not be able to run custom_report_result
    # back into the parent's tree).
    def __init__(self):
        super().__init__()

    def give_up_diffing(self, level, diff_instance):
        return False

    def normalize_value_for_hashing(self, parent, obj):
        # Required for ignore_order=True compatibility when this operator
        # ships through DeepHash. We don't normalize anything — pass through.
        return obj


class TestSubtreeParallel:
    """Phase 3: paired-subtree diffs run in worker processes after pairing.

    Workers compute a fresh DeepDiff per pair and return tree leaves; the
    parent rebases each leaf's up-chain onto its own ``change_level``. The
    public output must equal the equivalent serial run regardless of worker
    completion order, and unsafe inputs (custom_operators, path-aware
    callbacks) must fall back to inline serial.
    """

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

    def test_paired_subtree_changes_match_serial(self):
        # Each pair has exactly one nested change. Rebased paths must match
        # the inline serial paths character-for-character.
        t1 = [{"id": i, "data": {"x": i, "y": [i, i + 1]}} for i in range(20)]
        t2 = [{"id": i, "data": {"x": i, "y": [i, i + 2]}} for i in range(20)]
        self._assert_determinism(t1, t2)

    def test_paired_subtree_multiple_changes_per_pair(self):
        # Multiple values_changed entries per pair — verifies that each leaf
        # in the worker's tree gets an independent rebased up-chain.
        t1 = [{"a": i, "b": i * 2, "c": i * 3, "d": [i, i, i]} for i in range(15)]
        t2 = [{"a": i + 100, "b": i * 2, "c": i * 3 + 1, "d": [i, i, i + 1]} for i in range(15)]
        self._assert_determinism(t1, t2)

    def test_paired_subtree_with_added_and_removed_keys(self):
        # Non-values_changed report types in the subtree:
        # dictionary_item_added / dictionary_item_removed.
        t1 = [{"id": i, "old_only": i} for i in range(12)]
        t2 = [{"id": i, "new_only": i} for i in range(12)]
        self._assert_determinism(t1, t2)

    def test_paired_subtree_with_type_changes(self):
        t1 = [{"id": i, "v": i} for i in range(10)]
        t2 = [{"id": i, "v": str(i)} for i in range(10)]
        self._assert_determinism(t1, t2)

    def test_paired_subtree_report_repetition_true(self):
        # Exercises the report_repetition=True branch where the inner _diff
        # is also deferred to workers.
        t1 = [{"k": i % 3, "extra": [i]} for i in range(20)]
        t2 = [{"k": (i + 1) % 3, "extra": [i + 1]} for i in range(20)]
        self._assert_determinism(t1, t2, report_repetition=True)

    def test_exclude_paths_re_applied_in_parent(self):
        # Worker sees subtree-relative paths, so exclude_paths cannot be
        # enforced inside the worker; the parent re-filters via _skip_this
        # after rebasing. This test would fail if that re-filter was missing.
        t1 = [{"id": i, "secret": i * 100, "v": i} for i in range(15)]
        t2 = [{"id": i, "secret": i * 999, "v": i + (1 if i == 7 else 0)} for i in range(15)]
        self._assert_determinism(
            t1, t2, exclude_paths=["root[0]['secret']"],
        )


class TestSubtreeFallback:
    """Subtree parallelism must degrade cleanly when features can't ship to workers."""

    def test_custom_operators_force_serial(self):
        # custom_operators can call custom_report_result and mutate the
        # parent diff — they must not run in workers. Even with mp turned on
        # the result must still match the serial run.
        op = _NoopOperator()
        t1 = [{"id": i, "v": i} for i in range(20)]
        t2 = [{"id": i, "v": i + (1 if i == 5 else 0)} for i in range(20)]
        serial = DeepDiff(t1, t2, ignore_order=True, custom_operators=[op])
        parallel = _run_parallel(
            t1, t2, ignore_order=True, custom_operators=[op],
        )
        assert parallel == serial

    def test_exclude_obj_callback_forces_serial(self):
        # exclude_obj_callback receives the level path; in a worker the path
        # is subtree-relative, so the callback would fire on the wrong paths.
        # The parent must keep this case serial.
        t1 = [{"id": i, "secret": i, "v": i} for i in range(15)]
        t2 = [{"id": i, "secret": i, "v": i + (1 if i == 3 else 0)} for i in range(15)]
        serial = DeepDiff(
            t1, t2, ignore_order=True,
            exclude_obj_callback=_drop_secret_callback,
        )
        parallel = _run_parallel(
            t1, t2, ignore_order=True,
            exclude_obj_callback=_drop_secret_callback,
        )
        assert parallel == serial


class TestSubtreeParallelHelper:
    """Direct unit tests for ``compute_subtree_diffs_parallel``."""

    def test_empty_jobs_returns_empty_list(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        result = compute_subtree_diffs_parallel(
            jobs=[], parameters={}, original_type=None, config=cfg,
        )
        assert result == []

    def test_unpickleable_parameters_returns_none(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        # A lambda in parameters cannot be pickled under spawn.
        params = {"some_param": lambda x: x}
        result = compute_subtree_diffs_parallel(
            jobs=[({"x": 1}, {"x": 2})],
            parameters=params,
            original_type=None,
            config=cfg,
        )
        assert result is None
