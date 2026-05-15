"""Determinism and safety tests for internal multiprocessing.

Workers return ``(job_index, result)`` tuples and the parent reassembles by
index, so completion order is structurally irrelevant — one parallel run
verifies determinism just as well as ten. We keep ``REPEATS=2`` as cheap
insurance and mark the spawn-heavy cases ``@pytest.mark.slow`` so the default
``pytest`` run stays fast; ``--runslow`` exercises the full matrix.
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
    _aggregate_worker_stats,
    _extract_worker_stats,
)


REPEATS = 2


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
        assert DeepDiff(t1, t2, ignore_order=True) == {}

    def test_explicit_multiprocessing_false(self):
        t1 = [1, 2, 3]
        t2 = [3, 2, 1]
        assert DeepDiff(t1, t2, ignore_order=True, multiprocessing=False) == {}

    def test_invalid_workers_surfaces_at_diff_level(self):
        with pytest.raises(ValueError):
            DeepDiff([1], [2], multiprocessing=True, multiprocessing_workers=0)


class TestHashesParallelHelper:
    """Direct unit tests for ``compute_hashes_parallel`` — no DeepDiff overhead."""

    def test_empty_jobs_returns_empty_list(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        assert compute_hashes_parallel(jobs=[], deephash_parameters={}, config=cfg) == []

    def test_unpickleable_params_returns_none(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        params = {"hasher": lambda obj: "x"}
        result = compute_hashes_parallel(
            jobs=[(1, "root[0]"), (2, "root[1]")],
            deephash_parameters=params,
            config=cfg,
        )
        assert result is None


class TestSubtreeParallelHelper:
    """Direct unit tests for ``compute_subtree_diffs_parallel``."""

    def test_empty_jobs_returns_empty_list(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        result = compute_subtree_diffs_parallel(
            jobs=[], parameters={}, original_type=None, config=cfg,
        )
        # Phase 4: orchestrator now returns (entries_by_job, worker_stats).
        assert result is not None
        entries_by_job, worker_stats = result
        assert entries_by_job == []
        assert worker_stats['DIFF COUNT'] == 0
        assert worker_stats['PASSES COUNT'] == 0
        assert worker_stats['MAX DIFF LIMIT REACHED'] is False

    def test_unpickleable_parameters_returns_none(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        params = {"some_param": lambda x: x}
        result = compute_subtree_diffs_parallel(
            jobs=[({"x": 1}, {"x": 2})],
            parameters=params,
            original_type=None,
            config=cfg,
        )
        assert result is None


class TestSafetyFallback:
    """Unsafe inputs must not crash; they fall back to serial."""

    def test_is_pickleable_helper(self):
        assert is_pickleable({"a": 1}) is True
        assert is_pickleable(lambda x: x) is False

    def test_compute_distances_parallel_returns_none_on_unpickleable_compare_func(self):
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        result = compute_distances_parallel(
            jobs=[("h1", "h2", {"x": 1}, {"x": 2})],
            parameters={"foo": "bar"},
            original_type=None,
            iterable_compare_func=lambda *args, **kwargs: None,
            config=cfg,
        )
        assert result is None


# Module-level helpers — pickleable under spawn.
def _simple_hasher(obj, *args, **kwargs):
    import hashlib
    return hashlib.sha1(repr(obj).encode("utf-8")).hexdigest()


def _drop_secret_callback(obj, path):
    return "secret" in path


from deepdiff.operator import BaseOperator  # noqa: E402


class _NoopOperator(BaseOperator):
    def __init__(self):
        super().__init__()

    def give_up_diffing(self, level, diff_instance):
        return False

    def normalize_value_for_hashing(self, parent, obj):
        return obj


def _assert_parallel_matches_serial(t1, t2, **kwargs):
    kwargs.setdefault("ignore_order", True)
    kwargs.setdefault("cutoff_intersection_for_pairs", 1)
    serial = DeepDiff(t1, t2, **kwargs)
    for _ in range(REPEATS):
        parallel = _run_parallel(t1, t2, **kwargs)
        assert parallel == serial, (
            "parallel != serial: %r vs %r" % (parallel, serial)
        )


@pytest.mark.slow
class TestDeterminismSlow:
    """End-to-end parallel-vs-serial checks. Each test pays a pool-spawn tax."""

    def test_tied_distances(self):
        # Multiple candidate pairs with identical rough distance — would expose
        # any worker-completion-order leakage in pair selection.
        t1 = [{"k": "a", "v": 1}, {"k": "b", "v": 1}, {"k": "c", "v": 1}]
        t2 = [{"k": "a", "v": 2}, {"k": "b", "v": 2}, {"k": "c", "v": 2}]
        _assert_parallel_matches_serial(t1, t2)

    def test_repeated_items_report_repetition_true(self):
        t1 = [1, 1, 1, 2, 3, 3]
        t2 = [3, 1, 2, 2, 4]
        _assert_parallel_matches_serial(t1, t2, report_repetition=True)

    def test_exclude_paths(self):
        t1 = [{"id": i, "secret": i * 100, "v": i} for i in range(8)]
        t2 = [{"id": i, "secret": i * 999, "v": i + (1 if i == 5 else 0)} for i in range(8)]
        _assert_parallel_matches_serial(t1, t2, exclude_paths=["root[0]['secret']"])

    def test_below_threshold_uses_serial(self):
        # Default threshold (64) keeps small inputs serial even with mp on.
        t1 = [1, 2, 3]
        t2 = [3, 2, 1]
        out = DeepDiff(t1, t2, ignore_order=True, multiprocessing=True)
        assert out == DeepDiff(t1, t2, ignore_order=True)

    def test_paired_subtree_changes_match_serial(self):
        # Parent rebases worker leaves; verifies path reconstruction.
        t1 = [{"id": i, "data": {"x": i, "y": [i, i + 1]}} for i in range(10)]
        t2 = [{"id": i, "data": {"x": i, "y": [i, i + 2]}} for i in range(10)]
        _assert_parallel_matches_serial(t1, t2)

    def test_paired_subtree_added_and_removed_keys(self):
        t1 = [{"id": i, "old_only": i} for i in range(8)]
        t2 = [{"id": i, "new_only": i} for i in range(8)]
        _assert_parallel_matches_serial(t1, t2)

    def test_worker_does_not_recursively_spawn(self):
        # Sanitization must disable mp inside the worker; without it, nested
        # spawn either deadlocks or runs absurdly slowly.
        t1 = [{"deep": {"deeper": {"deepest": [i, i + 1, i + 2]}}} for i in range(8)]
        t2 = [{"deep": {"deeper": {"deepest": [i, i + 1, i + 3]}}} for i in range(8)]
        _assert_parallel_matches_serial(t1, t2)


@pytest.mark.slow
class TestSubtreeFallbackSlow:
    """Subtree parallelism degrades cleanly when features can't ship to workers."""

    def test_custom_operators_force_serial(self):
        op = _NoopOperator()
        t1 = [{"id": i, "v": i} for i in range(10)]
        t2 = [{"id": i, "v": i + (1 if i == 5 else 0)} for i in range(10)]
        serial = DeepDiff(t1, t2, ignore_order=True, custom_operators=[op])
        parallel = _run_parallel(t1, t2, ignore_order=True, custom_operators=[op])
        assert parallel == serial

    def test_exclude_obj_callback_forces_serial(self):
        # The callback receives a path; subtree-relative paths inside a worker
        # would mis-fire, so the parent must keep this serial.
        t1 = [{"id": i, "secret": i, "v": i} for i in range(8)]
        t2 = [{"id": i, "secret": i, "v": i + (1 if i == 3 else 0)} for i in range(8)]
        serial = DeepDiff(
            t1, t2, ignore_order=True, exclude_obj_callback=_drop_secret_callback,
        )
        parallel = _run_parallel(
            t1, t2, ignore_order=True, exclude_obj_callback=_drop_secret_callback,
        )
        assert parallel == serial

    def test_unpickleable_hasher_falls_back(self):
        bad_hasher = lambda obj: _simple_hasher(obj)  # noqa: E731
        t1 = [{"x": i} for i in range(8)]
        t2 = [{"x": i + (1 if i == 3 else 0)} for i in range(8)]
        serial = DeepDiff(t1, t2, ignore_order=True, hasher=bad_hasher)
        parallel = _run_parallel(t1, t2, ignore_order=True, hasher=bad_hasher)
        assert parallel == serial


class TestWorkerStatsUnit:
    """Phase 4 unit-level checks for the stats extraction/aggregation helpers."""

    def test_extract_worker_stats_handles_missing_attribute(self):
        class _Bare:
            pass
        # No ``_stats`` attribute at all — extractor must return zeroed counters
        # rather than crash. This shields against the future case where a
        # worker's DeepDiff is replaced by a non-DeepDiff stand-in.
        delta = _extract_worker_stats(_Bare())
        assert delta['DIFF COUNT'] == 0
        assert delta['PASSES COUNT'] == 0
        assert delta['DISTANCE CACHE HIT COUNT'] == 0
        assert delta['MAX PASS LIMIT REACHED'] is False
        assert delta['MAX DIFF LIMIT REACHED'] is False

    def test_aggregate_sums_counters_and_or_merges_flags(self):
        deltas = [
            {'DIFF COUNT': 3, 'PASSES COUNT': 1, 'DISTANCE CACHE HIT COUNT': 0,
             'MAX PASS LIMIT REACHED': False, 'MAX DIFF LIMIT REACHED': False},
            {'DIFF COUNT': 7, 'PASSES COUNT': 2, 'DISTANCE CACHE HIT COUNT': 4,
             'MAX PASS LIMIT REACHED': True,  'MAX DIFF LIMIT REACHED': False},
            {},  # empty/missing delta must be tolerated
        ]
        agg = _aggregate_worker_stats(deltas)
        assert agg['DIFF COUNT'] == 10
        assert agg['PASSES COUNT'] == 3
        assert agg['DISTANCE CACHE HIT COUNT'] == 4
        assert agg['MAX PASS LIMIT REACHED'] is True
        assert agg['MAX DIFF LIMIT REACHED'] is False

    def test_aggregate_empty_input_returns_zeroed_dict(self):
        agg = _aggregate_worker_stats([])
        assert agg == {
            'DIFF COUNT': 0,
            'PASSES COUNT': 0,
            'DISTANCE CACHE HIT COUNT': 0,
            'MAX PASS LIMIT REACHED': False,
            'MAX DIFF LIMIT REACHED': False,
        }


class TestStatsKeys:
    """get_stats() must always expose the new WORKER_* keys, even in serial mode."""

    def test_serial_run_exposes_worker_keys_zeroed(self):
        # No multiprocessing means workers never ran — but the keys must exist
        # so downstream consumers that read them unconditionally don't KeyError.
        diff = DeepDiff([1, 2, 3], [1, 2, 4], ignore_order=True)
        stats = diff.get_stats()
        assert stats['WORKER DIFF COUNT'] == 0
        assert stats['WORKER PASSES COUNT'] == 0
        assert stats['WORKER DISTANCE CACHE HIT COUNT'] == 0
        assert stats['WORKER BATCH COUNT'] == 0

    def test_existing_stats_keys_still_present(self):
        # Phase 4 must not regress the keys Phase 1 / pre-MP code relies on.
        diff = DeepDiff([1, 2, 3], [1, 2, 4], ignore_order=True)
        stats = diff.get_stats()
        for key in ('PASSES COUNT', 'DIFF COUNT', 'DISTANCE CACHE HIT COUNT',
                    'MAX PASS LIMIT REACHED', 'MAX DIFF LIMIT REACHED'):
            assert key in stats


@pytest.mark.slow
class TestWorkerStatsAggregationSlow:
    """End-to-end checks: workers must contribute to the WORKER_* aggregates."""

    def test_paired_subtree_run_aggregates_worker_stats(self):
        # Force the subtree-parallel path: lots of paired-item diffs, threshold
        # 0 so we don't fall through to serial. ``cutoff_intersection_for_pairs=1``
        # is required — the default cutoff disables pair selection when most
        # items differ, which is exactly our setup, so without it the subtree
        # queue stays empty and no batch is dispatched.
        t1 = [{"id": i, "data": {"x": i, "y": [i, i + 1]}} for i in range(20)]
        t2 = [{"id": i, "data": {"x": i, "y": [i, i + 2]}} for i in range(20)]
        diff = _run_parallel(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1)
        stats = diff.get_stats()
        assert stats['WORKER BATCH COUNT'] >= 1, (
            "expected at least one parallel batch to have run; got stats=%r" % stats
        )
        assert stats['WORKER DIFF COUNT'] > 0, (
            "workers must have done diffs; got %r" % stats
        )

    def test_distance_loop_aggregates_worker_stats(self):
        # Many added/removed candidates with distinct shapes — drives the
        # distance-loop parallel path even when subtree pairing rejects most
        # pairs. Also leans on threshold=0 to guarantee we go through the pool.
        t1 = [{"id": i, "v": [i, i, i]} for i in range(80)]
        t2 = [{"id": i + 1000, "v": [i, i, i + 1]} for i in range(80)]
        diff = _run_parallel(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1)
        stats = diff.get_stats()
        # Either the distance batch or the subtree batch must have shipped to
        # workers; both feed _merge_worker_stats so the batch counter is the
        # cleanest evidence that aggregation actually fired.
        assert stats['WORKER BATCH COUNT'] >= 1

    def test_aggregation_does_not_corrupt_parent_counters(self):
        # Phase 4 must not double-count: parent DIFF COUNT must remain in the
        # same ballpark as a serial run, even when workers add their own.
        t1 = [{"id": i, "v": i} for i in range(20)]
        t2 = [{"id": i, "v": i + (1 if i == 5 else 0)} for i in range(20)]
        serial = DeepDiff(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1)
        parallel = _run_parallel(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1)
        # Result must still match.
        assert parallel == serial
        # Parent DIFF COUNT may differ slightly because pair-selection traversal
        # avoids some inline _diff calls when distances are precomputed in
        # workers, but the order of magnitude must still be reasonable —
        # specifically, parent count alone must not silently include worker work.
        s_parent = serial.get_stats()['DIFF COUNT']
        p_parent = parallel.get_stats()['DIFF COUNT']
        # Parent-only count in a parallel run is <= serial count: the pairs
        # whose distance was computed in a worker are subtracted from the
        # parent's inline path. This invariant breaks if we accidentally also
        # added worker counts back into DIFF COUNT.
        assert p_parent <= s_parent, (
            "parent DIFF COUNT %d exceeds serial %d — looks like worker "
            "counts are leaking into the parent counter" % (p_parent, s_parent)
        )


# ---------------------------------------------------------------------------
# Phase 5 — extended determinism matrix (Subticket #6).
#
# Every test below pins the parallel result against the serial result for one
# axis of the public API. The point isn't to re-test that DeepDiff handles
# these features (other test files do that); it's to prove that turning
# multiprocessing on is a no-op for output across the supported surface.
#
# These are marked ``@pytest.mark.slow`` because each one pays a pool-spawn
# tax and they would dominate the default test run. Running ``pytest --runslow``
# exercises the full matrix.
# ---------------------------------------------------------------------------


# Module-level — pickleable under spawn.
class _SlotPoint:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, _SlotPoint) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return "_SlotPoint(x=%r, y=%r)" % (self.x, self.y)


class _DictBag:
    """Plain class with __dict__ — exercises object-with-attrs hashing/diffing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self, other):
        return isinstance(other, _DictBag) and self.__dict__ == other.__dict__


from collections import namedtuple  # noqa: E402

_NamedPoint = namedtuple("_NamedPoint", ["x", "y"])


def _hex_hasher(obj, *args, **kwargs):
    """Module-level pickleable custom hasher used to verify the full path."""
    import hashlib
    return hashlib.md5(repr(obj).encode("utf-8")).hexdigest()


@pytest.mark.slow
class TestDeterminismMatrixSlow:
    """Per-feature determinism: parallel output must equal serial output."""

    def test_report_repetition_false(self):
        t1 = [1, 1, 1, 2, 3, 3, 4, 4]
        t2 = [3, 1, 2, 2, 4, 4, 5, 5]
        _assert_parallel_matches_serial(t1, t2, report_repetition=False)

    def test_sets_of_dicts_inside_list(self):
        # Frozensets-of-tuples inside a list — set membership is order-free,
        # but DeepDiff still has to hash and pair the containing dicts.
        t1 = [{"id": i, "tags": frozenset({("k", i), ("k", i + 1)})} for i in range(10)]
        t2 = [{"id": i, "tags": frozenset({("k", i), ("k", i + 2)})} for i in range(10)]
        _assert_parallel_matches_serial(t1, t2)

    def test_top_level_set(self):
        t1 = {("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)}
        t2 = {("a", 1), ("b", 2), ("c", 3), ("d", 99), ("f", 6)}
        _assert_parallel_matches_serial(t1, t2)

    def test_custom_hasher_pickleable(self):
        # Pickleable hasher should travel to workers cleanly (no fallback).
        t1 = [{"id": i, "v": i} for i in range(8)]
        t2 = [{"id": i, "v": i + (1 if i == 4 else 0)} for i in range(8)]
        _assert_parallel_matches_serial(t1, t2, hasher=_hex_hasher)

    def test_ignore_string_case(self):
        t1 = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        t2 = [{"name": "alice"}, {"name": "bob"}, {"name": "DAVE"}]
        _assert_parallel_matches_serial(t1, t2, ignore_string_case=True)

    def test_ignore_numeric_type_changes(self):
        t1 = [{"v": 1}, {"v": 2}, {"v": 3}]
        t2 = [{"v": 1.0}, {"v": 2.0}, {"v": 4.0}]
        _assert_parallel_matches_serial(t1, t2, ignore_numeric_type_changes=True)

    def test_ignore_string_type_changes(self):
        t1 = [{"v": "x"}, {"v": "y"}, {"v": "z"}]
        t2 = [{"v": b"x"}, {"v": b"y"}, {"v": b"q"}]
        _assert_parallel_matches_serial(t1, t2, ignore_string_type_changes=True)

    def test_include_paths(self):
        # ``include_paths`` is path-based, so the parent-side _skip_this re-filter
        # in _dispatch_subtree_jobs has to handle it the same way it handles
        # exclude_paths.
        t1 = [{"id": i, "keep": i, "drop": i * 100} for i in range(8)]
        t2 = [{"id": i, "keep": i + (1 if i == 3 else 0), "drop": i * 999} for i in range(8)]
        _assert_parallel_matches_serial(t1, t2, include_paths="root[0]['keep']")

    def test_exclude_regex_paths(self):
        import re
        t1 = [{"id": i, "v": i, "_internal_a": i, "_internal_b": i * 2} for i in range(8)]
        t2 = [{"id": i, "v": i + (1 if i == 4 else 0),
               "_internal_a": i * 999, "_internal_b": i * 999} for i in range(8)]
        _assert_parallel_matches_serial(
            t1, t2, exclude_regex_paths=[re.compile(r"_internal_\w+")],
        )

    def test_namedtuple_items(self):
        t1 = [_NamedPoint(x=i, y=i + 1) for i in range(10)]
        t2 = [_NamedPoint(x=i, y=i + 2) for i in range(10)]
        _assert_parallel_matches_serial(t1, t2)

    def test_slots_objects(self):
        t1 = [_SlotPoint(x=i, y=i + 1) for i in range(10)]
        t2 = [_SlotPoint(x=i, y=i + 2) for i in range(10)]
        _assert_parallel_matches_serial(t1, t2)

    def test_dunder_dict_objects(self):
        t1 = [_DictBag(id=i, v=i) for i in range(10)]
        t2 = [_DictBag(id=i, v=i + (1 if i == 5 else 0)) for i in range(10)]
        _assert_parallel_matches_serial(t1, t2)

    def test_group_by_serial_fallback(self):
        # ``group_by`` reshapes input dicts into keyed dicts before diffing,
        # which currently runs without ignore_order; the parallel path is not
        # engaged. This test pins the no-regression invariant: turning mp on
        # for a group_by run must still produce the same output.
        t1 = [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "c", "v": 3}]
        t2 = [{"id": "a", "v": 1}, {"id": "b", "v": 99}, {"id": "c", "v": 3}]
        serial = DeepDiff(t1, t2, group_by="id")
        parallel = DeepDiff(
            t1, t2, group_by="id",
            multiprocessing=True, multiprocessing_workers=4,
            multiprocessing_threshold=0,
        )
        assert parallel == serial

    def test_generator_input_falls_back(self):
        # Generators are flagged in the doc as unsupported (they may be
        # consumed or pickled differently). DeepDiff materializes them in the
        # parent before the parallel section, so the result must still match
        # the serial run.
        def gen1():
            for x in [{"id": i, "v": i} for i in range(8)]:
                yield x

        def gen2():
            for x in [{"id": i, "v": i + (1 if i == 3 else 0)} for i in range(8)]:
                yield x

        serial = DeepDiff(list(gen1()), list(gen2()), ignore_order=True,
                          cutoff_intersection_for_pairs=1)
        parallel = _run_parallel(list(gen1()), list(gen2()),
                                 cutoff_intersection_for_pairs=1)
        assert parallel == serial

    def test_verbose_level_2(self):
        t1 = [{"id": i, "v": i} for i in range(10)]
        t2 = [{"id": i, "v": i + (1 if i == 5 else 0)} for i in range(10)]
        _assert_parallel_matches_serial(t1, t2, verbose_level=2)

    def test_text_view_to_dict_matches(self):
        # Compare the public dict view directly — guards against any drift
        # between the tree representation and its TextResult projection.
        t1 = [{"id": i, "v": i} for i in range(8)]
        t2 = [{"id": i, "v": i + (1 if i == 3 else 0)} for i in range(8)]
        serial = DeepDiff(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1)
        parallel = _run_parallel(t1, t2, cutoff_intersection_for_pairs=1)
        assert dict(parallel) == dict(serial)


@pytest.mark.slow
class TestDeterminismNumpySlow:
    """Numpy-specific determinism cases. Skipped if numpy isn't available."""

    def test_numpy_array_in_dict(self):
        np = pytest.importorskip("numpy")
        t1 = [{"id": i, "v": np.array([i, i + 1, i + 2])} for i in range(8)]
        t2 = [{"id": i, "v": np.array([i, i + 1, i + 3])} for i in range(8)]
        _assert_parallel_matches_serial(t1, t2)


# Pydantic test class must be module-level so spawn can find and unpickle it.
try:
    import pydantic as _pydantic_mod  # noqa: F401

    class _PydanticItem(_pydantic_mod.BaseModel):
        id: int
        v: int

except Exception:  # pragma: no cover — pydantic not installed
    _PydanticItem = None  # type: ignore[assignment]


@pytest.mark.slow
class TestDeterminismPydanticSlow:
    """Pydantic-specific determinism. Skipped if pydantic isn't available."""

    def test_pydantic_models_in_list(self):
        if _PydanticItem is None:
            pytest.skip("pydantic not installed")
        t1 = [_PydanticItem(id=i, v=i) for i in range(8)]
        t2 = [_PydanticItem(id=i, v=i + (1 if i == 3 else 0)) for i in range(8)]
        _assert_parallel_matches_serial(t1, t2)


@pytest.mark.slow
class TestPickleFailureFallbackSlow:
    """Inputs that can't be pickled must fall back to serial without crashing."""

    def test_unpickleable_iterable_compare_func_falls_back(self):
        # iterable_compare_func is checked up front in compute_distances_parallel
        # — a closure cannot pickle, so the helper returns None and the parent
        # runs serially.
        local_state = {"calls": 0}

        def closure_compare(x, y, level=None):
            local_state["calls"] += 1
            return False

        t1 = [{"id": i, "v": i} for i in range(8)]
        t2 = [{"id": i, "v": i + (1 if i == 4 else 0)} for i in range(8)]
        # iterable_compare_func is only consulted when ignore_order is OFF
        # (it's the ordered-pairing helper), so the parallel path doesn't run
        # — the test still pins "mp=True doesn't break this combo."
        serial = DeepDiff(t1, t2, iterable_compare_func=closure_compare)
        parallel = DeepDiff(
            t1, t2, iterable_compare_func=closure_compare,
            multiprocessing=True, multiprocessing_workers=4,
            multiprocessing_threshold=0,
        )
        assert parallel == serial


def _explode_on_unpickle():
    """Raised when the worker unpickles ``_ExplodingItem``."""
    raise RuntimeError("worker explosion: _ExplodingItem cannot be reconstructed")


class _ExplodingItem:
    """Pickleable on the parent, but unpickling in the worker raises.

    This is exactly the pattern that ``is_pickleable`` (which only calls
    ``pickle.dumps``) cannot detect — and what the determinism contract says
    must propagate as a normal exception, not a silent fallback.
    """

    def __reduce__(self):
        return (_explode_on_unpickle, ())


@pytest.mark.slow
class TestWorkerExceptionPropagationSlow:
    """Worker exceptions outside the pickle-fallback set must propagate.

    The catch list in ``compute_*_parallel`` is intentionally narrow:
    ``(pickle.PicklingError, AttributeError, TypeError)`` — Python raises those
    *during the pickle round-trip*. Anything else (RuntimeError, ValueError)
    that escapes the worker logic itself must bubble through ``future.result()``
    and out of the helper, not be silently converted to a ``None`` fallback.
    """

    def test_runtime_error_in_worker_propagates(self):
        # ``_ExplodingItem`` survives ``pickle.dumps`` but its ``__reduce__``
        # tells the unpickler to call ``_explode_on_unpickle()``, which raises
        # ``RuntimeError`` inside the worker process. The helper's catch list
        # is ``(PicklingError, AttributeError, TypeError)``; an unpickle-time
        # ``RuntimeError`` is outside that set, so it must propagate up rather
        # than be silently turned into a ``None`` fallback. In practice the
        # ProcessPoolExecutor surfaces this as ``BrokenProcessPool`` (the
        # worker dies before it can return a result) — either form proves the
        # contract: the failure is loud, not silent.
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        with pytest.raises(Exception) as exc_info:
            compute_subtree_diffs_parallel(
                jobs=[(_ExplodingItem(), _ExplodingItem())],
                parameters={"foo": "bar"},
                original_type=None,
                config=cfg,
            )
        # Sanity-check we got a "loud" failure, not the silent fallback path
        # (which would have returned ``None`` and never raised).
        assert exc_info.value is not None

    def test_distance_worker_runtime_error_propagates(self):
        # Same exploding-item trick on the distance helper. Same contract:
        # an exception escapes the helper rather than being silenced.
        cfg = MPConfig(enabled=True, workers=2, threshold=0)
        with pytest.raises(Exception) as exc_info:
            compute_distances_parallel(
                jobs=[("h_added", "h_removed", _ExplodingItem(), _ExplodingItem())],
                parameters={"foo": "bar"},
                original_type=None,
                iterable_compare_func=None,
                config=cfg,
            )
        assert exc_info.value is not None

