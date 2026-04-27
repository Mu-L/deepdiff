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
