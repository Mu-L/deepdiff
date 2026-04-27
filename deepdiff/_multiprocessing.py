"""
Internal multiprocessing helpers for DeepDiff.

Phase 1 scope: parallelize the (added_hash x removed_hash) rough-distance loop
in ``DeepDiff._get_most_in_common_pairs_in_iterables`` for ``ignore_order=True``.

Determinism contract (see docs/multi_processing.md):
- Pair selection happens in the parent only.
- Workers compute distances. The parent submits jobs in a stable index order
  matching the serial nested loop and merges results by that index.
- Worker completion order (``as_completed``) never affects the public output.

Only module-level callables live here so the module is safe under the
``spawn`` start method (macOS/Windows).
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast


DEFAULT_MAX_WORKERS = 4
DEFAULT_THRESHOLD = 64


@dataclass(frozen=True)
class MPConfig:
    """Normalized internal multiprocessing configuration."""
    enabled: bool
    workers: int
    threshold: int

    def should_parallelize(self, n_jobs: int) -> bool:
        return self.enabled and self.workers > 1 and n_jobs >= self.threshold


def normalize_mp_config(
    multiprocessing: Any,
    multiprocessing_workers: Optional[int],
    multiprocessing_threshold: Optional[int],
) -> MPConfig:
    """Validate and normalize the public multiprocessing parameters.

    ``multiprocessing`` accepts True/False. ``multiprocessing_workers`` accepts
    None or a positive int. ``multiprocessing_threshold`` accepts None or a
    non-negative int.
    """
    if multiprocessing not in (True, False, 0, 1):
        raise ValueError(
            "multiprocessing must be True or False; got %r" % (multiprocessing,)
        )
    enabled = bool(multiprocessing)

    if multiprocessing_workers is None:
        cpu = os.cpu_count() or 1
        workers = min(DEFAULT_MAX_WORKERS, cpu)
    else:
        if not isinstance(multiprocessing_workers, int) or multiprocessing_workers < 1:
            raise ValueError(
                "multiprocessing_workers must be None or a positive integer; got %r"
                % (multiprocessing_workers,)
            )
        workers = multiprocessing_workers

    if multiprocessing_threshold is None:
        threshold = DEFAULT_THRESHOLD
    else:
        if not isinstance(multiprocessing_threshold, int) or multiprocessing_threshold < 0:
            raise ValueError(
                "multiprocessing_threshold must be None or a non-negative integer; got %r"
                % (multiprocessing_threshold,)
            )
        threshold = multiprocessing_threshold

    return MPConfig(enabled=enabled, workers=workers, threshold=threshold)


def is_pickleable(obj: Any) -> bool:
    """Return True if ``obj`` round-trips through ``pickle.dumps`` cleanly.

    Used to decide whether parallel execution is safe for a given input.
    A False result triggers serial fallback for that section.
    """
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _sanitize_parameters_for_worker(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Strip parent-process-only state from a ``_parameters`` snapshot.

    The parent's ``_parameters`` may carry references that should not be reused
    inside a worker (mutable shared caches) or that would cause nested
    multiprocessing inside the worker. This produces a copy safe to ship.
    """
    sanitized = dict(parameters)
    # Force serial inside the worker: a nested ProcessPoolExecutor would
    # deadlock or just waste process spawn time. Both the public flag and
    # the normalized config object must be neutralized — recursive DeepDiff
    # calls read ``_mp_config`` directly when ``_parameters`` is supplied.
    sanitized['multiprocessing'] = False
    sanitized['_mp_config'] = MPConfig(enabled=False, workers=1, threshold=0)
    sanitized.pop('_distance_cache', None)
    sanitized.pop('hashes', None)
    sanitized.pop('_numpy_paths', None)
    sanitized.pop('_stats', None)
    sanitized.pop('group_by_keys', None)
    sanitized.pop('tree', None)
    sanitized.pop('_iterable_opcodes', None)
    sanitized.pop('is_root', None)
    return sanitized


def _distance_worker(job: Tuple[int, Dict[str, Any], Any, Any, Any, Any]) -> Tuple[int, float]:
    """Compute the rough distance between two items in a worker process.

    ``job`` layout matches what ``compute_distances_parallel`` ships:
    ``(job_index, sanitized_parameters, removed_item, added_item,
        original_type, iterable_compare_func)``.

    The worker constructs a fresh root ``DeepDiff`` (no shared parent state),
    requests the DELTA_VIEW so we hit the same code path as the serial call in
    ``_get_rough_distance_of_hashed_objs``, and returns the resulting float.
    """
    # Imported here to keep module import cheap and to dodge any circular
    # import surprises under spawn.
    from deepdiff.diff import DeepDiff
    from deepdiff.helper import DELTA_VIEW

    job_index, parameters, removed_item, added_item, original_type, iterable_compare_func = job
    diff = DeepDiff(
        removed_item,
        added_item,
        _parameters=parameters,
        view=DELTA_VIEW,
        _original_type=original_type,
        iterable_compare_func=iterable_compare_func,
        # The worker is spawned without _shared_parameters, so DeepDiff treats
        # it as a root run and would purge ``_distance_cache``/``hashes`` at
        # the end of __init__. We need them alive for the _get_rough_distance
        # call below, hence cache_purge_level=0.
        cache_purge_level=0,
    )
    return job_index, cast(float, diff._get_rough_distance())


def compute_distances_parallel(
    jobs: List[Tuple[Any, Any, Any, Any]],
    parameters: Dict[str, Any],
    original_type: Any,
    iterable_compare_func: Optional[Callable],
    config: MPConfig,
) -> Optional[Dict[Tuple[Any, Any], float]]:
    """Run ``_distance_worker`` over ``jobs`` and return distances by pair.

    ``jobs`` is a list of ``(added_hash, removed_hash, added_item, removed_item)``
    tuples in the exact order the serial nested loop visits them. The parent
    is responsible for that ordering; this helper does not reorder anything.

    Returns:
        A dict ``{(added_hash, removed_hash): distance}``, or ``None`` if the
        section is unsafe to parallelize (unpickleable inputs/parameters,
        worker import error, etc.). On ``None`` the caller MUST fall back to
        the serial path so correctness is preserved.

    Workers may finish out of order; we collect results into a dict keyed by
    the original job index, so callers see the same result regardless of
    completion order.
    """
    if not jobs:
        return {}

    sanitized_params = _sanitize_parameters_for_worker(parameters)

    # Picklability check. Failing fast here means a clear serial fallback
    # rather than an opaque worker crash.
    if not is_pickleable(sanitized_params):
        return None
    if iterable_compare_func is not None and not is_pickleable(iterable_compare_func):
        return None
    # Sample-pickle items: full check of every job is expensive, but pickling
    # the first job catches the common "lambda in custom_operators" failure
    # while keeping overhead bounded.
    if not is_pickleable(jobs[0]):
        return None

    # Imported lazily so importing this module does not pay the cost when
    # multiprocessing is disabled.
    from concurrent.futures import ProcessPoolExecutor, as_completed

    payloads = []
    for i, job in enumerate(jobs):
        added_item = job[2]
        removed_item = job[3]
        payloads.append(
            (i, sanitized_params, removed_item, added_item, original_type, iterable_compare_func)
        )

    results_by_index: Dict[int, float] = {}
    try:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(_distance_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                # Re-raise worker exceptions in the parent so they surface as
                # normal DeepDiff exceptions instead of being swallowed.
                idx, distance = future.result()
                results_by_index[idx] = distance
    except (pickle.PicklingError, AttributeError, TypeError):
        # Pickling/spawn-related failures: surface as a serial fallback rather
        # than crashing the diff. Other exceptions (worker logic bugs, user
        # callback errors) propagate.
        return None

    out: Dict[Tuple[Any, Any], float] = {}
    for i, job in enumerate(jobs):
        out[(job[0], job[1])] = results_by_index[i]
    return out
