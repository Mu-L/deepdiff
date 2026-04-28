"""Benchmarks for the internal multiprocessing mode (Subticket #7).

Goal: provide a reproducible "is multiprocessing actually faster?" check for
the workloads multi_processing.md flags as the primary targets — the
``ignore_order=True`` distance loop, paired-subtree diffs, and large lists of
nested dicts. Each workload runs serial first, then parallel at a few worker
counts; we print a single results table.

Usage::

    source ~/.venvs/deep/bin/activate
    python -m benchmarks.multiprocessing_bench

    # Smaller, faster sweep:
    python -m benchmarks.multiprocessing_bench --quick

    # Just one workload:
    python -m benchmarks.multiprocessing_bench --only paired_subtree

The script also asserts that the parallel result equals the serial result for
every workload — a benchmark that produces wrong answers is worse than no
benchmark at all. If any pair diverges the script exits non-zero.

The numbers here are not committed; they're meant to inform threshold tuning
(see DEFAULT_THRESHOLD in deepdiff/_multiprocessing.py) and to expose
regressions when the hot path changes. Re-run on your hardware before drawing
conclusions — process spawn overhead and IPC pickle cost vary wildly across
machines.
"""

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

# Make the package importable when the script is run from a checkout.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deepdiff import DeepDiff  # noqa: E402


# ---------------------------------------------------------------------------
# Workloads.
#
# Each builder returns ``(t1, t2, kwargs)`` where ``kwargs`` is the DeepDiff
# constructor arguments common to both the serial and parallel runs.
# Multiprocessing parameters are added by the runner; workloads should not set
# them.
# ---------------------------------------------------------------------------


def workload_paired_subtree(scale: int) -> Tuple[Any, Any, Dict[str, Any]]:
    """Heavy paired-subtree diff path.

    Each item is a small dict whose nested ``data`` differs by one element;
    pairing kicks in for every item, so the subtree-parallel path runs.
    """
    n = scale
    t1 = [{"id": i, "data": {"x": i, "y": [i, i + 1, i + 2]}} for i in range(n)]
    t2 = [{"id": i, "data": {"x": i, "y": [i, i + 1, i + 3]}} for i in range(n)]
    return t1, t2, {"ignore_order": True, "cutoff_intersection_for_pairs": 1}


def workload_distance_loop(scale: int) -> Tuple[Any, Any, Dict[str, Any]]:
    """Heavy added-vs-removed distance grid.

    All ids are disjoint between t1 and t2, so every t2 item is "added" and
    every t1 item is "removed". The candidate distance grid is N*N, which is
    where the distance worker pool earns its keep.
    """
    n = scale
    t1 = [{"id": i, "v": [i, i, i]} for i in range(n)]
    t2 = [{"id": i + 10_000, "v": [i, i, i + 1]} for i in range(n)]
    return t1, t2, {"ignore_order": True, "cutoff_intersection_for_pairs": 1}


def workload_large_nested_dicts(scale: int) -> Tuple[Any, Any, Dict[str, Any]]:
    """Large list of moderately-deep dicts with one mutation each.

    The shape mirrors the JSON-like blobs the doc calls out: each item is
    several layers deep with a mix of strings, ints, and nested lists.
    """
    n = scale

    def make(i: int, mutate: int) -> Dict[str, Any]:
        return {
            "id": i,
            "name": "name-%d" % i,
            "tags": ["t%d" % (i + j) for j in range(5)],
            "details": {
                "score": i + mutate,
                "history": [{"step": j, "value": j * 2 + mutate} for j in range(4)],
                "meta": {"created_at": "2024-01-%02d" % ((i % 28) + 1),
                         "owner": "user-%d" % (i % 17)},
            },
        }

    t1 = [make(i, 0) for i in range(n)]
    t2 = [make(i, 1 if i % 7 == 0 else 0) for i in range(n)]
    return t1, t2, {"ignore_order": True, "cutoff_intersection_for_pairs": 1}


WORKLOADS: Dict[str, Callable[[int], Tuple[Any, Any, Dict[str, Any]]]] = {
    "paired_subtree": workload_paired_subtree,
    "distance_loop": workload_distance_loop,
    "large_nested_dicts": workload_large_nested_dicts,
}


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


def _time(fn: Callable[[], Any]) -> Tuple[float, Any]:
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def run_one(name: str, scale: int, worker_counts: List[int]) -> List[Dict[str, Any]]:
    """Run one workload serial + parallel and return one row per worker count.

    The serial result is computed once and reused as the correctness reference
    for every parallel run.
    """
    t1, t2, kwargs = WORKLOADS[name](scale)
    print(f"\n=== {name} (scale={scale}) ===")
    print(f"input shape: t1 has {len(t1)} items, t2 has {len(t2)} items")

    serial_time, serial_result = _time(lambda: DeepDiff(t1, t2, **kwargs))
    print(f"serial: {serial_time:.3f}s")

    rows: List[Dict[str, Any]] = [{
        "workload": name, "scale": scale,
        "mode": "serial", "workers": 1,
        "time_s": serial_time, "speedup": 1.0,
        "ok": True,
    }]

    for workers in worker_counts:
        parallel_time, parallel_result = _time(lambda: DeepDiff(
            t1, t2,
            multiprocessing=True,
            multiprocessing_workers=workers,
            multiprocessing_threshold=0,
            **kwargs,
        ))
        ok = parallel_result == serial_result
        speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
        marker = "" if ok else "  !! RESULT MISMATCH !!"
        print(f"parallel(workers={workers}): {parallel_time:.3f}s "
              f"speedup={speedup:.2f}x{marker}")
        rows.append({
            "workload": name, "scale": scale,
            "mode": "parallel", "workers": workers,
            "time_s": parallel_time, "speedup": speedup,
            "ok": ok,
        })
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Compact summary table at the end of the run."""
    print("\n=== summary ===")
    header = ("workload", "scale", "mode", "workers", "time_s", "speedup", "ok")
    print("%-22s %6s %-9s %7s %10s %9s %4s" % header)
    print("-" * 72)
    for r in rows:
        print("%-22s %6d %-9s %7d %10.3f %9.2f %4s" % (
            r["workload"], r["scale"], r["mode"],
            r["workers"], r["time_s"], r["speedup"],
            "yes" if r["ok"] else "NO",
        ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only", choices=list(WORKLOADS), action="append", default=None,
        help="run only the named workload(s); may be repeated. Default: all.",
    )
    parser.add_argument(
        "--workers", type=int, action="append", default=None,
        help="explicit worker count to test; may be repeated. "
             "Default: 2 and min(4, cpu_count).",
    )
    parser.add_argument(
        "--scale", type=int, default=None,
        help="override per-workload scale (number of items). Larger = more "
             "wall time. Default: a per-workload value below.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="use small scales for a fast sanity-check run.",
    )
    args = parser.parse_args()

    workloads = args.only or list(WORKLOADS)
    cpu = os.cpu_count() or 1
    workers_list = args.workers or [2, min(4, cpu)]
    # Deduplicate while preserving order — repeated --workers flags shouldn't
    # cause duplicate rows.
    workers_list = list(dict.fromkeys(workers_list))

    # Default scales tuned so each row takes a few seconds serially. Override
    # via --scale or --quick. These are starting points, not gospel.
    default_scales = {
        "paired_subtree": 200,
        "distance_loop": 120,
        "large_nested_dicts": 200,
    }
    quick_scales = {
        "paired_subtree": 60,
        "distance_loop": 40,
        "large_nested_dicts": 60,
    }
    scales = quick_scales if args.quick else default_scales
    if args.scale is not None:
        scales = {name: args.scale for name in workloads}

    print("DeepDiff multiprocessing benchmark")
    print(f"cpu_count={cpu}  workers tested={workers_list}")

    all_rows: List[Dict[str, Any]] = []
    for name in workloads:
        all_rows.extend(run_one(name, scales[name], workers_list))

    print_table(all_rows)

    # Non-zero exit if any parallel run produced a different result than its
    # serial reference — that's the one regression mode this script must catch.
    if any(not r["ok"] for r in all_rows):
        print("\nFAIL: at least one parallel run did not match its serial reference.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
