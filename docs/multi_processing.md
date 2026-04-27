# Ticket: Add Deterministic Internal Multiprocessing for DeepDiff and DeepHash

## Implementation Status

**Phase 1 — landed (2026-04-27).** Subtickets #1 (config + safety fallback) and #3
(parallel rough-distance loop) are implemented. Subtickets #2, #4, #5, #6 (extended
matrix), and #7 are still open.

What works today:

- `DeepDiff(..., multiprocessing=True, multiprocessing_workers=N, multiprocessing_threshold=K)`.
  Defaults are `False`, `min(4, cpu_count())`, and 64 jobs respectively. Defaults to
  off, so existing users see no behavior change.
- The `(added_hash, removed_hash)` distance loop in
  `_get_most_in_common_pairs_in_iterables` (the `ignore_order=True` hot path) is
  optionally parallelized through `concurrent.futures.ProcessPoolExecutor`.
  Workers compute distances only; pair selection runs in the parent in the same
  serial nested-loop order, so worker completion order never reaches the
  output.
- Safe by construction: pre-calculated distances and distance-cache hits are
  filtered out in the parent before jobs are dispatched. Workers run with
  `cache_purge_level=0` and a sanitized `_parameters` snapshot
  (`multiprocessing=False`, `_mp_config` disabled, no shared mutable caches),
  so they cannot fork-bomb or write back to parent state.
- Picklability of the parameters dict, the iterable compare func, and a
  representative job is checked up front. Any failure causes a clean serial
  fallback rather than an opaque worker crash.
- 23 determinism / fallback tests in `tests/test_multiprocessing.py` (10x
  serial-vs-parallel comparison, tied distances, repeated items in both
  `report_repetition` modes, sets, exclude_paths, ignore_string_case, custom
  module-level hasher, lambda compare-func fallback, recursive-no-nesting).
  All 1149 existing tests still pass.

Code locations:

- `deepdiff/_multiprocessing.py` — `MPConfig`, `normalize_mp_config`,
  `is_pickleable`, `_distance_worker` (module-level for `spawn`),
  `compute_distances_parallel`.
- `deepdiff/diff.py::DeepDiff.__init__` — three new parameters, normalized into
  `self._mp_config`, propagated through `_parameters`.
- `deepdiff/diff.py::DeepDiff._maybe_compute_pair_distances_parallel` — the
  per-call decision/dispatch helper.
- `deepdiff/diff.py::DeepDiff._get_most_in_common_pairs_in_iterables` — gains
  one extra lookup before `_get_rough_distance_of_hashed_objs`.

Not yet implemented (deferred, intentional):

- **Subticket #2** — parallel `_create_hashtable` / `_prep_iterable` /
  `_prep_dict`. The doc itself flags cycle-handling and identity-after-pickle
  risks; these need their own test pass.
- **Subticket #4** — subtree diff parallelism after pairing. `DiffLevel`
  pickling and custom-operator interaction require dedicated work.
- **Subticket #5** — multiprocessing-aware stats semantics. Parent-only stats
  remain meaningful in Phase 1, but no aggregation across workers.
- **Subticket #6** — extended test matrix (numpy, pydantic, namedtuple, group_by,
  large-mixed structures, worker exception propagation tests). Phase 1 ships
  the core determinism harness; the rest is additive.
- **Subticket #7** — benchmarks. The doc says default thresholds shouldn't
  change before benchmarks land; the current `DEFAULT_THRESHOLD = 64` is a
  conservative placeholder.

---

## Goal

Add an opt-in internal multiprocessing mode that can speed up expensive deep hashing and diffing workloads while keeping the final DeepDiff/DeepHash outcome deterministic.

The most important target is `DeepDiff(..., ignore_order=True)`, because that mode often spends the most time hashing iterable items, calculating candidate pair distances, and recursively diffing nested structures.

The result of a multiprocessing run must be the same as a single-process run for supported inputs. Worker completion order must never affect reports, matching decisions, paths, or output ordering.

## Non-Goals

- Do not make the whole recursive engine concurrently mutate one `DeepDiff` instance.
- Do not share `self.tree`, `self.hashes`, `_distance_cache`, or `_stats` directly between worker processes.
- Do not make `max_diffs` and `max_passes` exact replicas of serial accounting. They are stop guards. It is acceptable for their counts to differ in multiprocessing mode as long as they still cap runaway work.
- Do not silently parallelize unsafe callables. If callbacks, custom operators, hashers, or compare functions cannot be safely pickled or executed in workers, fall back to serial behavior or disable only the unsafe parallel section.

## Current Baseline

DeepDiff is already safe to call from multiple separate processes as independent top-level calls. See:

- `tests/test_diff_other.py::TestDiffOther::test_multi_processing1`
- `tests/test_diff_other.py::TestDiffOther::test_multi_processing2_with_ignore_order`
- `tests/test_diff_other.py::TestDiffOther::test_multi_processing3_deephash`

Those tests do not cover internal multiprocessing inside one `DeepDiff` run. This ticket is about one DeepDiff invocation splitting part of its own work across workers.

Important implementation points in the current code:

- `deepdiff/diff.py::DeepDiff.__init__` creates shared mutable state for one diff run:
  - `self.tree`
  - `self.hashes`
  - `self._distance_cache`
  - `self._stats`
  - `self.group_by_keys`
  - `self._numpy_paths`
- `deepdiff/diff.py::_diff` is the main recursive dispatcher.
- `deepdiff/diff.py::_diff_iterable_with_deephash` is the main expensive path for `ignore_order=True`.
- `deepdiff/diff.py::_create_hashtable` hashes iterable items via `DeepHash`.
- `deepdiff/diff.py::_get_most_in_common_pairs_in_iterables` calculates distances between added and removed hashes, then serially chooses pairs.
- `deepdiff/deephash.py::_hash`, `_prep_dict`, and `_prep_iterable` recursively hash child objects.
- Result reporting goes through `deepdiff/diff.py::_report_result`, which writes to `TreeResult` containers backed by `SetOrdered`.

## Determinism Contract

Multiprocessing mode must obey these invariants:

1. A supported multiprocessing run must produce the same public DeepDiff result as the equivalent serial run.
2. Pair selection in `ignore_order=True` must be independent of worker completion order.
3. Result merge order must be based on serial traversal order, not `as_completed()` order.
4. Hash aggregation order must match existing semantics:
   - dictionaries and unordered iterables still sort the hash components where the current implementation sorts them.
   - ordered iterable hashing must preserve item index order when order matters.
5. Workers must not mutate parent process state.
6. Any worker exception must surface as a normal DeepDiff exception, not be swallowed or turned into partial output.
7. Multiprocessing mode must have a reliable serial fallback for unsupported or unsafe inputs.

## Proposed API

Add conservative, opt-in parameters to `DeepDiff` and possibly `DeepHash`.

Suggested names:

```python
DeepDiff(
    t1,
    t2,
    multiprocessing=False,
    multiprocessing_workers=None,
    multiprocessing_threshold=None,
)
```

Open design choice: `multiprocessing` may also accept an integer worker count. If so, keep the API unambiguous and document it.

Suggested behavior:

- `multiprocessing=False`: existing serial behavior.
- `multiprocessing=True`: use `os.cpu_count()` or a conservative default such as `min(4, os.cpu_count() or 1)`.
- `multiprocessing_workers=N`: explicit worker count.
- `multiprocessing_threshold`: minimum amount of work before spawning tasks. Default should avoid slowing small diffs.

The first implementation can keep the parameters private or experimental if preferred, but tests should exercise them explicitly.

## Architecture

Use multiprocessing only around deterministic batches of independent work. The parent process owns traversal decisions, pair selection, result merging, stats finalization, and public result conversion.

Recommended internal structure:

- A small execution helper module or class, for example `deepdiff/multiprocessing.py` or private helpers in `diff.py`.
- A worker input dataclass or plain dict containing:
  - job kind
  - stable job index
  - path string
  - t1/t2 or item object
  - sanitized DeepDiff/DeepHash parameters
  - relevant context such as `_original_type`
- A worker output dataclass or plain dict containing:
  - job index
  - path string
  - computed hash/result/distance/local tree
  - local stats
  - exception details if needed

Do not return live `DiffLevel` objects across process boundaries unless tests prove they pickle reliably and preserve path behavior. Prefer returning plain serializable data for hash and distance tasks. For subtree diff tasks, returning a `TreeResult` may work but must be tested heavily; a safer approach is to return text/delta-style plain result data and merge at the parent.

## Subtickets

### 1. Add Multiprocessing Configuration and Serial Fallback

Implement opt-in configuration without changing serial behavior.

Tasks:

- Add constructor parameters to `DeepDiff`.
- Store normalized multiprocessing settings in `_parameters` so recursive child `DeepDiff` instances receive the same configuration where appropriate.
- Add validation:
  - worker count must be `None` or a positive integer.
  - threshold must be `None` or a non-negative integer.
- Add a helper that decides whether a section may parallelize.
- Add a helper that detects unsafe worker state:
  - unpickleable `custom_operators`
  - unpickleable `hasher`
  - unpickleable `exclude_obj_callback`
  - unpickleable `include_obj_callback`
  - unpickleable `ignore_order_func`
  - unpickleable `iterable_compare_func`
  - objects that fail pickling
- If unsafe, fall back to serial for that section.

Acceptance criteria:

- All existing tests pass with default parameters.
- `DeepDiff(..., multiprocessing=False)` is exactly the current path.
- Unsupported multiprocessing inputs fall back to serial or raise a clear documented error if fallback is not possible.

### 2. Parallelize DeepHash Child Hashing

Start with hashing because parent hash aggregation is already naturally deterministic when child hashes are gathered and combined in serial order.

Candidate locations:

- `deepdiff/deephash.py::_prep_iterable`
- `deepdiff/deephash.py::_prep_dict`
- `deepdiff/diff.py::_create_hashtable`

Recommended first implementation:

- Parallelize `_create_hashtable` for large iterables in `ignore_order=True`.
- Create one job per item, including the item index and parent path.
- Each worker runs `DeepHash(item, hashes=None, parent=parent, apply_hash=True, **deephash_parameters)`.
- Parent sorts outputs by original item index before calling `_add_hash`.
- Parent may merge returned object hashes into `self.hashes` only in deterministic job-index order.

Risks:

- Shared `self.hashes` currently avoids recalculating repeated object hashes. Worker-local hashing loses some cache reuse.
- Some objects cannot be pickled.
- Object identity and cycles may not behave the same after pickling.

Mitigations:

- Enable only above a threshold where process overhead is likely worth it.
- Detect pickling failures and use serial hashing.
- Add cycle tests before enabling parallel hashing for arbitrary recursive objects. Until then, fall back to serial when cycles are detected or suspected.

Acceptance criteria:

- Serial and multiprocessing results match for large lists of dicts, lists of lists, sets, repeated items, and nested mixed structures.
- Result order matches serial output.
- Tests include both `report_repetition=False` and `report_repetition=True`.

### 3. Parallelize Ignore-Order Distance Calculation

This is likely the highest-value optimization for `ignore_order=True`.

Candidate location:

- `deepdiff/diff.py::_get_most_in_common_pairs_in_iterables`

Current serial shape:

1. Build `hashes_added` and `hashes_removed`.
2. Calculate rough distances for candidate `(added_hash, removed_hash)` pairs.
3. Store candidates under `most_in_common_pairs`.
4. Select final pairs serially by ascending distance and `SetOrdered` iteration behavior.

Required deterministic design:

- Parent creates candidate pair jobs in a stable nested-loop order matching current code:
  - outer loop: `hashes_added`
  - inner loop: `hashes_removed`
- Workers compute only distance for one or more candidate pairs.
- Parent receives distance outputs and sorts by original job index before inserting into `most_in_common_pairs`.
- Parent runs the final pairing algorithm serially and unchanged as much as possible.

Do not let workers choose pairs.

Risks:

- Worker-local `_distance_cache` changes cache hit statistics and performance shape.
- `DeepDiff(..., view=DELTA_VIEW)` inside `_get_rough_distance_of_hashed_objs` must receive equivalent parameters.
- `iterable_compare_func` may be unpickleable or side-effectful.
- Floating-point distances must compare the same after process boundaries.

Mitigations:

- Cache stats do not need to match exactly, but final results must.
- Fall back to serial when `iterable_compare_func` is unsafe.
- Keep the final `sorted(distances_to_from_hashes.keys())` pairing step in the parent.
- Add tests that run the same multiprocessing diff many times and compare with serial output.

Acceptance criteria:

- `ignore_order=True` output matches serial for all existing `tests/test_ignore_order.py` cases where multiprocessing mode is enabled.
- Repeated runs with multiprocessing produce identical output.
- Tests include collisions/ties where multiple candidate pairs have the same rough distance.

### 4. Parallelize Selected Subtree Diffs After Pairing

Once `ignore_order=True` pairing is fixed, paired item diffs can be farmed out in some cases.

Candidate locations:

- `deepdiff/diff.py::_diff_iterable_with_deephash`
- `deepdiff/diff.py::_diff_by_forming_pairs_and_comparing_one_by_one`
- dictionary shared-key child comparisons in `_diff_dict`

Recommended approach:

- Parent first determines the exact child jobs in serial traversal order.
- Workers compute local diffs for child pairs.
- Parent merges child results in job index order.

Important: do not parallelize parent-level reporting of added/removed items by completion order. Parent should report or merge in the same order serial traversal would have used.

Risks:

- `DiffLevel` paths and `up/down` links may not be safe to construct in one process and merge in another.
- `TreeResult` contains `DiffLevel` objects and `SetOrdered`; pickling and equality need explicit tests.
- Custom operators can call `custom_report_result` and mutate the diff instance.

Mitigations:

- Initially disable subtree parallelism when custom operators are present.
- Prefer plain result payloads over cross-process `DiffLevel` objects if pickling proves fragile.
- Keep `values_changed`, `iterable_item_added`, `iterable_item_removed`, and `type_changes` merge logic centralized in the parent.

Acceptance criteria:

- Serial and multiprocessing output match for text view, tree view, delta view where supported, and verbose levels 0, 1, and 2.
- Existing delta tests pass if subtree multiprocessing is enabled for delta-compatible cases.
- Custom operators either work deterministically or force serial fallback.

### 5. Stats, Limits, and Progress Logging

Multiprocessing stats do not need to be byte-for-byte identical to serial stats, but they must remain meaningful.

Tasks:

- Define stats semantics for multiprocessing:
  - parent diff count
  - worker diff count aggregate
  - worker pass count aggregate
  - cache hits from parent only, or aggregate worker-local hits separately
- Keep `max_diffs` and `max_passes` as approximate stop guards.
- Ensure workers can stop early if a shared or parent-supplied budget is exhausted.
- Do not run one progress timer per worker.

Suggested behavior:

- Parent owns the progress timer.
- Worker stats are returned and merged after each batch.
- If `max_diffs` or `max_passes` is reached in parent or aggregated worker stats, stop scheduling new work and report the existing warning.

Acceptance criteria:

- `get_stats()` still returns the existing keys.
- Existing `max_diffs` and `max_passes` tests still pass in serial mode.
- Multiprocessing mode has tests showing limits stop runaway work, without requiring exact serial counts.

### 6. Test Matrix for Determinism and Flake Prevention

Add tests that compare serial and multiprocessing outputs directly.

Required test categories:

- `ignore_order=True`, nested lists of dicts.
- `ignore_order=True`, repeated items with `report_repetition=True`.
- `ignore_order=True`, repeated items with `report_repetition=False`.
- Tied candidate distances where more than one pairing is plausible.
- Large mixed structures that trigger the multiprocessing threshold.
- Sets and frozensets.
- Custom `hasher`.
- `ignore_string_case`, `ignore_numeric_type_changes`, `ignore_string_type_changes`.
- `exclude_paths`, `include_paths`, and regex path exclusions.
- `group_by` and `group_by_sort_key`.
- Numpy arrays if numpy is available.
- Objects with `__dict__`, `__slots__`, namedtuple, and pydantic objects if the existing optional dependency setup supports it.
- Pickle failure fallback.
- Worker exception propagation.

Determinism test pattern:

```python
serial = DeepDiff(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1)
for _ in range(20):
    parallel = DeepDiff(
        t1,
        t2,
        ignore_order=True,
        cutoff_intersection_for_pairs=1,
        multiprocessing=True,
        multiprocessing_workers=4,
        multiprocessing_threshold=0,
    )
    assert parallel == serial
```

Also compare `parallel.to_dict()` or equivalent public representation for views where direct object equality is too sensitive.

### 7. Benchmarks

Add benchmark coverage before tuning thresholds.

Candidate workloads:

- Large list of nested dictionaries with `ignore_order=True`.
- Existing benchmark shapes referenced in `docs/optimizations.rst`:
  - deeply nested object with cache disabled/enabled
  - large array-like structures
  - big JSON-like blobs
- Large iterable where many added/removed items require rough distance pairing.

Measure:

- wall time
- peak memory if available
- process spawn overhead
- pickle time if practical
- speedup vs serial
- correctness vs serial result

Acceptance criteria:

- Multiprocessing mode is not enabled by default until benchmarks show a clear win for targeted workloads.
- Default threshold avoids slowdowns on small inputs.

## Implementation Notes

### Stable Job Ordering

Every batch must assign a monotonically increasing `job_index` before submitting work. Parent code must merge by `job_index`.

Do not use `as_completed()` order except to collect results into a temporary map.

### Pairing in `ignore_order=True`

The final pair-selection algorithm is part of the observable behavior. Keep it serial.

Workers may compute distances, but the parent must insert distances into `most_in_common_pairs` in the same order the serial nested loops would have inserted them. This matters when distances tie.

### Caches

Avoid process-shared mutable caches in the first implementation.

Accept that worker-local hashing/distance calculation may reduce cache reuse. A later optimization can add a deterministic parent-owned cache merge, but correctness should come first.

If merging hash cache entries from workers:

- merge in job index order.
- do not overwrite an existing parent entry with a different value.
- add tests for repeated equal-but-not-identical objects.

### Pickling and Start Methods

Use the standard library `concurrent.futures.ProcessPoolExecutor`.

Do not assume Linux `fork` behavior. The implementation should work with `spawn`, especially for macOS and Windows users.

This means worker functions must be module-level functions, not nested closures.

### Thresholds

Multiprocessing should only run when there is enough work to offset serialization and process overhead.

Possible heuristics:

- iterable length above a threshold.
- candidate distance pair count above a threshold.
- estimated nested item count from `DeepHash` count data.

Start conservative. Add benchmarks before changing defaults.

### Unsupported Inputs

Fallback to serial for:

- unpickleable objects.
- unpickleable callables.
- active custom operators unless explicitly tested.
- detected cycles until cycle behavior is proven equivalent.
- generator inputs, because multiprocessing may consume or pickle them differently.

## Risks

- **Non-deterministic pair choices**: if distance jobs are merged by completion order, tied distances can produce different pairings. Mitigation: stable job indices and serial parent pairing.
- **Different object identity after pickling**: cycle detection and identity-sensitive behavior may change in workers. Mitigation: fallback for cycles and tests for self-referential inputs.
- **Callback side effects**: callbacks and custom operators may depend on process-local state or mutate global state. Mitigation: fallback unless proven safe.
- **Result ordering drift**: `TreeResult` and `TextResult` depend on insertion order through `SetOrdered`. Mitigation: parent-only ordered merge.
- **Cache behavior drift**: multiprocessing changes cache locality and stats. Mitigation: do not require exact stats equality; require result equality.
- **Memory growth**: large objects must be pickled and copied into workers. Mitigation: thresholds and benchmarks.
- **Platform differences**: `fork` can hide pickling issues that fail under `spawn`. Mitigation: tests should force or simulate spawn where possible.

## Definition of Done

- Multiprocessing is opt-in.
- Default serial behavior is unchanged.
- `ignore_order=True` multiprocessing results match serial results across the new determinism test matrix.
- Repeated multiprocessing runs are stable.
- Unsupported inputs fall back to serial or raise a clear documented error.
- Tests cover worker exception propagation and pickle fallback.
- Benchmarks demonstrate speedup for at least one realistic `ignore_order=True` workload.
- Documentation explains the experimental status, supported cases, and known limitations.
