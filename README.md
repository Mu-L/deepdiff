# DeepDiff v 8.7.0

![Downloads](https://img.shields.io/pypi/dm/deepdiff.svg?style=flat)
![Python Versions](https://img.shields.io/pypi/pyversions/deepdiff.svg?style=flat)
![License](https://img.shields.io/pypi/l/deepdiff.svg?version=latest)
[![Build Status](https://github.com/seperman/deepdiff/workflows/Unit%20Tests/badge.svg)](https://github.com/seperman/deepdiff/actions)
[![codecov](https://codecov.io/gh/seperman/deepdiff/branch/master/graph/badge.svg?token=KkHZ3siA3m)](https://codecov.io/gh/seperman/deepdiff)

**DeepDiff is now part of [Qluster](https://getqluster.com).**

*If you're building workflows around data validation and correction, [Qluster](https://getqluster.com) gives your team a structured way to manage rules, review failures, approve fixes, and reuse decisions—without building the entire system from scratch.*

## Modules

- [DeepDiff](https://zepworks.com/deepdiff/current/diff.html): Deep Difference of dictionaries, iterables, strings, and ANY other object.
- [DeepSearch](https://zepworks.com/deepdiff/current/dsearch.html): Search for objects within other objects.
- [DeepHash](https://zepworks.com/deepdiff/current/deephash.html): Hash any object based on their content.
- [Delta](https://zepworks.com/deepdiff/current/delta.html): Store the difference of objects and apply them to other objects.
- [Extract](https://zepworks.com/deepdiff/current/extract.html): Extract an item from a nested Python object using its path.
- [commandline](https://zepworks.com/deepdiff/current/commandline.html): Use DeepDiff from commandline.

Tested on Python 3.10+ and PyPy3.

- **[Documentation](https://zepworks.com/deepdiff/8.7.0/)**

## What is new?

Please check the [ChangeLog](CHANGELOG.md) file for the detailed information.

DeepDiff 8-7-0
- migration note:
    - `to_dict()` and `to_json()` now accept a `verbose_level` parameter and always return a usable text-view dict. When the original view is `'tree'`, they default to `verbose_level=2` for full detail. The old `view_override` parameter is removed. To get the previous results, you will need to pass the explicit verbose_level to `to_json` and `to_dict` if you are using the tree view.
- Dropping support for Python 3.9
- Support for python 3.14
- Added support for callable `group_by` thanks to @echan5
- Added `FlatDeltaDict` TypedDict for `to_flat_dicts` return type
- Fixed colored view display when all list items are removed thanks to @yannrouillard
- Fixed `hasattr()` swallowing `AttributeError` in `__slots__` handling for objects with `__getattr__` thanks to @tpvasconcelos
- Fixed `ignore_order=True` missing int-vs-float type changes
- Always use t1 path for reporting thanks to @devin13cox
- Fixed `_convert_oversized_ints` failing on NamedTuples
- Fixed orjson `TypeError` for integers exceeding 64-bit range
- Fixed parameter bug in `to_flat_dicts` where `include_action_in_path` and `report_type_changes` were not being passed through
- Fixed `ignore_keys` issue in `detailed__dict__` thanks to @vitalis89
- Fixed logarithmic similarity type hint thanks to @ljames8
- Added `Fraction` numeric support thanks to @akshat62

DeepDiff 8-6-2
- **Security (CVE-2026-33155):** Fixed a memory exhaustion DoS vulnerability in `_RestrictedUnpickler` by limiting the maximum allocation size for `bytes` and `bytearray` during deserialization.

DeepDiff 8-6-1
- Patched security vulnerability in the Delta class which was vulnerable to class pollution via its constructor, and when combined with a gadget available in DeltaDiff itself, it could lead to Denial of Service and Remote Code Execution (via insecure Pickle deserialization).

DeepDiff 8-6-0

- Added Colored View thanks to @mauvilsa 
- Added support for applying deltas to NamedTuple thanks to @paulsc 
- Fixed test_delta.py with Python 3.14 thanks to @Romain-Geissler-1A
- Added python property serialization to json
- Added ip address serialization
- Switched to UV from pip
- Added Claude.md
- Added uuid hashing thanks to @akshat62
- Added `ignore_uuid_types` flag to DeepDiff to avoid type reports when comparing UUID and string.
- Added comprehensive type hints across the codebase (multiple commits for better type safety)
- Added support for memoryview serialization
- Added support for bytes serialization (non-UTF8 compatible)
- Fixed bug where group_by with numbers would leak type info into group path reports
- Fixed bug in `_get_clean_to_keys_mapping without` explicit significant digits
- Added support for python dict key serialization
- Enhanced support for IP address serialization with safe module imports
- Added development tooling improvements (pyright config, .envrc example)
- Updated documentation and development instructions


DeepDiff 8-5-0

- Updating deprecated pydantic calls
- Switching to pyproject.toml
- Fix for moving nested tables when using iterable_compare_func.  by 
- Fix recursion depth limit when hashing numpy.datetime64
- Moving from legacy setuptools use to pyproject.toml


DeepDiff 8-4-2

- fixes the type hints for the base
- fixes summarize so if json dumps fails, we can still get a repr of the results
- adds ipaddress support


## Installation

### Install from PyPi:

`pip install deepdiff`

If you want to use DeepDiff from commandline:

`pip install "deepdiff[cli]"`

If you want to improve the performance of DeepDiff with certain functionalities such as improved json serialization:

`pip install "deepdiff[optimize]"`

Install optional packages:
- [yaml](https://pypi.org/project/PyYAML/)
- [tomli](https://pypi.org/project/tomli/) (python 3.10 and older) and [tomli-w](https://pypi.org/project/tomli-w/) for writing
- [clevercsv](https://pypi.org/project/clevercsv/) for more robust CSV parsing
- [orjson](https://pypi.org/project/orjson/) for speed and memory optimized parsing
- [pydantic](https://pypi.org/project/pydantic/)


# Documentation

<https://zepworks.com/deepdiff/current/>

# ChangeLog

Please take a look at the [CHANGELOG](CHANGELOG.md) file.

# Survey

:mega: **Please fill out our [fast 5-question survey](https://forms.gle/E6qXexcgjoKnSzjB8)** so that we can learn how & why you use DeepDiff, and what improvements we should make. Thank you! :dancers:

# Local dev

1. Clone the repo
2. Switch to the dev branch
3. Create your own branch
4. Install dependencies

    - Method 1: Use [`uv`](https://github.com/astral-sh/uv) to install the dependencies:  `uv sync --all-extras`.
    - Method 2: Use pip: `pip install -e ".[cli,coverage,dev,docs,static,test]"`
5. Build `flit build`

# Contribute

1. Please make your PR against the dev branch
2. Please make sure that your PR has tests. Since DeepDiff is used in many sensitive data driven projects, we strive to maintain around 100% test coverage on the code.

Please run `pytest --cov=deepdiff --runslow` to see the coverage report. Note that the `--runslow` flag will run some slow tests too. In most cases you only want to run the fast tests which so you won't add the `--runslow` flag.

Or to see a more user friendly version, please run: `pytest --cov=deepdiff --cov-report term-missing --runslow`.

Thank you!

# Authors

Please take a look at the [AUTHORS](AUTHORS.md) file.
