# DeepDiff v 9.0.0

![Downloads](https://img.shields.io/pypi/dm/deepdiff.svg?style=flat)
![Python Versions](https://img.shields.io/pypi/pyversions/deepdiff.svg?style=flat)
![License](https://img.shields.io/pypi/l/deepdiff.svg?version=latest)
[![Build Status](https://github.com/seperman/deepdiff/workflows/Unit%20Tests/badge.svg)](https://github.com/seperman/deepdiff/actions)
[![codecov](https://codecov.io/gh/seperman/deepdiff/branch/master/graph/badge.svg?token=KkHZ3siA3m)](https://codecov.io/gh/seperman/deepdiff)

**DeepDiff is now part of [Qluster](/qluster).**

*If you're building workflows around data validation and correction, [Qluster](/qluster) gives your team a structured way to manage rules, review failures, approve fixes, and reuse decisions—without building the entire system from scratch.*

## Modules

- [DeepDiff](https://zepworks.com/deepdiff/current/diff.html): Deep Difference of dictionaries, iterables, strings, and ANY other object.
- [DeepSearch](https://zepworks.com/deepdiff/current/dsearch.html): Search for objects within other objects.
- [DeepHash](https://zepworks.com/deepdiff/current/deephash.html): Hash any object based on their content.
- [Delta](https://zepworks.com/deepdiff/current/delta.html): Store the difference of objects and apply them to other objects.
- [Extract](https://zepworks.com/deepdiff/current/extract.html): Extract an item from a nested Python object using its path.
- [commandline](https://zepworks.com/deepdiff/current/commandline.html): Use DeepDiff from commandline.

Tested on Python 3.10+ and PyPy3.

- **[Documentation](https://zepworks.com/deepdiff/9.0.0/)**

## What is new?

Please check the [ChangeLog](CHANGELOG.md) file for the detailed information.

DeepDiff 9-0-0
- migration note:
    - `to_dict()` and `to_json()` now accept a `verbose_level` parameter and always return a usable text-view dict. When the original view is `'tree'`, they default to `verbose_level=2` for full detail. The old `view_override` parameter is removed. To get the previous results, you will need to pass the explicit verbose_level to `to_json` and `to_dict` if you are using the tree view.
- Dropping support for Python 3.9
- Support for python 3.14
- Added support for callable `group_by` thanks to @echan5
- Added `FlatDeltaDict` TypedDict for `to_flat_dicts` return type
- Fixed colored view display when all list items are removed thanks to @yannrouillard
- Fixed `hasattr()` swallowing `AttributeError` in `__slots__` handling for objects with `__getattr__` thanks to @tpvasconcelos
- Fixed `ignore_order=True` missing int-vs-float type changes
- Fixed Delta producing phantom entries when items both move and change values with `iterable_compare_func` thanks to @devin13cox
- Fixed `_convert_oversized_ints` failing on NamedTuples
- Fixed orjson `TypeError` for integers exceeding 64-bit range
- Fixed parameter bug in `to_flat_dicts` where `include_action_in_path` and `report_type_changes` were not being passed through
- Fixed `ignore_keys` issue in `detailed__dict__` thanks to @vitalis89
- Fixed logarithmic similarity type hint thanks to @ljames8
- Added `Fraction` numeric support thanks to @akshat62

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
5. Build `uv build`

# Contribute

1. Please make your PR against the dev branch
2. Please make sure that your PR has tests. Since DeepDiff is used in many sensitive data driven projects, we strive to maintain around 100% test coverage on the code.

Please run `pytest --cov=deepdiff --runslow` to see the coverage report. Note that the `--runslow` flag will run some slow tests too. In most cases you only want to run the fast tests which so you won't add the `--runslow` flag.

Or to see a more user friendly version, please run: `pytest --cov=deepdiff --cov-report term-missing --runslow`.

Thank you!

# Authors

Please take a look at the [AUTHORS](AUTHORS.md) file.
