#!/usr/bin/env python
"""Tests for fractions.Fraction support in DeepDiff."""
import pytest
import logging
from fractions import Fraction
from decimal import Decimal
from functools import partial
from deepdiff import DeepDiff, DeepHash
from deepdiff.deephash import prepare_string_for_hashing
from deepdiff.helper import number_to_string
from deepdiff.serialization import json_dumps, json_loads


logging.disable(logging.CRITICAL)

# Only the prep part of DeepHash. We don't need to test the actual hash function.
DeepHashPrep = partial(DeepHash, apply_hash=False)


class TestFractionDiff:
    """Tests for DeepDiff with Fraction objects."""

    def test_fraction_value_change(self):
        t1 = {1: Fraction(1, 3)}
        t2 = {1: Fraction(2, 3)}
        ddiff = DeepDiff(t1, t2)
        result = {
            'values_changed': {
                'root[1]': {
                    'new_value': Fraction(2, 3),
                    'old_value': Fraction(1, 3)
                }
            }
        }
        assert result == ddiff

    def test_fraction_no_change(self):
        t1 = Fraction(1, 3)
        t2 = Fraction(1, 3)
        ddiff = DeepDiff(t1, t2)
        assert {} == ddiff

    def test_fraction_vs_float_type_change(self):
        t1 = Fraction(1, 2)
        t2 = 0.5
        ddiff = DeepDiff(t1, t2)
        assert 'type_changes' in ddiff
        assert ddiff['type_changes']['root']['old_type'] == Fraction
        assert ddiff['type_changes']['root']['new_type'] == float

    def test_fraction_vs_int_type_change(self):
        t1 = Fraction(2, 1)
        t2 = 2
        ddiff = DeepDiff(t1, t2)
        assert 'type_changes' in ddiff
        assert ddiff['type_changes']['root']['old_type'] == Fraction
        assert ddiff['type_changes']['root']['new_type'] == int

    def test_fraction_vs_decimal_type_change(self):
        t1 = Fraction(1, 2)
        t2 = Decimal('0.5')
        ddiff = DeepDiff(t1, t2)
        assert 'type_changes' in ddiff

    def test_fraction_in_dict(self):
        t1 = {"a": Fraction(1, 3), "b": Fraction(2, 3)}
        t2 = {"a": Fraction(1, 3), "b": Fraction(3, 4)}
        ddiff = DeepDiff(t1, t2)
        assert 'values_changed' in ddiff
        assert "root['b']" in ddiff['values_changed']

    def test_fraction_in_list(self):
        t1 = [Fraction(1, 2), Fraction(1, 3)]
        t2 = [Fraction(1, 2), Fraction(1, 4)]
        ddiff = DeepDiff(t1, t2)
        result = {
            'values_changed': {
                'root[1]': {
                    'new_value': Fraction(1, 4),
                    'old_value': Fraction(1, 3)
                }
            }
        }
        assert result == ddiff

    def test_fraction_nested(self):
        t1 = {"data": [{"val": Fraction(1, 3)}]}
        t2 = {"data": [{"val": Fraction(2, 3)}]}
        ddiff = DeepDiff(t1, t2)
        assert 'values_changed' in ddiff
        assert "root['data'][0]['val']" in ddiff['values_changed']


class TestFractionIgnoreNumericTypeChanges:
    """Tests for ignore_numeric_type_changes with Fraction."""

    def test_fraction_vs_float_ignored(self):
        t1 = Fraction(1, 2)
        t2 = 0.5
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True)
        assert {} == ddiff

    def test_fraction_vs_int_ignored(self):
        t1 = Fraction(2, 1)
        t2 = 2
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True)
        assert {} == ddiff

    def test_fraction_vs_decimal_ignored(self):
        t1 = Fraction(1, 2)
        t2 = Decimal('0.5')
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True)
        assert {} == ddiff

    def test_fraction_vs_float_different_values(self):
        t1 = Fraction(1, 3)
        t2 = 0.5
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True)
        assert 'values_changed' in ddiff

    def test_fraction_vs_float_in_list_ignored(self):
        t1 = [Fraction(1, 2), Fraction(3, 4)]
        t2 = [0.5, 0.75]
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True)
        assert {} == ddiff

    def test_fraction_vs_int_in_dict_ignored(self):
        t1 = {"a": Fraction(5, 1), "b": Fraction(10, 1)}
        t2 = {"a": 5, "b": 10}
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True)
        assert {} == ddiff

    @pytest.mark.parametrize("t1, t2, significant_digits, result", [
        ([0.5], [Fraction(1, 2)], 5, {}),
        ([Fraction(1, 3)], [0.333333], 5, {}),
        ([Fraction(1, 3)], [Decimal('0.33333')], 5, {}),
        ([1], [Fraction(1, 1)], 5, {}),
        ([-Fraction(1, 2)], [-0.5], 5, {}),
        ([Fraction(22, 7)], [3.14286], 4, {}),
    ])
    def test_ignore_numeric_type_changes_with_fraction(self, t1, t2, significant_digits, result):
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True, significant_digits=significant_digits)
        assert result == ddiff


class TestFractionSignificantDigits:
    """Tests for significant_digits with Fraction."""

    def test_fraction_significant_digits_equal(self):
        t1 = Fraction(1, 3)  # 0.333...
        t2 = Fraction(334, 1000)  # 0.334
        ddiff = DeepDiff(t1, t2, significant_digits=2)
        assert {} == ddiff

    def test_fraction_significant_digits_different(self):
        t1 = Fraction(1, 3)  # 0.333...
        t2 = Fraction(1, 2)  # 0.5
        ddiff = DeepDiff(t1, t2, significant_digits=1)
        assert 'values_changed' in ddiff

    @pytest.mark.parametrize("test_num, t1, t2, significant_digits, number_format_notation, result", [
        (1, Fraction(1, 3), Fraction(334, 1000), 2, "f", {}),
        (2, Fraction(1, 2), Fraction(499, 1000), 2, "f", {}),
        (3, Fraction(1, 2), Fraction(1, 3), 0, "f", {}),
        (4, Fraction(1, 2), Fraction(1, 3), 1, "f",
            {'values_changed': {'root': {'new_value': Fraction(1, 3), 'old_value': Fraction(1, 2)}}}),
        (5, Fraction(22, 7), Fraction(355, 113), 2, "f", {}),  # Two approximations of pi agree to 2 digits
        (6, Fraction(22, 7), Fraction(355, 113), 3, "f",
            {'values_changed': {'root': {'new_value': Fraction(355, 113), 'old_value': Fraction(22, 7)}}}),
    ])
    def test_fraction_significant_digits_and_notation(self, test_num, t1, t2, significant_digits, number_format_notation, result):
        ddiff = DeepDiff(t1, t2, significant_digits=significant_digits,
                         number_format_notation=number_format_notation)
        assert result == ddiff, f"test_fraction_significant_digits_and_notation #{test_num} failed."


class TestFractionMathEpsilon:
    """Tests for math_epsilon with Fraction."""

    def test_fraction_math_epsilon_close(self):
        d1 = {"a": Fraction(7175, 1000)}
        d2 = {"a": Fraction(7174, 1000)}
        res = DeepDiff(d1, d2, math_epsilon=0.01)
        assert res == {}

    def test_fraction_math_epsilon_not_close(self):
        d1 = {"a": Fraction(7175, 1000)}
        d2 = {"a": Fraction(7174, 1000)}
        res = DeepDiff(d1, d2, math_epsilon=0.0001)
        assert 'values_changed' in res

    def test_fraction_vs_float_math_epsilon(self):
        d1 = {"a": Fraction(1, 3)}
        d2 = {"a": 0.333}
        res = DeepDiff(d1, d2, math_epsilon=0.001, ignore_numeric_type_changes=True)
        assert res == {}


class TestFractionIgnoreOrder:
    """Tests for ignore_order with Fraction."""

    def test_fraction_ignore_order(self):
        t1 = [{1: Fraction(1, 3)}, {2: Fraction(2, 3)}]
        t2 = [{2: Fraction(2, 3)}, {1: Fraction(1, 3)}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {} == ddiff

    def test_fraction_ignore_order_with_change(self):
        t1 = [Fraction(1, 2), Fraction(1, 3)]
        t2 = [Fraction(1, 3), Fraction(1, 4)]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert ddiff != {}


class TestFractionAsKey:
    """Tests for Fraction used as dictionary key."""

    def test_fraction_as_dict_key(self):
        t1 = {Fraction(1, 2): "half"}
        t2 = {Fraction(1, 2): "one half"}
        ddiff = DeepDiff(t1, t2)
        assert 'values_changed' in ddiff

    def test_fraction_vs_float_key(self):
        # Fraction(1, 2) == 0.5 and hash(Fraction(1, 2)) == hash(0.5) in Python,
        # so they resolve to the same dict key. DeepDiff sees no difference.
        t1 = {Fraction(1, 2): "value"}
        t2 = {0.5: "value"}
        ddiff = DeepDiff(t1, t2)
        assert ddiff == {}

    def test_fraction_vs_float_key_ignore_numeric(self):
        t1 = {Fraction(1, 2): "value"}
        t2 = {0.5: "value"}
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True)
        assert {} == ddiff


class TestFractionNumberToString:
    """Tests for number_to_string with Fraction."""

    @pytest.mark.parametrize("t1, t2, significant_digits, number_format_notation, expected_result", [
        (Fraction(1, 3), 0.333333, 5, "f", True),
        (Fraction(1, 2), 0.5, 5, "f", True),
        (Fraction(1, 2), 0.5, 5, "e", True),
        (Fraction(1, 3), Fraction(1, 4), 1, "f", ('0.3', '0.2')),
        (Fraction(22, 7), 3.14286, 4, "f", True),
        (Fraction(0), 0.0, 5, "f", True),
        (Fraction(-1, 2), -0.5, 5, "f", True),
    ])
    def test_number_to_string_fraction(self, t1, t2, significant_digits, number_format_notation, expected_result):
        st1 = number_to_string(t1, significant_digits=significant_digits, number_format_notation=number_format_notation)
        st2 = number_to_string(t2, significant_digits=significant_digits, number_format_notation=number_format_notation)
        if expected_result is True:
            assert st1 == st2
        else:
            assert st1 == expected_result[0]
            assert st2 == expected_result[1]


class TestFractionDeepHash:
    """Tests for DeepHash with Fraction."""

    def test_fraction_hash(self):
        result = DeepHash(Fraction(1, 3))
        assert result[Fraction(1, 3)]

    def test_fraction_same_value_same_hash(self):
        result1 = DeepHash(Fraction(1, 2))
        result2 = DeepHash(Fraction(1, 2))
        assert result1[Fraction(1, 2)] == result2[Fraction(1, 2)]

    def test_fraction_different_value_different_hash(self):
        result1 = DeepHash(Fraction(1, 2))
        result2 = DeepHash(Fraction(1, 3))
        assert result1[Fraction(1, 2)] != result2[Fraction(1, 3)]

    def test_fraction_vs_float_hash_different_by_default(self):
        result1 = DeepHash(Fraction(1, 2))
        result2 = DeepHash(0.5)
        assert result1[Fraction(1, 2)] != result2[0.5]

    def test_fraction_vs_float_hash_same_with_ignore_numeric_type(self):
        result1 = DeepHash(Fraction(1, 2), ignore_numeric_type_changes=True)
        result2 = DeepHash(0.5, ignore_numeric_type_changes=True)
        assert result1[Fraction(1, 2)] == result2[0.5]

    def test_fraction_hash_prep(self):
        result = DeepHashPrep(Fraction(1, 3))
        assert 'Fraction' in result[Fraction(1, 3)]

    def test_fraction_hash_prep_ignore_numeric_type(self):
        result = DeepHashPrep(Fraction(1, 2), ignore_numeric_type_changes=True)
        assert 'number' in result[Fraction(1, 2)]

    def test_fraction_hash_significant_digits(self):
        r1 = DeepHashPrep(Fraction(1, 3), significant_digits=2)
        r2 = DeepHashPrep(Fraction(334, 1000), significant_digits=2)
        assert r1[Fraction(1, 3)] == r2[Fraction(334, 1000)]


class TestFractionSerialization:
    """Tests for JSON serialization of Fraction values."""

    def test_fraction_to_json(self):
        t1 = Fraction(1, 3)
        t2 = Fraction(2, 3)
        ddiff = DeepDiff(t1, t2)
        json_str = ddiff.to_json()
        assert json_str is not None
        assert '"new_value"' in json_str

    def test_fraction_integer_value_serialization(self):
        """Fraction with denominator 1 should serialize as int."""
        result = json_dumps(Fraction(5, 1))
        assert result == '5'

    def test_fraction_float_value_serialization(self):
        """Fraction with denominator != 1 should serialize as float."""
        result = json_dumps(Fraction(1, 2))
        assert result == '0.5'

    def test_fraction_json_roundtrip(self):
        t1 = {"a": Fraction(1, 2), "b": [1, 2]}
        t2 = {"a": Fraction(3, 4), "b": [1, 3]}
        ddiff = DeepDiff(t1, t2)
        json_str = ddiff.to_json()
        loaded = json_loads(json_str)
        assert loaded is not None
