"""Tests for the group_by parameter of Deepdiff"""

import pytest

from deepdiff import DeepDiff


class TestGetKeyForGroupBy:
    def test_group_by_string(self):
        """Test where group_by is a single key (string)."""
        row = {'first': 'John', 'middle': 'Joe', 'last': 'Smith'}
        group_by = 'first'
        item_name = 't1'
        actual = DeepDiff._get_key_for_group_by(row, group_by, item_name)
        expected = 'John'

        assert actual == expected

    def test_group_by_callable(self):
        """Test where group_by is callable."""
        row = {'id': 123, 'demographics': {'names': {'first': 'John', 'middle': 'Joe', 'last': 'Smith'}}}
        group_by = lambda x: x['demographics']['names']['first']
        item_name = 't1'
        actual = DeepDiff._get_key_for_group_by(row, group_by, item_name)
        expected = 'John'
        assert actual == expected

    def test_group_by_key_error(self):
        """Test where group_by is a key that is not in the row."""
        row = {'id': 123, 'demographics': {'names': {'first': 'John', 'middle': 'Joe', 'last': 'Smith'}}}
        group_by = 'someotherkey'
        item_name = 't1'
        with pytest.raises(KeyError):
            DeepDiff._get_key_for_group_by(row, group_by, item_name)


class TestGroupBy:
    def test_group_by_callable(self):
        """Test where group_by is a callable."""
        t1 = [
            {'id': 'AA', 'demographics': {'names': {'first': 'Joe', 'middle': 'John', 'last': 'Nobody'}}},
            {'id': 'BB', 'demographics': {'names': {'first': 'James', 'middle': 'Joyce', 'last': 'Blue'}}},
            {'id': 'CC', 'demographics': {'names': {'first': 'Mike', 'middle': 'Mark', 'last': 'Apple'}}},
        ]

        t2 = [
            {'id': 'AA', 'demographics': {'names': {'first': 'Joe', 'middle': 'John', 'last': 'Nobody'}}},
            {'id': 'BB', 'demographics': {'names': {'first': 'James', 'middle': 'Joyce', 'last': 'Brown'}}},
            {'id': 'CC', 'demographics': {'names': {'first': 'Mike', 'middle': 'Charles', 'last': 'Apple'}}},
        ]

        actual = DeepDiff(t1, t2, group_by=lambda x: x['demographics']['names']['first'])
        expected = {
            'values_changed': {
                "root['James']['demographics']['names']['last']": {'new_value': 'Brown', 'old_value': 'Blue'},
                "root['Mike']['demographics']['names']['middle']": {'new_value': 'Charles', 'old_value': 'Mark'},
            },
        }
        assert actual == expected
