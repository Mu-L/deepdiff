import pytest
from deepdiff import DeepDiff, DeepHash, DeepSearch, grep
from deepdiff.path import (
    GlobPathMatcher, compile_glob_paths, path_has_wildcard,
    _path_to_elements, SINGLE_WILDCARD, MULTI_WILDCARD,
)
from deepdiff.helper import separate_wildcard_and_exact_paths


# ── path_has_wildcard detection ──────────────────────────────────────


class TestPathHasWildcard:

    @pytest.mark.parametrize("path, expected", [
        ("root[*]", True),
        ("root[**]", True),
        ("root.*", True),
        ("root.**", True),
        ("root['users'][*]['name']", True),
        ("root[**]['password']", True),
        ("root['*']", False),       # literal key named '*'
        ("root['**']", False),      # literal key named '**'
        ("root['foo']['bar']", False),
        ("root[0][1]", False),
        ("root.foo.bar", False),
        ("root[*][*]", True),       # multiple wildcards
        ("root[**][**]", True),
        ("root.*.bar.*", True),     # multiple dot wildcards
    ])
    def test_detection(self, path, expected):
        assert path_has_wildcard(path) is expected


# ── _path_to_elements parsing of wildcards ───────────────────────────


class TestWildcardParsing:

    @pytest.mark.parametrize("path, expected", [
        ("root[*]", (('root', 'GETATTR'), (SINGLE_WILDCARD, 'GET'))),
        ("root[**]", (('root', 'GETATTR'), (MULTI_WILDCARD, 'GET'))),
        ("root['users'][*]['password']", (
            ('root', 'GETATTR'), ('users', 'GET'), (SINGLE_WILDCARD, 'GET'), ('password', 'GET'),
        )),
        ("root[**]['secret']", (
            ('root', 'GETATTR'), (MULTI_WILDCARD, 'GET'), ('secret', 'GET'),
        )),
        ("root.*.name", (
            ('root', 'GETATTR'), (SINGLE_WILDCARD, 'GETATTR'), ('name', 'GETATTR'),
        )),
        ("root[*][*]", (
            ('root', 'GETATTR'), (SINGLE_WILDCARD, 'GET'), (SINGLE_WILDCARD, 'GET'),
        )),
    ])
    def test_parsing(self, path, expected):
        assert _path_to_elements(path) == expected

    def test_literal_star_key_not_wildcard(self):
        """root['*'] should parse as a literal string '*', not a wildcard token."""
        elems = _path_to_elements("root['*']")
        # The element should be a plain string, not a _WildcardToken
        assert elems[1][0] == '*'
        assert elems[1][0] != SINGLE_WILDCARD
        assert isinstance(elems[1][0], str)

    def test_literal_double_star_key_not_wildcard(self):
        """root['**'] should parse as a literal string '**', not a wildcard token."""
        elems = _path_to_elements("root['**']")
        assert elems[1][0] == '**'
        assert elems[1][0] != MULTI_WILDCARD
        assert isinstance(elems[1][0], str)

    def test_wildcard_token_repr(self):
        """_WildcardToken repr should return the symbol string."""
        assert repr(SINGLE_WILDCARD) == '*'
        assert repr(MULTI_WILDCARD) == '**'

    def test_wildcard_token_hash(self):
        """_WildcardToken instances should be hashable and usable in sets/dicts."""
        s = {SINGLE_WILDCARD, MULTI_WILDCARD}
        assert len(s) == 2
        assert SINGLE_WILDCARD in s
        d = {SINGLE_WILDCARD: 'one', MULTI_WILDCARD: 'many'}
        assert d[SINGLE_WILDCARD] == 'one'


# ── separate_wildcard_and_exact_paths ────────────────────────────────


class TestSeparateWildcardPaths:

    def test_none_input(self):
        exact, globs = separate_wildcard_and_exact_paths(None)
        assert exact is None
        assert globs is None

    def test_empty_input(self):
        exact, globs = separate_wildcard_and_exact_paths(set())
        assert exact is None
        assert globs is None

    def test_all_exact(self):
        exact, globs = separate_wildcard_and_exact_paths({"root['foo']", "root['bar']"})
        assert exact == {"root['foo']", "root['bar']"}
        assert globs is None

    def test_all_wildcards(self):
        exact, globs = separate_wildcard_and_exact_paths({"root[*]", "root[**]['x']"})
        assert exact is None
        assert len(globs) == 2

    def test_mixed(self):
        exact, globs = separate_wildcard_and_exact_paths(
            {"root['foo']", "root[*]['bar']"}
        )
        assert exact == {"root['foo']"}
        assert len(globs) == 1
        assert globs[0].original_pattern == "root[*]['bar']"

    def test_wildcard_must_start_with_root(self):
        with pytest.raises(ValueError, match="Wildcard paths must start with 'root'"):
            separate_wildcard_and_exact_paths({"[*]['foo']"})


# ── GlobPathMatcher.match ────────────────────────────────────────────


class TestGlobPathMatcherMatch:

    # ── single wildcard [*] ──

    @pytest.mark.parametrize("target, expected", [
        ("root['a']", True),
        ("root[0]", True),
        ("root[99]", True),
        ("root", False),            # too short
        ("root['a']['b']", False),  # too long
    ])
    def test_single_wildcard_basic(self, target, expected):
        m = GlobPathMatcher("root[*]")
        assert m.match(target) is expected

    @pytest.mark.parametrize("target, expected", [
        ("root['users']['alice']['password']", True),
        ("root['users'][0]['password']", True),
        ("root['users'][99]['password']", True),
        ("root['users']['password']", False),            # missing middle segment
        ("root['users']['a']['b']['password']", False),  # too many middle segments
        ("root['users']['alice']['email']", False),      # wrong last segment
    ])
    def test_single_wildcard_in_middle(self, target, expected):
        m = GlobPathMatcher("root['users'][*]['password']")
        assert m.match(target) is expected

    def test_multiple_single_wildcards(self):
        """root[*][*] matches exactly two segments after root."""
        m = GlobPathMatcher("root[*][*]")
        assert m.match("root['a']['b']") is True
        assert m.match("root[0][1]") is True
        assert m.match("root['a']") is False
        assert m.match("root['a']['b']['c']") is False

    # ── double wildcard [**] ──

    @pytest.mark.parametrize("target, expected", [
        ("root", True),                       # zero segments
        ("root['a']", True),                  # one segment
        ("root['a']['b']['c']", True),        # many segments
        ("root[0][1][2]", True),              # numeric indices
    ])
    def test_double_wildcard_standalone(self, target, expected):
        m = GlobPathMatcher("root[**]")
        assert m.match(target) is expected

    @pytest.mark.parametrize("target, expected", [
        ("root['password']", True),                         # ** matches zero
        ("root['a']['password']", True),                    # ** matches one
        ("root['a']['b']['c']['password']", True),          # ** matches many
        ("root['a']['b']", False),                          # doesn't end with password
        ("root['password']['extra']", False),               # extra after password
    ])
    def test_double_wildcard_before_key(self, target, expected):
        m = GlobPathMatcher("root[**]['password']")
        assert m.match(target) is expected

    def test_double_wildcard_both_ends(self):
        m = GlobPathMatcher("root[**]['config'][**]['value']")
        assert m.match("root['config']['value']") is True
        assert m.match("root['a']['config']['value']") is True
        assert m.match("root['a']['config']['b']['c']['value']") is True
        assert m.match("root['config']['x']") is False
        assert m.match("root['value']") is False

    def test_double_wildcard_zero_match_in_middle(self):
        """** between two fixed segments can match zero segments."""
        m = GlobPathMatcher("root['a'][**]['b']")
        assert m.match("root['a']['b']") is True           # ** matches zero
        assert m.match("root['a']['x']['b']") is True      # ** matches one
        assert m.match("root['a']['x']['y']['b']") is True  # ** matches two

    def test_adjacent_double_wildcards(self):
        m = GlobPathMatcher("root[**][**]['x']")
        assert m.match("root['x']") is True
        assert m.match("root['a']['x']") is True
        assert m.match("root['a']['b']['x']") is True

    # ── dot notation wildcards ──

    def test_dot_single_wildcard(self):
        m = GlobPathMatcher("root.*.name")
        assert m.match("root.user.name") is True
        assert m.match("root.name") is False

    def test_dot_double_wildcard(self):
        m = GlobPathMatcher("root.**.name")
        assert m.match("root.name") is True
        assert m.match("root.a.name") is True
        assert m.match("root.a.b.name") is True

    # ── mixed bracket and dot ──

    def test_mixed_bracket_and_dot_wildcard(self):
        m = GlobPathMatcher("root[*].name")
        assert m.match("root['user'].name") is True
        assert m.match("root[0].name") is True


# ── GlobPathMatcher.match_or_is_ancestor ─────────────────────────────


class TestGlobPathMatcherAncestor:

    def test_ancestor_of_double_wildcard(self):
        m = GlobPathMatcher("root[**]['password']")
        assert m.match_or_is_ancestor("root['users']") is True
        assert m.match_or_is_ancestor("root") is True

    def test_match_also_returns_true(self):
        m = GlobPathMatcher("root[**]['password']")
        assert m.match_or_is_ancestor("root['password']") is True

    def test_any_path_is_ancestor_with_double_wildcard(self):
        """With ** in the pattern, any intermediate path could lead to a match."""
        m = GlobPathMatcher("root[**]['password']")
        assert m.match_or_is_ancestor("root['x']") is True
        assert m.match_or_is_ancestor("root['x']['y']['z']") is True

    def test_single_wildcard_ancestor_positive(self):
        m = GlobPathMatcher("root['users'][*]['password']")
        assert m.match_or_is_ancestor("root['users']") is True
        assert m.match_or_is_ancestor("root") is True

    def test_single_wildcard_ancestor_negative(self):
        """A path that diverges from a single-wildcard pattern is not an ancestor."""
        m = GlobPathMatcher("root['users'][*]['password']")
        assert m.match_or_is_ancestor("root['other']") is False


# ── GlobPathMatcher.match_or_is_descendant ───────────────────────────


class TestGlobPathMatcherDescendant:

    def test_descendant_of_match(self):
        m = GlobPathMatcher("root[**]['config']")
        assert m.match_or_is_descendant("root['config']['value']") is True
        assert m.match_or_is_descendant("root['config']['a']['b']") is True

    def test_exact_match(self):
        m = GlobPathMatcher("root[**]['config']")
        assert m.match_or_is_descendant("root['config']") is True

    def test_not_descendant_or_match(self):
        m = GlobPathMatcher("root[**]['secret']")
        assert m.match_or_is_descendant("root['config']['db']['host']") is False

    def test_ancestor_is_not_descendant(self):
        m = GlobPathMatcher("root['users'][*]['password']")
        assert m.match_or_is_descendant("root['users']") is False

    def test_descendant_of_single_wildcard_match(self):
        m = GlobPathMatcher("root[*]")
        assert m.match_or_is_descendant("root['a']['nested']") is True


# ── compile_glob_paths ───────────────────────────────────────────────


class TestCompileGlobPaths:

    def test_none_returns_none(self):
        assert compile_glob_paths(None) is None

    def test_empty_returns_none(self):
        assert compile_glob_paths([]) is None

    def test_compiles_list(self):
        result = compile_glob_paths(["root[*]", "root[**]['x']"])
        assert len(result) == 2
        assert all(isinstance(r, GlobPathMatcher) for r in result)


# ── DeepDiff integration: exclude_paths with wildcards ───────────────


class TestDeepDiffExcludeGlob:

    def test_exclude_single_wildcard(self):
        t1 = {'users': {'alice': {'name': 'Alice', 'pw': 's1'}, 'bob': {'name': 'Bob', 'pw': 's2'}}}
        t2 = {'users': {'alice': {'name': 'Alice', 'pw': 'c1'}, 'bob': {'name': 'Bobby', 'pw': 'c2'}}}
        diff = DeepDiff(t1, t2, exclude_paths=["root['users'][*]['pw']"])
        changed = diff.get('values_changed', {})
        assert "root['users']['bob']['name']" in changed
        assert "root['users']['alice']['pw']" not in changed
        assert "root['users']['bob']['pw']" not in changed

    def test_exclude_double_wildcard(self):
        t1 = {
            'config': {'db': {'host': 'localhost', 'secret': 'abc'},
                       'api': {'nested': {'secret': 'xyz'}}},
            'name': 'app'
        }
        t2 = {
            'config': {'db': {'host': 'remotehost', 'secret': 'def'},
                       'api': {'nested': {'secret': 'uvw'}}},
            'name': 'app2'
        }
        diff = DeepDiff(t1, t2, exclude_paths=["root[**]['secret']"])
        changed = diff.get('values_changed', {})
        assert "root['config']['db']['host']" in changed
        assert "root['name']" in changed
        assert "root['config']['db']['secret']" not in changed
        assert "root['config']['api']['nested']['secret']" not in changed

    def test_exclude_wildcard_with_list(self):
        t1 = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
        t2 = [{'name': 'Alice', 'age': 31}, {'name': 'Bobby', 'age': 26}]
        diff = DeepDiff(t1, t2, exclude_paths=["root[*]['age']"])
        changed = diff.get('values_changed', {})
        assert "root[1]['name']" in changed
        assert "root[0]['age']" not in changed
        assert "root[1]['age']" not in changed

    def test_exclude_mix_exact_and_wildcard(self):
        t1 = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
        t2 = {'a': 10, 'b': 20, 'c': {'d': 30, 'e': 40}}
        diff = DeepDiff(t1, t2, exclude_paths=["root['a']", "root['c'][*]"])
        changed = diff.get('values_changed', {})
        assert "root['b']" in changed
        assert "root['a']" not in changed
        assert "root['c']['d']" not in changed
        assert "root['c']['e']" not in changed

    def test_exclude_nested_list_of_dicts(self):
        t1 = {'data': [{'id': 1, 'meta': {'ts': 100}}, {'id': 2, 'meta': {'ts': 200}}]}
        t2 = {'data': [{'id': 1, 'meta': {'ts': 999}}, {'id': 2, 'meta': {'ts': 888}}]}
        diff = DeepDiff(t1, t2, exclude_paths=["root['data'][*]['meta']"])
        assert diff == {}

    def test_exclude_with_type_changes(self):
        t1 = {'a': {'x': 1, 'y': 'hello'}}
        t2 = {'a': {'x': 'changed_type', 'y': 'world'}}
        diff = DeepDiff(t1, t2, exclude_paths=["root[*]['x']"])
        changed = diff.get('values_changed', {})
        assert "root['a']['y']" in changed
        assert 'type_changes' not in diff


# ── DeepDiff integration: include_paths with wildcards ───────────────


class TestDeepDiffIncludeGlob:

    def test_include_single_wildcard(self):
        t1 = {'users': {'alice': {'name': 'Alice', 'pw': 's1'}, 'bob': {'name': 'Bob', 'pw': 's2'}}}
        t2 = {'users': {'alice': {'name': 'Alice2', 'pw': 'c1'}, 'bob': {'name': 'Bobby', 'pw': 'c2'}}}
        diff = DeepDiff(t1, t2, include_paths=["root['users'][*]['name']"])
        changed = diff.get('values_changed', {})
        assert "root['users']['alice']['name']" in changed
        assert "root['users']['bob']['name']" in changed
        assert "root['users']['alice']['pw']" not in changed
        assert "root['users']['bob']['pw']" not in changed

    def test_include_double_wildcard(self):
        t1 = {
            'config': {'db': {'host': 'localhost', 'secret': 'abc'},
                       'api': {'url': 'http://api', 'nested': {'secret': 'xyz'}}},
            'name': 'app'
        }
        t2 = {
            'config': {'db': {'host': 'remotehost', 'secret': 'def'},
                       'api': {'url': 'http://api2', 'nested': {'secret': 'uvw'}}},
            'name': 'app2'
        }
        diff = DeepDiff(t1, t2, include_paths=["root[**]['secret']"])
        changed = diff.get('values_changed', {})
        assert "root['config']['db']['secret']" in changed
        assert "root['config']['api']['nested']['secret']" in changed
        assert "root['config']['db']['host']" not in changed
        assert "root['config']['api']['url']" not in changed
        assert "root['name']" not in changed

    def test_include_mix_exact_and_wildcard(self):
        t1 = {
            'config': {'db': {'host': 'localhost', 'secret': 'abc'}},
            'name': 'app'
        }
        t2 = {
            'config': {'db': {'host': 'remotehost', 'secret': 'def'}},
            'name': 'app2'
        }
        diff = DeepDiff(t1, t2, include_paths=["root[**]['secret']", "root['name']"])
        changed = diff.get('values_changed', {})
        assert "root['config']['db']['secret']" in changed
        assert "root['name']" in changed
        assert "root['config']['db']['host']" not in changed

    def test_include_wildcard_no_changes(self):
        t1 = {'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}}
        t2 = {'a': {'x': 1, 'y': 20}, 'b': {'x': 3, 'y': 40}}
        diff = DeepDiff(t1, t2, include_paths=["root[*]['x']"])
        assert diff == {}

    def test_include_wildcard_with_added_keys(self):
        """When a new key is added, include_paths restricts reporting to matching paths only."""
        t1 = {'a': {'name': 'x'}}
        t2 = {'a': {'name': 'y'}, 'b': {'name': 'z'}}
        diff = DeepDiff(t1, t2, include_paths=["root[*]['name']"])
        changed = diff.get('values_changed', {})
        assert "root['a']['name']" in changed
        # root['b'] addition is not reported because the add is at root['b'],
        # not at root[*]['name']
        assert 'dictionary_item_added' not in diff

    def test_include_double_wildcard_with_nested_list(self):
        t1 = {'data': [{'scores': [1, 2]}, {'scores': [3, 4]}]}
        t2 = {'data': [{'scores': [1, 2]}, {'scores': [3, 5]}]}
        diff = DeepDiff(t1, t2, include_paths=["root[**]['scores']"])
        changed = diff.get('values_changed', {})
        assert "root['data'][1]['scores'][1]" in changed
        assert len(changed) == 1


# ── Backward compatibility ───────────────────────────────────────────


class TestBackwardCompatibility:

    def test_exact_exclude_paths_unchanged(self):
        t1 = {"for life": "vegan", "ingredients": ["no meat", "no eggs"]}
        t2 = {"for life": "vegan", "ingredients": ["veggies", "tofu"]}
        ddiff = DeepDiff(t1, t2, exclude_paths={"root['ingredients']"})
        assert ddiff == {}

    def test_exact_include_paths_unchanged(self):
        t1 = {"for life": "vegan", "ingredients": ["no meat", "no eggs"]}
        t2 = {"for life": "vegan2", "ingredients": ["veggies", "tofu"]}
        ddiff = DeepDiff(t1, t2, include_paths={"root['for life']"})
        changed = ddiff.get('values_changed', {})
        assert "root['for life']" in changed
        assert len(changed) == 1

    def test_exclude_regex_paths_unchanged(self):
        t1 = [{'a': 1, 'b': 2}, {'c': 4, 'b': 5}]
        t2 = [{'a': 1, 'b': 3}, {'c': 4, 'b': 5}]
        ddiff = DeepDiff(t1, t2, exclude_regex_paths=[r"root\[\d+\]\['b'\]"])
        assert ddiff == {}

    def test_shorthand_paths_unchanged(self):
        t1 = {"for life": "vegan", "ingredients": ["no meat"]}
        t2 = {"for life": "vegan", "ingredients": ["veggies"]}
        ddiff = DeepDiff(t1, t2, exclude_paths={"ingredients"})
        assert ddiff == {}

    def test_include_paths_with_nested_prefix(self):
        """Existing prefix-based include logic must still work."""
        t1 = {"foo": {"bar": {"veg": "potato", "fruit": "apple"}}}
        t2 = {"foo": {"bar": {"veg": "potato", "fruit": "peach"}}}
        ddiff = DeepDiff(t1, t2, include_paths="root['foo']['bar']")
        changed = ddiff.get('values_changed', {})
        assert "root['foo']['bar']['fruit']" in changed


# ── DeepSearch integration ───────────────────────────────────────────


class TestDeepSearchGlob:

    def test_exclude_glob_in_search(self):
        obj = {'a': {'secret': 'find_me', 'name': 'x'}, 'b': {'secret': 'find_me', 'name': 'y'}}
        result = DeepSearch(obj, 'find_me', exclude_paths=["root[*]['secret']"])
        assert result == {}

    def test_exclude_deep_glob_in_search(self):
        obj = {'level1': {'level2': {'target': 'needle', 'other': 'needle'}}}
        result = DeepSearch(obj, 'needle', exclude_paths=["root[**]['target']"])
        matched = result.get('matched_values', {})
        assert "root['level1']['level2']['other']" in matched
        assert "root['level1']['level2']['target']" not in matched

    def test_exclude_glob_via_grep(self):
        obj = [{'secret': 'findme', 'name': 'x'}, {'secret': 'findme', 'name': 'y'}]
        result = obj | grep('findme', exclude_paths=["root[*]['secret']"])
        assert result == {}

    def test_exclude_deep_glob_in_list_search(self):
        obj = [[1, 2, 'target'], [3, 'target', 4]]
        result = DeepSearch(obj, 'target', exclude_paths=["root[*][2]"])
        matched = result.get('matched_values', {})
        assert 'root[1][1]' in matched
        assert 'root[0][2]' not in matched

    def test_search_with_mixed_exact_and_glob_exclude(self):
        obj = {'a': 'val', 'b': {'c': 'val'}, 'd': {'e': {'f': 'val'}}}
        result = DeepSearch(obj, 'val', exclude_paths=["root['a']", "root[**]['f']"])
        matched = result.get('matched_values', {})
        assert "root['b']['c']" in matched
        assert "root['a']" not in matched
        assert "root['d']['e']['f']" not in matched


# ── DeepHash integration ─────────────────────────────────────────────


class TestDeepHashGlob:

    def test_exclude_exact_makes_hash_equal(self):
        t1 = {'name': 'app', 'secret': 'abc'}
        t2 = {'name': 'app', 'secret': 'def'}
        h1 = DeepHash(t1, exclude_paths=["root['secret']"])
        h2 = DeepHash(t2, exclude_paths=["root['secret']"])
        assert h1[t1] == h2[t2]

    def test_exclude_glob_wildcard_makes_hash_equal(self):
        t1 = {'a': {'secret': 'x', 'name': 'n1'}, 'b': {'secret': 'y', 'name': 'n2'}}
        t2 = {'a': {'secret': 'X', 'name': 'n1'}, 'b': {'secret': 'Y', 'name': 'n2'}}
        h1 = DeepHash(t1, exclude_paths=["root[*]['secret']"])
        h2 = DeepHash(t2, exclude_paths=["root[*]['secret']"])
        assert h1[t1] == h2[t2]

    def test_exclude_deep_glob_makes_hash_equal(self):
        t1 = {'a': {'b': {'secret': 1, 'val': 2}}}
        t2 = {'a': {'b': {'secret': 99, 'val': 2}}}
        h1 = DeepHash(t1, exclude_paths=["root[**]['secret']"])
        h2 = DeepHash(t2, exclude_paths=["root[**]['secret']"])
        assert h1[t1] == h2[t2]

    def test_exclude_glob_hash_not_equal_when_included_part_differs(self):
        t1 = {'a': {'secret': 'x', 'name': 'n1'}}
        t2 = {'a': {'secret': 'x', 'name': 'DIFFERENT'}}
        h1 = DeepHash(t1, exclude_paths=["root[*]['secret']"])
        h2 = DeepHash(t2, exclude_paths=["root[*]['secret']"])
        assert h1[t1] != h2[t2]


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:

    def test_wildcard_with_ignore_order(self):
        t1 = [{'name': 'a', 'pw': '1'}, {'name': 'b', 'pw': '2'}]
        t2 = [{'name': 'b', 'pw': '20'}, {'name': 'a', 'pw': '10'}]
        diff = DeepDiff(t1, t2, ignore_order=True, exclude_paths=["root[*]['pw']"])
        assert diff == {}

    def test_include_wildcard_with_ignore_order(self):
        t1 = [{'name': 'a', 'pw': '1'}, {'name': 'b', 'pw': '2'}]
        t2 = [{'name': 'b', 'pw': '20'}, {'name': 'a', 'pw': '10'}]
        diff = DeepDiff(t1, t2, ignore_order=True, include_paths=["root[*]['name']"])
        assert diff == {}

    def test_wildcard_with_added_removed_keys(self):
        t1 = {'users': {'alice': {'name': 'Alice', 'pw': 'a'}}}
        t2 = {'users': {'alice': {'name': 'Alice', 'pw': 'b'}, 'bob': {'name': 'Bob', 'pw': 'c'}}}
        diff = DeepDiff(t1, t2, exclude_paths=["root['users'][*]['pw']"])
        added = diff.get('dictionary_item_added', [])
        assert any("bob" in str(p) for p in added)

    def test_empty_diff_with_wildcard(self):
        t1 = {'a': 1}
        t2 = {'a': 1}
        diff = DeepDiff(t1, t2, exclude_paths=["root[*]"])
        assert diff == {}

    def test_root_double_wildcard_excludes_everything(self):
        t1 = {'a': 1, 'b': {'c': 2}}
        t2 = {'a': 10, 'b': {'c': 20}}
        diff = DeepDiff(t1, t2, exclude_paths=["root[**]"])
        assert diff == {}

    def test_wildcard_with_custom_object(self):
        class Obj:
            def __init__(self, name, secret):
                self.name = name
                self.secret = secret
        o1 = Obj('a', 's1')
        o2 = Obj('b', 's2')
        diff = DeepDiff(o1, o2, exclude_paths=["root.secret"])
        changed = diff.get('values_changed', {})
        assert 'root.name' in changed
        assert 'root.secret' not in changed

    def test_exclude_wildcard_with_removed_items(self):
        t1 = [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}, {'x': 5, 'y': 6}]
        t2 = [{'x': 1, 'y': 2}]
        diff = DeepDiff(t1, t2, exclude_paths=["root[*]['y']"])
        removed = diff.get('iterable_item_removed', {})
        assert len(removed) == 2

    def test_wildcard_verbose_level_2(self):
        t1 = {'a': {'x': 1}, 'b': {'x': 2}}
        t2 = {'a': {'x': 10}, 'b': {'x': 20}}
        diff = DeepDiff(t1, t2, exclude_paths=["root[*]['x']"], verbose_level=2)
        assert diff == {}

    def test_multiple_wildcards_in_one_pattern(self):
        t1 = {'a': {'b': {'c': 1}}, 'x': {'y': {'z': 2}}}
        t2 = {'a': {'b': {'c': 10}}, 'x': {'y': {'z': 20}}}
        diff = DeepDiff(t1, t2, exclude_paths=["root[*][*][*]"])
        assert diff == {}

    def test_wildcard_does_not_affect_identical_objects(self):
        t1 = {'a': [1, 2, 3], 'b': {'c': 'd'}}
        diff = DeepDiff(t1, t1, exclude_paths=["root[**]"])
        assert diff == {}

    def test_wildcard_as_single_exclude_path_string(self):
        """exclude_paths accepts a single string, not just a list."""
        t1 = {'a': {'x': 1}, 'b': {'x': 2}}
        t2 = {'a': {'x': 10}, 'b': {'x': 20}}
        diff = DeepDiff(t1, t2, exclude_paths="root[*]['x']")
        assert diff == {}

    def test_include_wildcard_as_single_string(self):
        """include_paths accepts a single string, not just a list."""
        t1 = {'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}}
        t2 = {'a': {'x': 10, 'y': 2}, 'b': {'x': 30, 'y': 4}}
        diff = DeepDiff(t1, t2, include_paths="root[*]['x']")
        changed = diff.get('values_changed', {})
        assert len(changed) == 2
        assert "root['a']['y']" not in changed

    def test_literal_star_key_not_treated_as_wildcard(self):
        """A dict key named '*' should be treated literally, not as a wildcard."""
        t1 = {'*': 1, 'a': 2, 'b': 3}
        t2 = {'*': 10, 'a': 20, 'b': 30}
        # Exclude only the literal '*' key, not all keys
        diff = DeepDiff(t1, t2, exclude_paths=["root['*']"])
        changed = diff.get('values_changed', {})
        # '*' key should be excluded, but 'a' and 'b' should still show changes
        assert "root['*']" not in changed
        assert "root['a']" in changed
        assert "root['b']" in changed

    def test_glob_matcher_literal_star_vs_wildcard(self):
        """GlobPathMatcher(root['*']) should only match literal '*' key."""
        matcher = GlobPathMatcher("root['*']")
        # Should NOT match arbitrary keys (that's what root[*] is for)
        assert not matcher.match("root['hello']")
        assert not matcher.match("root['a']")
        # Should match the literal '*' key
        assert matcher.match("root['*']")

    def test_exclude_takes_precedence_over_include(self):
        """When a path matches both include and exclude, exclude should win."""
        t1 = {'x': 1, 'y': 2}
        t2 = {'x': 10, 'y': 20}
        diff = DeepDiff(t1, t2, include_paths=["root['x']"], exclude_paths=["root['x']"])
        assert diff == {}

    def test_exclude_glob_takes_precedence_over_include_glob(self):
        """Exclude glob should take precedence over include glob for same path."""
        t1 = {'a': {'x': 1}, 'b': {'x': 2}}
        t2 = {'a': {'x': 10}, 'b': {'x': 20}}
        diff = DeepDiff(t1, t2, include_paths=["root[*]['x']"], exclude_paths=["root['a'][*]"])
        changed = diff.get('values_changed', {})
        assert "root['a']['x']" not in changed
        assert "root['b']['x']" in changed

    def test_include_glob_with_custom_operator(self):
        """include_glob_paths should filter custom operator reports to only matching paths."""
        from deepdiff.operator import BaseOperator

        class AlwaysReport(BaseOperator):
            """Reports on dict-level comparisons, which are ancestors of the glob target."""
            def give_up_diffing(self, level, diff_instance):
                diff_instance.custom_report_result(
                    'custom_report', level, {'message': 'custom'})
                return True

        t1 = {'a': {'x': 1}, 'b': {'x': 2}}
        t2 = {'a': {'x': 10}, 'b': {'x': 20}}
        # Operator fires on dict type — so it reports at root['a'] and root['b'] level
        op = AlwaysReport(types=[dict])
        diff = DeepDiff(t1, t2, include_paths=["root[*]['x']"], custom_operators=[op])
        custom = diff.get('custom_report', set())
        # root['a'] and root['b'] are ancestors of the glob pattern, not matches
        # or descendants — _skip_report_for_include_glob should filter them out
        assert "root['a']" not in custom
        assert "root['b']" not in custom

    def test_mixed_exact_include_and_glob_include(self):
        """When both exact include_paths and glob include_paths are used together,
        exact matches should pass through without glob filtering."""
        t1 = {'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}}
        t2 = {'a': {'x': 10, 'y': 20}, 'b': {'x': 30, 'y': 40}}
        diff = DeepDiff(
            t1, t2,
            include_paths=["root['a']", "root[*]['x']"],
        )
        changed = diff.get('values_changed', {})
        # root['a']['y'] is covered by exact include root['a']
        assert "root['a']['y']" in changed
        # root['b']['x'] is covered by glob root[*]['x']
        assert "root['b']['x']" in changed
        # root['b']['y'] is NOT covered by either
        assert "root['b']['y']" not in changed
