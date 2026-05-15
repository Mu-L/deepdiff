import random
import concurrent.futures
from deepdiff.helper import not_found
from deepdiff.lfucache import DistanceCache


class TestDistanceCache:

    def test_lru_cache(self, benchmark):
        benchmark(self._test_lru_cache)

    def _test_lru_cache(self):
        cache = DistanceCache(2)
        cache.set('a', value='a_cached')
        cache.set('b', value='b_cached')
        assert 'a' in cache
        assert cache.get('a') == 'a_cached'
        cache.set('c', value='c_cached')
        assert cache.get('a') == 'a_cached'
        assert cache.get('b') is not_found
        assert cache.get('c') == 'c_cached'
        assert cache.get('missing') is not_found

    def test_report_type_values_are_accumulated(self):
        cache = DistanceCache(2)
        cache.set('a', report_type='values_changed', value='root[0]')
        cache.set('a', report_type='values_changed', value='root[1]')
        assert cache.get('a') == {'values_changed': {'root[0]', 'root[1]'}}

    def test_get_multithreading(self):
        keys = 'aaaaaaaaaaaaaaaaaaaaaaaaaaabbc'
        cache = DistanceCache(2)

        def _do_set(cache, key):
            cache.set(key, value='{}_cached'.format(key))

        def _do_get(cache, key):
            return cache.get(key)

        def _key_gen():
            i = 0
            while i < 30000:
                i += 1
                yield random.choice(keys)

        def _random_func(cache, key):
            return random.choice([_do_get, _do_get, _do_set])(cache, key)

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = (executor.submit(_random_func, cache, key) for key in _key_gen())
            for future in concurrent.futures.as_completed(futures):
                future.result()
