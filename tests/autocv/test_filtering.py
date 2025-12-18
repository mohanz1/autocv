from autocv.utils import filtering


class Dummy:
    def __init__(self, val):
        self.foo = type("Bar", (), {"bar": val})()


def test_get_first_match():
    items = [Dummy(1), Dummy(2), Dummy(3)]
    found = filtering.get_first(items, foo__bar=2)
    assert isinstance(found, Dummy)
    assert found.foo.bar == 2


def test_get_first_no_match():
    items = [Dummy(1), Dummy(2)]
    assert filtering.get_first(items, foo__bar=5) is None


def test_get_first_no_kwargs_returns_none():
    items = [Dummy(1), Dummy(2)]
    assert filtering.get_first(items) is None


def test_get_first_multiple_matchers():
    class MultiDummy:
        def __init__(self, foo_bar, other):
            self.foo = type("Bar", (), {"bar": foo_bar})()
            self.other = other

    items = [MultiDummy(1, "a"), MultiDummy(2, "b"), MultiDummy(2, "c")]
    found = filtering.get_first(items, foo__bar=2, other="b")
    assert isinstance(found, MultiDummy)
    assert found.foo.bar == 2
    assert found.other == "b"


def test_get_first_multiple_matchers_no_match():
    class MultiDummy:
        def __init__(self, foo_bar, other):
            self.foo = type("Bar", (), {"bar": foo_bar})()
            self.other = other

    items = [MultiDummy(2, "b")]
    assert filtering.get_first(items, foo__bar=2, other="missing") is None


def test_make_attr_getter_is_cached():
    getter_1 = filtering._make_attr_getter("foo__bar")
    getter_2 = filtering._make_attr_getter("foo__bar")
    assert getter_1 is getter_2


def test_find_first_match():
    assert filtering.find_first(lambda x: x > 5, [1, 3, 7]) == 7


def test_find_first_no_match():
    assert filtering.find_first(lambda x: x > 10, [1, 3, 7]) is None
