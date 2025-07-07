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


def test_find_first_match():
    assert filtering.find_first(lambda x: x > 5, [1, 3, 7]) == 7


def test_find_first_no_match():
    assert filtering.find_first(lambda x: x > 10, [1, 3, 7]) is None
