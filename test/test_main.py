import pytest

class TestMain(object):
    def test_tautology(self):
        assert True

    def test_equality(self):
        two = 1 + 1
        assert two == 2
