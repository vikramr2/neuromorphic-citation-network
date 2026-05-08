import pytest
from kg_builder.schemas import Triple


def test_triple_creation():
    t = Triple(h="Alice", r="knows", t="Bob")
    assert t.h == "Alice"
    assert t.r == "knows"
    assert t.t == "Bob"


def test_triple_validation():
    with pytest.raises(ValueError):
        Triple(h="", r="knows", t="Bob")  # empty h
