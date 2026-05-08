from kg_builder.io_utils import normalize_text


def test_normalize_text():
    assert normalize_text("Hello World!") == "hello world"
    assert normalize_text("Café") == "cafe"
    assert normalize_text("  Test   ") == "test"
    assert normalize_text("") == ""
