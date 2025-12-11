import pytest
from datascience import Table

def test_apply_non_callable():
    table = Table().with_columns("numbers", [1, 2, 3])
    with pytest.raises(TypeError):
        table.apply(123, "numbers")

def test_apply_invalid_column_name():
    table = Table().with_columns("numbers", [1, 2, 3])
    with pytest.raises(ValueError):
        table.apply(lambda x: x, "non_existent_column")

def test_apply_function_returns_invalid_type():
    table = Table().with_columns("numbers", [1, 2, 3])

    def bad_fn(x):
        return list(range(x))  
    with pytest.raises(Exception):
        table.apply(bad_fn, "numbers")

