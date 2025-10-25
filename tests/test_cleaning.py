import numpy as np
import pytest
from datascience import Table


def test_drop_na_rows_any():
    t = Table().with_columns(
        'a', [1, None, 3, np.nan],
        'b', ['x', 'y', None, 'z'],
    )
    res = t.drop_na()
    # rows with indices 0 and 3 remain
    assert res.num_rows == 2
    assert list(res['a']) == [1, np.nan]
    assert list(res['b']) == ['x', 'z']


def test_drop_na_rows_subset_all():
    t = Table().with_columns(
        'a', [None, None, 3],
        'b', [None, 2, None],
    )
    # Drop rows where all of subset ['a','b'] are NA -> only row 2 kept
    res = t.drop_na(subset=['a','b'], how='all')
    assert res.num_rows == 2


def test_drop_na_columns():
    t = Table().with_columns(
        'a', [1, 2, 3],
        'b', [None, None, None],
        'c', ['x', 'y', 'z'],
    )
    res = t.drop_na(axis='columns')
    assert res.labels == ('a','c')


def test_fill_na_value():
    t = Table().with_columns(
        'a', [1, None, 3],
        'b', ['x', None, 'z'],
    )
    res = t.fill_na(value=0)
    assert list(res['a']) == [1, 0, 3]
    assert list(res['b']) == ['x', 0, 'z']


def test_fill_na_strategy_mean():
    t = Table().with_columns(
        'a', [1.0, np.nan, 3.0],
    )
    res = t.fill_na(strategy='mean')
    assert pytest.approx(float(res['a'][1]), rel=1e-6) == 2.0


def test_drop_duplicates_basic():
    t = Table().with_columns(
        'a', [1, 1, 2, 2, 3],
        'b', ['x','x','y','z','z'],
    )
    # keep first by default
    res = t.drop_duplicates()
    assert res.num_rows == 4
    # subset only column a, drop all duplicates -> keep only unique a's
    res2 = t.drop_duplicates(subset='a', keep='none')
    assert list(res2['a']) == [3]


def test_convert_types_mapping_and_infer():
    t = Table().with_columns(
        'a', ['1','2','3'],
        'b', ['1.5','2.5','nan'],
        'c', ['x','y','z'],
    )
    res = t.convert_types(infer=True)
    assert all(isinstance(v, (int, np.integer)) for v in res['a'])
    # b becomes float (last is nan -> stays as string 'nan' may become float nan)
    assert len(res['b']) == 3
    # mapping overrides
    res2 = t.convert_types(mapping={'c': lambda v: v.upper()})
    assert list(res2['c']) == ['X','Y','Z']
