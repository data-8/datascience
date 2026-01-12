import numpy as np
import pytest

import datascience as ds


def test_hist_group_normal_no_error():
    t = ds.Table().with_columns(
        'value', ds.make_array(1, 2, 3, 2, 5),
        'cat', ds.make_array('a', 'a', 'a', 'b', 'b')
    )
    # Should not raise
    t.hist('value', group='cat')


def test_hist_group_invalid_label_raises_value_error():
    t = ds.Table().with_columns(
        'value', ds.make_array(1, 2, 3),
        'cat', ds.make_array('x', 'y', 'x')
    )
    with pytest.raises(ValueError):
        t.hist('value', group='missing_col')


def test_hist_group_empty_data_no_error():
    # Empty table after filtering
    t = ds.Table().with_columns(
        'value', ds.make_array(),
        'cat', ds.make_array()
    )
    # Should not raise; creates an empty figure
    t.hist('value', group='cat')

