import doctest
import re
import pytest
import warnings
import numpy as np
from numpy.testing import assert_array_equal
from datascience import *
import pandas as pd
from io import BytesIO


#########
# Utils #
#########


@pytest.fixture(scope='function')
def table():
    """Setup Scrabble table"""
    return Table().with_columns([
        'letter', ['a', 'b', 'c', 'z'],
        'count', [9, 3, 3, 1],
        'points', [1, 2, 2, 10],
        ])


@pytest.fixture(scope='function')
def table2():
    """Setup second table"""
    return Table().with_columns([
        ['points', (1, 2, 3)],
        ['names', ('one', 'two', 'three')],
        ])


@pytest.fixture(scope='function')
def table3():
    """Setup third table; same labels as first but in a different order."""
    return Table().with_columns([
        'count', [0, 54, 5],
        'points', [3, 10, 24],
        'letter', ['x', 'y', 'z'],
        ])

@pytest.fixture(scope='function')
def table4():
    """Setup fourth table; three overlapping columns with table."""
    return Table().with_columns([
        'letter', ['a', 'b', 'c', '8', 'a'],
        'count', [9, 3, 2, 0, 9],
        'different label', [1, 4, 2, 1, 1],
        'name', ['Gamma', 'Delta', 'Epsilon', 'Alpha', 'Beta']
        ])

@pytest.fixture(scope='function')
def table5():
    """Setup fifth table; has NaNs in it"""
    return Table().with_columns([
        'letter', ['a', 'b', 'c', 'd', 'y', 'z'],
        'count', [9, 3, 3, 4, 2, 1],
        'points', [1, 2, 2, 2, 4, 10],
        'person_id', [np.float64('nan'), np.float64('nan'), np.float64('nan'), 1, 2, 3]
        ])

@pytest.fixture(scope='function')
def numbers_table():
    """Setup table containing only numbers"""
    return Table().with_columns([
        'count', [9, 3, 3, 1],
        'points', [1, 2, 2, 10],
        ])

@pytest.fixture(scope='function')
def categories_table():
    """Setup a table with a column to serve as pivot keys and
    a columns of values to bin for each key."""
    return Table(['key', 'val']).with_rows([
            ['a', 1],
            ['a', 1],
            ['a', 2],
            ['b', 1],
            ['b', 2],
            ['b', 2]])

@pytest.fixture(scope='module')
def t():
    """Create one table for entire module"""
    return table()


@pytest.fixture(scope='module')
def u():
    """Setup second alphanumeric table"""
    return table2()

@pytest.fixture(scope='function')
def scrabble_table2():
    """Setup Scrabble table"""
    return Table().with_columns([
        'letter', ['a', 'b', 'c', 'z'],
        'count', [9, 3, 3, 1],
        'count_2', [9, 3, 3, 1],
        'pointsplus1', [2, 3, 3, 11],
        ])

@pytest.fixture(scope='function')
def empty_table():
    """Setup empty table"""
    return Table().with_columns("A",[])


def assert_equal(string1, string2):
    string1, string2 = str(string1), str(string2)
    whitespace = re.compile(r'\s')
    purify = lambda s: whitespace.sub('', s)
    assert purify(string1) == purify(string2), "\n%s\n!=\n%s" % (string1, string2)


############
# Doctests #
############


def test_doctests():
    results = doctest.testmod(tables, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.failed == 0


############
# Overview #
############


def test_basic(table):
    """Tests that t works"""
    t = table
    assert_equal(t, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)

def test_column(table):
    """Test table.values()"""
    t = table
    assert_array_equal(t.column('letter'), np.array(['a', 'b', 'c', 'z']))
    assert_array_equal(t.column(1), np.array([9, 3, 3, 1]))
    with pytest.raises(ValueError):
        t.column(-1)
    with pytest.raises(ValueError):
        t.column('abc')

def test_values():
    t1 = Table().with_columns({
        'row1': ['a', 'b', 'c'],
        'row2': ['d', 'e', 'f'],
    })
    assert_array_equal(t1.values, np.array(t1.columns, None).T)
    t2 = Table().with_columns({
        'row1': ['x', 'y', 'z'],
        'row2': [1, 2, 3],
    })
    assert_array_equal(t2.values, np.array(t2.columns, object).T)

def test_basic_points(table):
    t = table
    assert_array_equal(t['points'], np.array([1, 2, 2, 10]))


def test_basic_rows(table):
    t = table
    assert_equal(
        t.rows[2],
        "Row(letter='c', count=3, points=2)")

def test_row_conversion_to_np_array(table):
    t = table
    t_subset = t.select("count", "points")
    assert_array_equal(np.array(t_subset.row(0)), np.array([9, 1]))

def test_select(table):
    t = table
    test = t.select('points', 1)
    assert_equal(test, """
    points | count
    1      | 9
    2      | 3
    2      | 3
    10     | 1
    """)

def test_drop(table):
    t = table
    test = t.drop(['points', 1])
    assert_equal(test, """
    letter
    a
    b
    c
    z
    """)

def test_take(table):
    t = table
    test = t.take([1, 2])
    assert_equal(test, """
    letter | count | points
    b      | 3     | 2
    c      | 3     | 2
    """)


def test_take_slice(table):
    t = table
    test = t.take[1:3]
    assert_equal(test, """
    letter | count | points
    b      | 3     | 2
    c      | 3     | 2
    """)


def test_take_slice_single(table):
    t = table
    test = t.take[1]
    assert_equal(test, """
    letter | count | points
    b      | 3     | 2
    """)


def test_take_iterable(table):
    t = table
    test = t.take[0, 2]
    assert_equal(test, """
    letter | count | points
    a      | 9     | 1
    c      | 3     | 2
    """)

def test_take_floating_args(table):
    t = table
    test = t.take(0, 2)
    assert_equal(test, """
    letter | count | points
    a      | 9     | 1
    c      | 3     | 2
    """)


def test_exclude(table):
    t = table
    test = t.exclude([1, 3])
    assert_equal(test, """
    letter | count | points
    a      | 9     | 1
    c      | 3     | 2
    """)


def test_exclude_slice(table):
    t = table
    test = t.exclude[1:3]
    assert_equal(test, """
    letter | count | points
    a      | 9     | 1
    z      | 1     | 10
    """)


def test_exclude_slice_single(table):
    t = table
    test = t.exclude[1]
    assert_equal(test, """
    letter | count | points
    a      | 9     | 1
    c      | 3     | 2
    z      | 1     | 10
    """)


def test_exclude_iterable(table):
    t = table
    test = t.exclude[0, 2]
    assert_equal(test, """
    letter | count | points
    b      | 3     | 2
    z      | 1     | 10
    """)

def test_exclude_floating_args(table):
    t = table
    test = t.exclude(1, 3)
    assert_equal(test, """
    letter | count | points
    a      | 9     | 1
    c      | 3     | 2
    """)


def test_stats(table):
    t = table
    test = t.stats()
    assert_equal(test, """
    statistic | letter | count | points
    min       | a      | 1     | 1
    max       | z      | 9     | 10
    median    |        | 3     | 2
    sum       |        | 16    | 15
    """)


def test_stats_with_numpy(table):
    t = table
    test = t.stats([np.mean, np.std, np.var])
    assert_equal(test, """
    statistic | letter | count | points
    mean      |        | 4     | 3.75
    std       |        | 3     | 3.63146
    var       |        | 9     | 13.1875""")


def test_where(table):
    t = table
    test = t.where('points', 2)
    assert_equal(test, """
    letter | count | points
    b      | 3     | 2
    c      | 3     | 2
    """)
    test = t.where(2, 2)
    assert_equal(test, """
    letter | count | points
    b      | 3     | 2
    c      | 3     | 2
    """)


def test_where_conditions(table):
    t = table
    t['totals'] = t['points'] * t['count']
    test = t.where(t['totals'] > 8)
    assert_equal(test, """
    letter | count | points | totals
    a      | 9     | 1      | 9
    z      | 1     | 10     | 10
    """)


def test_where_predicates(table):
    t = table
    t['totals'] = t['points'] * t['count']
    test = t.where('totals', are.between(9, 11))
    assert_equal(test, """
    letter | count | points | totals
    a      | 9     | 1      | 9
    z      | 1     | 10     | 10
    """)

@pytest.mark.filterwarnings("error")
def test_where_predicates_nowarning_on_str(table):
    t = table
    test = t.where('letter', are.equal_to('a'))
    assert_equal(test, """
    letter | count | points
    a      | 9     | 1
    """)

def test_where_predicates_warning(table, capsys):
    t1 = table.copy()
    count1 = t1['count'] - 1
    count1[0] += 1
    t1['count1'] = count1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with (pytest.raises(ValueError)):
            test = t1.where('count', are.equal_to(t1.column("count1")))
        assert len(w) == 1
        assert "Do not pass an array or list to a predicate." in str(w[-1].message)
    test = t1.where('count', are.equal_to, t1.column('count1'))
    assert_equal(test, """
    letter | count | points | count1
    a      | 9     | 1      | 9
    """)


def test_sort(table):
    t = table
    t['totals'] = t['points'] * t['count']
    test = t.sort('points')
    assert_equal(test, """
    letter | count | points | totals
    a      | 9     | 1      | 9
    b      | 3     | 2      | 6
    c      | 3     | 2      | 6
    z      | 1     | 10     | 10
    """)
    test = t.sort(3)
    assert_equal(test, """
    letter | count | points | totals
    b      | 3     | 2      | 6
    c      | 3     | 2      | 6
    a      | 9     | 1      | 9
    z      | 1     | 10     | 10
    """)


def test_sort_args(table):
    t = table
    t['totals'] = t['points'] * t['count']
    test = t.sort('points', descending=True, distinct=True)
    assert_equal(test, """
    letter | count | points | totals
    z      | 1     | 10     | 10
    b      | 3     | 2      | 6
    a      | 9     | 1      | 9
    """)


def test_sort_descending(table):
    sorted_table = table.sort('points', descending=True)
    assert_equal(sorted_table, """
    letter | count | points
    z      | 1     | 10
    b      | 3     | 2
    c      | 3     | 2
    a      | 9     | 1
    """)


def test_sort_syntax(table):
    t = table
    t['totals'] = t['points'] * t['count']
    test = t.sort(-t['totals'])
    assert_equal(test, """
    letter | count | points | totals
    z      | 1     | 10     | 10
    a      | 9     | 1      | 9
    b      | 3     | 2      | 6
    c      | 3     | 2      | 6
    """)


def test_group(table, table5):
    t = table
    test = t.group('points')
    assert_equal(test, """
    points | count
    1      | 1
    2      | 2
    10     | 1
    """)
    test = t.group(2)
    assert_equal(test, """
    points | count
    1      | 1
    2      | 2
    10     | 1
    """)

def test_group_nans(table5):
    t = table5
    test = t.group('person_id')
    assert_equal(test, """
    person_id  | count
    nan        | 3
    1          | 1
    2          | 1
    3          | 1
    """)


def test_group_with_func(table):
    t = table
    t['totals'] = t['points'] * t['count']
    test = t.group('points', sum)
    assert_equal(test, """
    points | letter sum | count sum | totals sum
    1      |            | 9         | 9
    2      |            | 6         | 12
    10     |            | 1         | 10
    """)


def test_groups(table):
    t = table.copy()
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    test = t.groups(['points', 'early'])
    assert_equal(test, """
    points | early | count
    1      | False | 1
    1      | True  | 1
    2      | True  | 2
    10     | False | 1
    """)

def test_groups_using_group(table):
    t = table.copy()
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    test = t.group(['points', 'early'])
    assert_equal(test, """
    points | early | count
    1      | False | 1
    1      | True  | 1
    2      | True  | 2
    10     | False | 1
    """)

def test_groups_list(table):
    t = table.copy()
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    test = t.groups(['points', 'early'], lambda s: s)
    assert_equal(test, """
    points | early | letter    | count | totals
    1      | False | ['e']     | [12]  | [12]
    1      | True  | ['a']     | [9]   | [9]
    2      | True  | ['b' 'c'] | [3 3] | [6 6]
    10     | False | ['z']     | [1]   | [10]
    """)


def test_groups_collect(table):
    t = table.copy()
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    test = t.select(['points', 'early', 'count']).groups(['points', 'early'], sum)
    assert_equal(test, """
    points | early | count sum
    1      | False | 12
    1      | True  | 9
    2      | True  | 6
    10     | False | 1
    """)

def test_groups_nans(table5):
    t = table5
    test = t.group(['person_id', 'points'])
    assert_equal(test, """
    person_id  | points |count
    nan        | 1      | 1
    nan        | 2      | 2
    1          | 2      | 1
    2          | 4      | 1
    3          | 10     | 1
    """)

def test_groups_with_nonexistent_label(table):
    t = table.copy()
    with (pytest.raises(ValueError)):
        t.groups(['bad_label', 'points'])

def test_join(table, table2):
    """Tests that join works, not destructive"""
    t = table
    u = table2
    t['totals'] = t['points'] * t['count']
    assert_equal(t.join('points', u), """
    points | letter | count | totals | names
    1      | a      | 9     | 9      | one
    2      | b      | 3     | 6      | two
    2      | c      | 3     | 6      | two
    """)
    assert_equal(u, """
    points  | names
    1       | one
    2       | two
    3       | three
    """)
    assert_equal(t, """
    letter | count | points | totals
    a      | 9     | 1      | 9
    b      | 3     | 2      | 6
    c      | 3     | 2      | 6
    z      | 1     | 10     | 10
    """)

def test_set_format_with_no_format_column(mocker, table):
    MockNumberFormatter = mocker.NonCallableMock(spec=NumberFormatter)
    del MockNumberFormatter.format_column
    with pytest.raises(Exception):
        table.set_format('count', MockNumberFormatter)

def test_join_html(table, table2):
    """Test that join doesn't crash with formatting."""
    t = table
    u = table2
    t = t.set_format('count', NumberFormatter)
    t.as_html()
    u.join('points', t, 'points').as_html()

def test_pivot_counts(table, table2):
    t = table.copy()
    u = table2
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    test = t.pivot('points', 'early')
    assert_equal(test, """
    early | 1 | 2 | 10
    False | 1 | 0 | 1
    True  | 1 | 2 | 0
    """)

def test_pivot_counts_with_indices(table):
    t = table.copy()
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    test = t.pivot(2, 4)
    assert_equal(test, """
    early | 1 | 2 | 10
    False | 1 | 0 | 1
    True  | 1 | 2 | 0
    """)

def test_pivot_values(table):
    t = table.copy()
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    t['exists'] = 2
    summed = t.pivot('points', 'early', 'exists', sum)
    assert_equal(summed, """
    early | 1 | 2 | 10
    False | 2 | 0 | 2
    True  | 2 | 4 | 0
    """)
    maxed = t.pivot('points', 'early', 'exists', max, -1)
    assert_equal(maxed, """
    early | 1 | 2  | 10
    False | 2 | -1 | 2
    True  | 2 | 2  | -1
    """)


def test_pivot_multiple_rows(table):
    t = table.copy()
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    t['late'] = t['letter'] > 'c'
    t['exists'] = 1
    test = t.pivot('points', ['early', 'late'], 'exists', sum)
    assert_equal(test, """
    early | late  | 1 | 2 | 10
    False | True  | 1 | 0 | 1
    True  | False | 1 | 2 | 0
    """)


def test_pivot_sum(table):
    t = table.copy()
    t['totals'] = t['points'] * t['count']
    t.append(('e', 12, 1, 12))
    t['early'] = t['letter'] < 'd'
    t['exists'] = 1
    test = t.pivot('points', 'early', 'exists', sum)
    assert_equal(test, """
    early | 1 | 2 | 10
    False | 1 | 0 | 1
    True  | 1 | 2 | 0
    """)


def test_apply(table):
    t = table.copy()
    assert_array_equal(t.apply(lambda x, y: x * y, 'count', 'points'),
                       np.array([9, 6, 6, 10]))
    assert_array_equal(t.apply(lambda x: x * x, 'points'),
                       np.array([1, 4, 4, 100]))
    assert_array_equal(t.apply(lambda row: row.item('count') * 2),
                       np.array([18, 6, 6, 2]))
    with(pytest.raises(ValueError)):
        t.apply(lambda x, y: x + y, 'count', 'score')
    with(pytest.warns(FutureWarning)):
        assert_array_equal(t.apply(lambda x: x * x, ['points']),
                           np.array([1, 4, 4, 100]))

    # Deprecated behavior
    assert_array_equal(t.apply(lambda x, y: x * y, 'count', 'points'),
                       np.array([9, 6, 6, 10]))

def test_first(table):
    t = table
    t['totals'] = t['points'] * t['count']
    assert_equal(t, """
    letter | count | points | totals
    a      | 9     | 1      | 9
    b      | 3     | 2      | 6
    c      | 3     | 2      | 6
    z      | 1     | 10     | 10
    """)
    assert_equal(t.first(1), 9)
    assert_equal(t.first("points"), 1)

def test_last(table):
    t = table
    t['totals'] = t['points'] * t['count']
    assert_equal(t, """
    letter | count | points | totals
    a      | 9     | 1      | 9
    b      | 3     | 2      | 6
    c      | 3     | 2      | 6
    z      | 1     | 10     | 10
    """)
    assert_equal(t.last(1), 1)
    assert_equal(t.last("points"), 10)


########
# Init #
########


def test_tuples(table, table2):
    """Tests that different-sized tuples are allowed."""
    t = table
    u = table2
    different = [((5, 1), (1, 2, 2, 10)), ('short', 'long')]
    t = Table().with_columns('tuple', different[0], 'size', different[1])
    assert_equal(t, """
    tuple         | size
    (5, 1)        | short
    (1, 2, 2, 10) | long
    """)
    same = [((5, 4, 3, 1), (1, 2, 2, 10)), ('long', 'long')]
    u = Table().with_columns('tuple', same[0], 'size', same[1])
    assert_equal(u, """
    tuple         | size
    [5 4 3 1]     | long
    [ 1  2  2 10] | long
    """)


def test_keys_and_values():
    """Tests that a table can be constructed from keys and values."""
    d = {1: 2, 3: 4}
    t = Table().with_columns('keys', d.keys(), 'values', d.values())
    assert_equal(t, """
    keys | values
    1    | 2
    3    | 4
    """)


##########
# Modify #
##########


def test_move_to_start(table):
    assert table.labels == ('letter', 'count', 'points')
    table.move_to_start('points')
    assert table.labels == ('points', 'letter', 'count')


def test_move_to_end(table):
    assert table.labels == ('letter', 'count', 'points')
    table.move_to_end('letter')
    assert table.labels == ('count', 'points', 'letter')

def test_move_to_end_start_int_labels(table):
    assert table.labels == ('letter', 'count', 'points')
    table.move_to_start(2)
    assert table.labels == ('points', 'letter', 'count')
    table.move_to_end(1)
    assert table.labels == ('points', 'count', 'letter')


def test_append_row(table):
    row = ['g', 2, 2]
    table.append(row)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    g      | 2     | 2
    """)

def test_append_none(table):
    row = [[], None, "", 0]
    for i in row:
        table.append(i)
        assert_equal(table, """
                     letter | count | points
                     a      | 9     | 1
                     b      | 3     | 2
                     c      | 3     | 2
                     z      | 1     | 10
                     """)

def test_append_row_by_array(table):
    row = np.array(['g', 2, 2])
    table.append(row)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    g      | 2     | 2
    """)


def test_append_row_different_num_cols(table):
    """Makes sure that any incoming row must have the same amount of columns as the table."""
    row = "abcd"
    with(pytest.raises(Exception)):
        table.append(row)

    row = ["e", 2, 4, 6]
    with(pytest.raises(Exception)):
        table.append(row)


def test_append_column(table):
    column_1 = [10, 20, 30, 40]
    column_2 = 'hello'
    table.append_column('new_col1', column_1)
    assert_equal(table, """
    letter | count | points | new_col1
    a      | 9     | 1      | 10
    b      | 3     | 2      | 20
    c      | 3     | 2      | 30
    z      | 1     | 10     | 40
    """)
    new_table = table.append_column('new_col2', column_2)
    assert_equal(table, """
    letter | count | points | new_col1 | new_col2
    a      | 9     | 1      | 10       | hello
    b      | 3     | 2      | 20       | hello
    c      | 3     | 2      | 30       | hello
    z      | 1     | 10     | 40       | hello
    """)
    assert_equal(new_table, """
    letter | count | points | new_col1 | new_col2
    a      | 9     | 1      | 10       | hello
    b      | 3     | 2      | 20       | hello
    c      | 3     | 2      | 30       | hello
    z      | 1     | 10     | 40       | hello
    """)

    with(pytest.raises(ValueError)):
        table.append_column('bad_col', [1, 2])
    with(pytest.raises(ValueError)):
        table.append_column(0, [1, 2, 3, 4])

def test_append_column_with_formatter(table):
    column_1 = [10, 20, 30, 40]
    column_2 = 'hello'
    table.append_column('new_col1', column_1, CurrencyFormatter)
    assert_equal(table, """
    letter | count | points | new_col1
    a      | 9     | 1      | $10
    b      | 3     | 2      | $20
    c      | 3     | 2      | $30
    z      | 1     | 10     | $40
    """)
    table.append_column('new_col2', column_2)
    assert_equal(table, """
    letter | count | points | new_col1  | new_col2
    a      | 9     | 1      | $10       | hello
    b      | 3     | 2      | $20       | hello
    c      | 3     | 2      | $30       | hello
    z      | 1     | 10     | $40       | hello
    """)

def test_with_column(table):
    column_1 = [10, 20, 30, 40]
    column_2 = 'hello'
    table2 = table.with_column('new_col1', column_1)
    table3 = table2.with_column('new_col2', column_2)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)
    assert_equal(table2, """
    letter | count | points | new_col1
    a      | 9     | 1      | 10
    b      | 3     | 2      | 20
    c      | 3     | 2      | 30
    z      | 1     | 10     | 40
    """)
    assert_equal(table3, """
    letter | count | points | new_col1 | new_col2
    a      | 9     | 1      | 10       | hello
    b      | 3     | 2      | 20       | hello
    c      | 3     | 2      | 30       | hello
    z      | 1     | 10     | 40       | hello
    """)

    with(pytest.raises(ValueError)):
        table.append_column('bad_col', [1, 2])
    with(pytest.raises(ValueError)):
        table.append_column(0, [1, 2, 3, 4])
def test_with_column_with_formatter(table):
    column_1 = [10, 20, 30, 40]
    column_2 = 'hello'
    table2 = table.with_column('new_col1', column_1, CurrencyFormatter)
    table3 = table2.with_column('new_col2', column_2)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)
    assert_equal(table2, """
    letter | count | points | new_col1
    a      | 9     | 1      | $10
    b      | 3     | 2      | $20
    c      | 3     | 2      | $30
    z      | 1     | 10     | $40
    """)
    assert_equal(table3, """
    letter | count | points | new_col1  | new_col2
    a      | 9     | 1      | $10       | hello
    b      | 3     | 2      | $20       | hello
    c      | 3     | 2      | $30       | hello
    z      | 1     | 10     | $40       | hello
    """)

def test_with_columns():
    players = Table().with_columns('player_id', make_array(110234, 110235), 'wOBA', make_array(.354, .236))
    assert_equal(players, """
    player_id | wOBA
    110234    | 0.354
    110235    | 0.236
    """)
    players = players.with_columns('salaries', 'N/A', 'season', 2016)
    assert_equal(players, """
    player_id | wOBA  | salaries | season
    110234    | 0.354 | N/A      | 2016
    110235    | 0.236 | N/A      | 2016
    """)
    salaries = Table().with_column('salary', make_array('$500,000', '$15,500,000'))
    players = players.with_columns('salaries', salaries.column('salary'), 'years', make_array(6, 1))
    assert_equal(players, """
    player_id | wOBA  | salaries    | season | years
    110234    | 0.354 | $500,000    | 2016   | 6
    110235    | 0.236 | $15,500,000 | 2016   | 1
    """)

def test_with_columns_exception_for_incorrect_usage():
    with(pytest.raises(TypeError)):
        Table.with_columns('player_id', make_array(110234, 110235), 'wOBA', make_array(.354, .236))

def test_with_columns_with_formats():
    players = Table().with_columns('player_id', make_array(110234, 110235), 'wOBA', make_array(.354, .236))
    assert_equal(players, """
    player_id  | wOBA
    110234    | 0.354
    110235    | 0.236
    """)
    players = players.with_columns('salaries', 'N/A', 'season', 2016)
    assert_equal(players, """
    player_id | wOBA  | salaries | season
    110234    | 0.354 | N/A      | 2016
    110235    | 0.236 | N/A      | 2016
    """)
    salaries = Table().with_column('salary', make_array(500000, 15500000))
    players2 = players.with_columns('salaries', salaries.column('salary'), 'years', make_array(6, 1), formatter=CurrencyFormatter)
    assert_equal(players2, """
    player_id | wOBA  | salaries     | season | years
    110234    | 0.354 | $500,000     | 2016   | $6
    110235    | 0.236 | $15,500,000  | 2016   | $1
    """)

    with(pytest.raises(Exception)):
        players3 = players.with_columns('salaries', salaries.column('salary'), make_array(7, 2), 'years', make_array(6, 1))

def test_with_columns(table):
    column_1 = [10, 20, 30, 40]
    column_2 = 'hello'
    table2 = table.with_columns(
        'new_col1', column_1,
        'new_col2', column_2)
    assert_equal(table2, """
    letter | count | points | new_col1 | new_col2
    a      | 9     | 1      | 10       | hello
    b      | 3     | 2      | 20       | hello
    c      | 3     | 2      | 30       | hello
    z      | 1     | 10     | 40       | hello
    """)

def test_append_table(table):
    table.append(table)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)


def test_append_different_table(table, table2):
    u = table2
    with pytest.raises(ValueError):
        table.append(u)


def test_append_different_order(table, table3):
    """Tests append with same columns, diff order"""
    table.append(table3)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    x      | 0     | 3
    y      | 54    | 10
    z      | 5     | 24
    """)


def test_relabel():
    table = Table().with_columns('points', (1, 2, 3), 'id', (12345, 123, 5123))
    table.relabel('id', 'todo')
    assert_equal(table, """
    points | todo
    1      | 12345
    2      | 123
    3      | 5123
    """)
    table.relabel(1, 'yolo')
    assert_equal(table, """
    points | yolo
    1      | 12345
    2      | 123
    3      | 5123
    """)
    table.relabel(['points', 'yolo'], ['red', 'blue'])
    assert_equal(table, """
    red    | blue
    1      | 12345
    2      | 123
    3      | 5123
    """)
    with(pytest.raises(ValueError)):
        table.relabel(['red', 'blue'], ['magenta', 'cyan', 'yellow'])
    with(pytest.raises(ValueError)):
        table.relabel(['red', 'green'], ['magenta', 'yellow'])

def test_relabel_with_chars(table):
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)
    table.relabel('points', 'minions')
    assert_equal(table, """
    letter | count | minions
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)

def test_relabeled(table):
    table2 = table.relabeled('points', 'minions')
    assert_equal(table2, """
    letter | count | minions
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)
    table3 = table.relabeled(['count', 'points'], ['ducks', 'ducklings'])
    assert_equal(table3, """
    letter | ducks | ducklings
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)

def test_relabeled_formatted(table):
    table.set_format('points', NumberFormatter)
    table2 = table.relabeled('points', 'very long label')
    assert_equal(table2, """
    letter | count | very long label
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)

def test_bin(table):
    binned = table.bin('count')
    assert_equal(binned.take(np.arange(5)), """
    bin  | count count
    1    | 1
    1.8  | 0
    2.6  | 2
    3.4  | 0
    4.2  | 0
    """)
    binned = table.select([1, 2]).bin(bins=4)
    assert_equal(binned, """
    bin  | count count | points count
    1    | 3           | 3
    3.25 | 0           | 0
    5.5  | 0           | 0
    7.75 | 1           | 1
    10   | 0           | 0
    """)
    binned = table.bin('points', bins=[-1, 1, 3, 5, 7, 9, 11, 13])
    assert_equal(binned, """
    bin  | points count
    -1   | 0
    1    | 3
    3    | 0
    5    | 0
    7    | 0
    9    | 1
    11   | 0
    13   | 0
    """)

def test_remove_multiple(table):
    table.remove([1, 3])
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    c      | 3     | 2
    """)

def test_remove_single(table):
    table.remove(1)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    c      | 3     | 2
    z      | 1     | 10
    """)

def test_remove_none(table):
    num_rows_before = table.num_rows
    assert table.remove([]) is None
    assert table.num_rows == num_rows_before

def test_remove_zeroth_row(table):
    table.remove(0)
    assert_equal(table, """
    letter | count | points
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)

def test_remove_num_rows(table):
    num_rows_before = table.num_rows
    table.remove(0)
    assert table.num_rows + 1 == num_rows_before

def test_remove_column_type(table):
    original_col = table.column("count")
    assert type(original_col) == np.ndarray
    table.remove(0)
    assert type(table.column("count")) == type(original_col)



##########
# Create #
##########


def test_empty():
    t = Table(['letter', 'count', 'points'])
    assert_equal(t, """
    letter | count | points
    """)

def test_empty_without_labels():
    t = Table()
    assert_equal(t, '')


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_from_rows():
    letters = [('a', 9, 1), ('b', 3, 2), ('c', 3, 2), ('z', 1, 10)]
    t = Table().from_rows(letters, ['letter', 'count', 'points'])
    assert_equal(t, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)


def test_from_records():
    letters = [
        {'letter': 'a',
         'count': 9,
         'points': 1,
        },
        {'letter': 'b',
         'count': 3,
         'points': 2,
        },
        {'letter': 'c',
         'count': 3,
         'points': 2,
        },
        {'letter': 'z',
         'count': 1,
         'points': 10,
        },
    ]
    t = Table.from_records(letters)
    assert_equal(t.select(['letter', 'count', 'points']), """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_from_columns_dict():
    columns_dict = {
        'letter': ['a', 'b', 'c', 'z'],
        'count': [9, 3, 3, 1],
        'points': [1, 2, 2, 10]
    }
    t = Table.from_columns_dict(columns_dict)
    assert_equal(t.select(['letter', 'count', 'points']), """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)


#############
# Transform #
#############


def test_group_by_tuples():
    tuples = [((5, 1), (1, 2, 2, 10), (1, 2, 2, 10)), (3, 3, 1)]
    t = Table().with_columns('tuples', tuples[0], 'ints', tuples[1])
    assert_equal(t, """
    tuples        | ints
    (5, 1)        | 3
    (1, 2, 2, 10) | 3
    (1, 2, 2, 10) | 1
    """)
    table = t.group('tuples', lambda s: s)
    assert_equal(table, """
    tuples        | ints
    (1, 2, 2, 10) | [3 1]
    (5, 1)        | [3]
    """)

def test_group_no_new_column(table):
    table.group(table.columns[1])
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)

def test_group_using_groups(table):
    table.groups(1)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)

def test_stack(table):
    test = table.stack(key='letter')
    assert_equal(test, """
    letter | column | value
    a      | count  | 9
    a      | points | 1
    b      | count  | 3
    b      | points | 2
    c      | count  | 3
    c      | points | 2
    z      | count  | 1
    z      | points | 10
    """)


def test_stack_restrict_columns(table):
    test = table.stack('letter', ['count'])
    assert_equal(test, """
    letter | column | value
    a      | count  | 9
    b      | count  | 3
    c      | count  | 3
    z      | count  | 1
    """)


def test_join_basic(table, table2):
    table['totals'] = table['points'] * table['count']
    test = table.join('points', table2)
    assert_equal(test, """
    points | letter | count | totals | names
    1      | a      | 9     | 9      | one
    2      | b      | 3     | 6      | two
    2      | c      | 3     | 6      | two
    """)


def test_join_with_booleans(table, table2):
    table['totals'] = table['points'] * table['count']
    table['points'] = table['points'] > 1
    table2['points'] = table2['points'] > 1

    assert_equal(table, """
    letter | count | points | totals
    a      | 9     | False  | 9
    b      | 3     | True   | 6
    c      | 3     | True   | 6
    z      | 1     | True   | 10
    """)

    assert_equal(table2, """
    points | names
    False  | one
    True   | two
    True   | three
    """)

    test = table.join('points', table2)
    assert_equal(test, """
    points | letter | count | totals | names
    False  | a      | 9     | 9      | one
    True   | b      | 3     | 6      | two
    True   | b      | 3     | 6      | three
    True   | c      | 3     | 6      | two
    True   | c      | 3     | 6      | three
    True   | z      | 1     | 10     | two
    True   | z      | 1     | 10     | three
    """)


def test_join_with_self(table):
    test = table.join('count', table)
    assert_equal(test, """
    count | letter | points | letter_2 | points_2
    1     | z      | 10     | z        | 10
    3     | b      | 2      | b        | 2
    3     | b      | 2      | c        | 2
    3     | c      | 2      | b        | 2
    3     | c      | 2      | c        | 2
    9     | a      | 1      | a        | 1
    """)


def test_join_with_strings(table):
    test = table.join('letter', table)
    assert_equal(test, """
    letter | count | points | count_2 | points_2
    a      | 9     | 1      | 9       | 1
    b      | 3     | 2      | 3       | 2
    c      | 3     | 2      | 3       | 2
    z      | 1     | 10     | 1       | 10
    """)

def test_join_with_same_formats(table):
    test = table.copy().set_format("points", CurrencyFormatter(int_to_float=True))
    assert_equal(test, """
    letter | count | points
    a      | 9     | $1.00
    b      | 3     | $2.00
    c      | 3     | $2.00
    z      | 1     | $10.00
    """)
    test_joined = test.join("points", test)
    assert_equal(test_joined, """
    points | letter | count | letter_2  | count_2
    $1.00  | a      | 9     | a         | 9
    $2.00  | b      | 3     | b         | 3
    $2.00  | b      | 3     | c         | 3
    $2.00  | c      | 3     | b         | 3
    $2.00  | c      | 3     | c         | 3
    $10.00 | z      | 1     | z         | 1
    """)

def test_join_with_one_formatted(table):
    test = table.copy().set_format("points", CurrencyFormatter(int_to_float=True))
    assert_equal(test, """
    letter | count | points
    a      | 9     | $1.00
    b      | 3     | $2.00
    c      | 3     | $2.00
    z      | 1     | $10.00
    """)
    test_joined = test.join("points", table)
    assert_equal(test_joined, """
    points | letter | count | letter_2  | count_2
    $1.00  | a      | 9     | a         | 9
    $2.00  | b      | 3     | b         | 3
    $2.00  | b      | 3     | c         | 3
    $2.00  | c      | 3     | b         | 3
    $2.00  | c      | 3     | c         | 3
    $10.00 | z      | 1     | z         | 1
    """)

def test_join_with_two_labels_one_format(table):
    test = table.copy().set_format("points", CurrencyFormatter(int_to_float=True))
    assert_equal(test, """
    letter | count | points
    a      | 9     | $1.00
    b      | 3     | $2.00
    c      | 3     | $2.00
    z      | 1     | $10.00
    """)
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)
    test2 = test.copy()
    table2 = table.copy()
    test_joined = test.join("letter", table)
    assert_equal(test_joined, """
    letter | count | points     | count_2 | points_2
    a      | 9     | $1.00      | 9       | 1
    b      | 3     | $2.00      | 3       | 2
    c      | 3     | $2.00      | 3       | 2
    z      | 1     | $10.00     | 1       | 10
    """)

    test_joined2 = table2.join("letter", test2)
    assert_equal(test_joined2, """
    letter | count | points | count_2 | points_2
    a      | 9     | 1      | 9       | $1.00
    b      | 3     | 2      | 3       | $2.00
    c      | 3     | 2      | 3       | $2.00
    z      | 1     | 10     | 1       | $10.00
    """)

def test_join_with_column_already_duplicated_simple(table, scrabble_table2):
    joined_forwards = table.join("letter", scrabble_table2)
    joined_backwards = scrabble_table2.join("letter", table)

    assert_equal(joined_forwards, """
    letter | count | points | count_3 | count_2 | pointsplus1
    a      | 9     | 1      | 9       | 9       | 2
    b      | 3     | 2      | 3       | 3       | 3
    c      | 3     | 2      | 3       | 3       | 3
    z      | 1     | 10     | 1       | 1       | 11
    """)

    assert_equal(joined_backwards, """
    letter | count | count_2 | pointsplus1 | count_3 | points
    a      | 9     | 9       | 2           | 9       | 1
    b      | 3     | 3       | 3           | 3       | 2
    c      | 3     | 3       | 3           | 3       | 2
    z      | 1     | 1       | 11          | 1       | 10
    """)


def test_join_with_column_already_duplicated_thorough(table, scrabble_table2):
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1
    b      | 3     | 2
    c      | 3     | 2
    z      | 1     | 10
    """)
    assert_equal(scrabble_table2, """
    letter | count | count_2 | pointsplus1
    a      | 9     | 9      | 2
    b      | 3     | 3      | 3
    c      | 3     | 3      | 3
    z      | 1     | 1      | 11
    """)

    joined_forwards = table.join("letter", scrabble_table2)
    joined_backwards = scrabble_table2.join("letter", table)

    # Test that the joined tables have the same labels
    assert_array_equal(list(sorted(joined_backwards.labels)), list(sorted(joined_forwards.labels)))

    # Check for each label that the same contents are in the column
    for col in joined_backwards.labels:
        assert_array_equal(joined_forwards.column(col), joined_backwards.column(col))

    # Create new test case
    table_copy = table.with_columns("pointsplus1", scrabble_table2.column("pointsplus1"), 
        "pointsplus1_2", scrabble_table2.column("pointsplus1") + 1)
    joined_forwards = table_copy.join("letter", scrabble_table2)
    joined_backwards = scrabble_table2.join("letter", table_copy)

    # Test that the joined tables have the same labels
    assert_array_equal(len(list(sorted(joined_backwards.labels))), len(list(sorted(joined_forwards.labels))))

    # Check for each label that the same contents are in the column
    for col in joined_backwards.labels:
        assert_array_equal(joined_forwards.column(col), joined_backwards.column(col))

def test_join_one_list_with_one_label(table, table4):
    table['totals'] = table['points'] * table['count']
    test = table.join(['letter'], table4.drop('count', 'different label'))
    assert_equal(test, """
    letter | count | points | totals | name
    a      | 9     | 1      | 9      | Gamma
    a      | 9     | 1      | 9      | Beta
    b      | 3     | 2      | 6      | Delta
    c      | 3     | 2      | 6      | Epsilon
    """)

def test_join_two_lists_same_label(table, table4):
    table['totals'] = table['points'] * table['count']
    test = table.join(['letter'], table4.drop('count', 'different label'), ['letter'])
    assert_equal(test, """
    letter | count | points | totals | name
    a      | 9     | 1      | 9      | Gamma
    a      | 9     | 1      | 9      | Beta
    b      | 3     | 2      | 6      | Delta
    c      | 3     | 2      | 6      | Epsilon
    """)

def test_join_two_lists_different_labels(table, table4):
	# also checks for multiple matches on one side
    table['totals'] = table['points'] * table['count']
    test = table.join(['points'], table4.drop('letter', 'count'), ['different label'])
    assert_equal(test, """
    points | letter | count | totals | name
    1      | a      | 9     | 9      | Gamma
    1      | a      | 9     | 9      | Alpha
    1      | a      | 9     | 9      | Beta
    2      | b      | 3     | 6      | Epsilon
    2      | c      | 3     | 6      | Epsilon
    """)

def test_join_two_lists_2_columns(table, table4):
    table['totals'] = table['points'] * table['count']
    test = table.join(['letter', 'points'], table4, ['letter', 'different label'])
    assert_equal(test, """
    letter | points | count | totals | count_2 | name
    a      | 1      | 9     | 9      | 9       | Gamma
    a      | 1      | 9     | 9      | 9       | Beta
    c      | 2      | 3     | 6      | 2       | Epsilon
    """)

def test_join_two_lists_3_columns(table, table4):
    table['totals'] = table['points'] * table['count']
    test = table.join(['letter', 'count', 'points'], table4, ['letter', 'count', 'different label'])
    assert_equal(test, """
    letter | count | points | totals | name
    a      | 9     | 1      | 9      | Gamma
    a      | 9     | 1      | 9      | Beta
    """)

def test_join_conflicting_column_names(table, table4):
    table['totals'] = table['points'] * table['count']
    test = table.join(['letter'], table4)
    assert_equal(test, """
    letter | count | points | totals | count_2 | different label | name
    a      | 9     | 1      | 9      | 9       | 1               | Gamma
    a      | 9     | 1      | 9      | 9       | 1               | Beta
    b      | 3     | 2      | 6      | 3       | 4               | Delta
    c      | 3     | 2      | 6      | 2       | 2               | Epsilon
    """)

def test_join_no_rows():
    t1 = Table(['letter'])
    t2 = Table(['letter'])
    t2.append(['b'])
    t3 = t1.join('letter', t2)
    assert t3 == None
    t4 = t2.join('letter', t1)
    assert t4 == None
    
def test_percentile(numbers_table):
    assert_equal(numbers_table.percentile(76), """
    count | points
    9     | 10
    """)

    assert_equal(numbers_table.percentile(75), """
    count | points
    3     | 2
    """)

def test_pivot_bin(categories_table):
    assert_equal(categories_table.pivot_bin('key', 'val', bins=[0, 1, 2, 3]), """
    bin  | a    | b
    0    | 0    | 0
    1    | 2    | 1
    2    | 1    | 2
    3    | 0    | 0
    """)

def test_move_column(table):
    assert table.labels == ('letter', 'count', 'points')
    table = table.move_column("letter", 1)
    assert table.labels == ('count', 'letter', 'points')
    table = table.move_column(2, 1)
    assert table.labels == ('count', 'points', 'letter')


def test_join_with_empty_self(table2):
    empty_table = Table().with_columns(['letter', []])
    result = empty_table.join('letter', table2)
    assert result is None, "Expected None when joining with an empty self table"

def test_join_with_empty_other(table):
    empty_table = Table().with_columns(['points', []])
    result = table.join('points', empty_table)
    assert result is None, "Expected None when joining with an empty other table"

def test_join_with_other_label(table, table2):
    result = table.join('points', table2, 'points')
    assert_equal(result, """
    points | letter | count | names
    1      | a      | 9     | one
    2      | b      | 3     | two
    2      | c      | 3     | two
    """)

def test_join_without_other_label(table, table2):
    result = table.join('points', table2)
    assert_equal(result, """
    points | letter | count | names
    1      | a      | 9     | one
    2      | b      | 3     | two
    2      | c      | 3     | two
    """)

def test_join_with_empty_table(table, empty_table):
    result = empty_table._join('points', table)
    assert result is None

    result = table._join('points', empty_table)
    assert result is None

def test_join_without_other_label(table, table2):
    result = table._join('points', table2)
    assert result is not None






##################
# Export/Display #
##################


def test_format_function(table):
    """Test that formatting can be applied by a function."""
    table = table.copy().set_format('points', lambda v: float(v))
    assert_equal(table, """
    letter | count | points
    a      | 9     | 1.0
    b      | 3     | 2.0
    c      | 3     | 2.0
    z      | 1     | 10.0
    """)


def test_sample_basic(table):
    """Tests that sample doesn't break"""
    table.sample(table.num_rows)


def test_sample_basic_modk(table):
    """Tests that sample k<n doesn't break"""
    table.sample(2)


def test_sample_wrepl_basic(table):
    """Tests that sample with_replacement=True doesn't break"""
    table.sample(table.num_rows, with_replacement=True)


def test_sample_wwgts_basic(table):
    """Tests that sample with weights doesn't break"""
    table.sample(table.num_rows, weights=[1/4]*4)


def test_sample_weights_ne1(table):
    """Tests that a series of weights with total != 1 is not accepted"""
    with pytest.raises(ValueError):
        table.sample(table.num_rows, weights=[1/4, 1/4, 1/4, 1/6])

    with pytest.raises(ValueError):
        table.sample(table.num_rows, weights=[1/4, 1/4, 1/4, 1/2])


def test_sample_weights_worepl(table):
    """Tests that with_replacement flag works - ensures with_replacement=False
    works by asserting unique rows for each iteration
    1000: ~3.90s
    2000: ~7.04s
    4000: ~13.2s
    10000: ~33.18s
    """
    iterations, i = 100,  0
    while i < iterations:
        u = table.sample(table.num_rows, with_replacement=False)
        assert len(set(u.rows)) == len(u.rows)
        i += 1


def test_sample_weights_with_none_k(table):
    """Tests that with_replacement flag works - ensures with_replacement=False
    works by asserting unique rows for each iteration, with k=None default
    """
    iterations, i = 100,  0
    while i < iterations:
        u = table.sample(with_replacement=False)
        assert len(set(u.rows)) == len(u.rows)
        i += 1

def test_shuffle_basic(table):
    """Tests that shuffle doesn't break"""
    table.shuffle()

def test_shuffle_correct_n(table):
    """Tests that shuffle returns the correct number of rows"""
    assert table.num_rows == table.shuffle().num_rows

def test_shuffle_all_rows_appear(table):
    """
    Tests that all rows appear in shuffled table / there are no 
    duplicated rows
    """
    assert set(table.column("letter")) == set(table.shuffle().column("letter"))

def test_shuffle_different_order(table):
    """
    Tests that all rows do not always appear in the same order
    as they were in the original table
    """
    # This is more or less a regression test - you could get very 
    # unlucky and have no change in the order of the rows by complete 
    # accident. However, this is highly unlikely to happen.
    
    original_order = table.column("letter")
    for _ in range(10):
        if not np.array_equal(table.shuffle().column("letter"), original_order):
            assert True
            return
    assert False

def test_split_basic(table):
    """Test that table.split works."""
    table.split(3)

def test_split_lengths(table):
    """Test that table.split outputs tables with the right number of rows."""
    sampled, rest = table.split(3)
    assert sampled.num_rows == 3
    assert rest.num_rows == table.num_rows - 3

def test_split_k_vals(table):
    """Test that invalid k values for table.split raises an error."""
    with pytest.raises(ValueError):
        table.split(0)
    with pytest.raises(ValueError):
        table.split(table.num_rows)

def test_split_table_labels(table):
    sampled, rest = table.split(3)
    assert sampled.labels == table.labels
    assert rest.labels == table.labels

###############
# Inheritance #
###############


class SubTable(Table):
    """Test inheritance through tables."""
    def __init__(self, *args):
        Table.__init__(self, *args)
        self.extra = "read all about it"

def test_subtable():
    """Tests that derived classes retain type through super methods."""
    st = SubTable().with_columns([("num", [1,2,3]),
                                  ('col', ['red', 'blu', 'red']),
                                  ("num2", [5,3,7])])
    assert(type(st) == SubTable)
    assert(type(st.select('col')) == type(st))
    assert(type(st.pivot_bin('col', 'num')) == type(st))
    assert(type(st.stats()) == type(st))
    assert(type(st.bin('num', bins=3)) == type(st))
    assert(type(st.copy()) == type(st))


#############
# Visualize #
#############

def test_compute_shading():

    h = np.arange(5)
    def single_test(start, end, x_expected, h_expected, w_expected):
        h_this = h[:]
        bins = np.arange(0, 51, 10)
        x_actual, h_actual, w_actual = tables._compute_shading(h_this, bins, start, end)
        assert np.allclose(x_actual, x_expected)
        assert np.allclose(h_actual, h_expected)
        assert np.allclose(w_actual, w_expected)

    # Given the entire range, create all bins
    single_test(0, 50,
                np.arange(0, 41, 10), h,
                10 * np.ones(5))

    # Test subrange that aligns with bin start/ends
    single_test(0, 30,
                np.arange(0, 21, 10), h[:3],
                np.array([10, 10, 10]))

    # Test subrange that doesn't align with bin start/ends
    single_test(5, 45,
                np.array([5, 10, 20, 30, 40]), h,
                np.array([5, 10, 10, 10, 5]))

    # Test subrange that crosses only one bin boundary
    single_test(5, 15,
                np.array([5, 10]), h[:2],
                np.array([5, 5]))

    # Test subrange within a single bin
    single_test(23, 28,
                np.array([23]), h[2:3],
                np.array([5]))

    # Test subrange starting before data range
    single_test(-32, 28,
                np.array([0, 10, 20]), h[:3],
                np.array([10, 10, 8]))

    # Test subrange ending after data range
    single_test(23, 200,
                np.array([23, 30, 40]), h[2:],
                np.array([7, 10, 10]))

    # Test subrange with exactly first bin
    single_test(0, 10,
                np.array([0]), h[0:1],
                np.array([10]))

    # Test subrange with exactly last bin
    single_test(40, 50,
                np.array([40]), h[4:5],
                np.array([10]))


def test_scatter(numbers_table):
    """Tests that Table.scatter doesn't raise an error when the table doesn't
    contains non-numerical values. Not working right now because of TKinter
    issues on Travis.

    TODO(sam): Fix Travis so this runs
    """

    # numbers_table.scatter('count')

def test_scatter_error(table):
    """Tests that Table.scatter raises an error when the table contains
    non-numerical values."""

    with pytest.raises(ValueError):
        table.scatter('nonexistentlabel')

def test_hist_of_counts(numbers_table):
    """Tests that hist_of_counts works OK for good bins.
    Probably won't work now because of TKinter issues on Travis.

    TODO(sam): Fix Travis so this runs
    """
    # # None of these should raise errors

    # Test integers
    numbers_table.hist_of_counts('count', bins=np.arange(10))

    # Test floats without rounding error
    numbers_table.hist_of_counts('count', bins=np.arange(0, 10, 0.25))

    # Test floats with rounding error
    numbers_table.hist_of_counts('count', bins=np.arange(0, 10, 0.1))

    # Test very small floats
    numbers_table.hist_of_counts('count', bins=np.arange(1e-20, 2e-20, 1e-21))
    pass

def test_hist_of_counts_raises_errors(numbers_table):
    """Tests that hist_of_counts raises errors for uneven bins
    """

    # Integers
    with pytest.raises(ValueError):
        numbers_table.hist_of_counts('count', bins=np.array([0, 1, 5, 10]))

    # floats
    with pytest.raises(ValueError):
        numbers_table.hist_of_counts('count', bins=np.array([0., 0.25, 1., 4.]))

    # Very small floats
    with pytest.raises(ValueError):
        numbers_table.hist_of_counts('count', bins=np.array([1e-20, 2e-20, 5e-20]))

def test_df_roundtrip(table):
    df = table.to_df()
    assert isinstance(df, pd.DataFrame)

    t = Table.from_df(df)

    for (c0, c1) in zip(t.columns, table.columns):
        assert_equal(c0, c1)

def test_from_df_with_index(table):
    from bokeh.sampledata.autompg import autompg as df
    assert isinstance(df, pd.DataFrame)
    test_no_index = Table.from_df(df)
    assert not ("index" in test_no_index.labels)
    test_with_index = Table.from_df(df, keep_index=True)
    assert "index" in test_with_index.labels
    assert test_with_index.num_columns == (test_no_index.num_columns + 1)

def test_array_roundtrip(table):
    arr = table.to_array()
    assert isinstance(arr, np.ndarray)

    t = Table.from_array(arr)
    for (c0, c1) in zip(t.columns, table.columns):
        assert_equal(c0, c1)

    # Now test using the ndarray attribute
    arr = table.values
    assert isinstance(arr, np.ndarray)

    t = table.with_columns([(nm, vals)
                              for nm, vals in zip(table.labels, arr.T)])
    for (c0, c1) in zip(t.columns, table.columns):
        assert_equal(c0, c1)


def test_url_parse():
    """Test that Tables parses URLs correctly"""
    with pytest.raises(ValueError):
        url = 'https://data8.berkeley.edu/something/something/dark/side'
        Table.read_table(url)

def test_read_table():
    """Test that Tables reads a csv file."""
    assert isinstance(Table().read_table("tests/us-unemployment.csv"), Table)
    assert isinstance(Table().read_table("tests/us-unemployment-copy"), Table)
    assert isinstance(Table().read_table("tests/us-unemployment.txt"), Table)
    assert isinstance(Table().read_table("https://raw.githubusercontent.com/data-8/textbook/main/assets/data/deflategate.csv"), Table)

    """Test that Tables reads csv content from a buffer."""
    buffer = BytesIO()
    with open('tests/us-unemployment.csv', 'rb') as f:
        buffer.write(f.read())
        buffer.seek(0)
        assert isinstance(Table().read_table(buffer), Table)

def test_no_records():
    """Test that Table returns nothing on an empty record"""
    empty_table = Table()
    records_empty_table = Table().from_records([])
    assert empty_table == records_empty_table

def test_column_index(table):
    """Test that Tables returns right index number"""
    index = table.column_index('letter')
    assert index == 0

def test_num_columns(table):
    """Test that Tables returns right number of columns"""
    number = table.num_columns
    assert number == 3
