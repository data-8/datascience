import re
import pytest
from numpy.testing import assert_array_equal
from datascience import *


#########
# Utils #
#########


@pytest.fixture(scope='function')
def table():
	"""Setup alphanumeric table"""
	letter, count, points = ['a', 'b', 'c', 'z'], [9, 3, 3, 1], [1, 2, 2, 10]
	return Table([letter, count, points], ['letter', 'count', 'points'])


@pytest.fixture(scope='function')
def table2():
	"""Setup second alphanumeric table"""
	points, names = (1, 2, 3), ('one', 'two', 'three')
	return Table([points, names], ['points', 'names'])


@pytest.fixture(scope='module')
def t():
	"""Create one table for entire module"""
	return table()


@pytest.fixture(scope='module')
def u():
	"""Setup second alphanumeric table"""
	return table2()


def assert_equal(string1, string2):
	string1, string2 = str(string1), str(string2)
	whitespace = re.compile('\s')
	purify = lambda s: whitespace.sub('', s)
	assert purify(string1) == purify(string2), string1 + " != " + string2


############
# Overview #
############


def test_basic(t):
	"""Tests that t works"""
	assert_equal(t, """\
	letter | count | points
	a      | 9     | 1
	b      | 3     | 2
	c      | 3     | 2
	z      | 1     | 10
	""")


def test_basic_points(t):
	assert_array_equal(t['points'], np.array([1, 2, 2, 10]))


def test_basic_rows(t):
	assert_equal(
		t.rows[2],
	     "Row(letter='c', count=3, points=2)")


def test_select(t):
	test = t.select(['points', 'count']).cumsum()
	assert_equal(test, """\
	points | count
	1      | 9
	3      | 12
	5      | 15
	15     | 16
	""")


def test_take(t):
	test = t.take([1, 2])
	assert_equal(test, """\
	letter | count | points
	b      | 3     | 2
	c      | 3     | 2
	""")


def test_stats(t):
	test = t.stats()
	assert_equal(test, """\
	statistic | letter | count | points
	min       | a      | 1     | 1
	max       | z      | 9     | 10
	median    |        | 3     | 2
	sum       |        | 16    | 15
	""")


def test_stats_with_numpy(t):
	test = t.stats([np.mean, np.std, np.var])
	assert_equal(test, """\
	statistic | letter | count | points
	mean      |        | 4     | 3.75
	std       |        | 3     | 3.63146
	var       |        | 9     | 13.1875""")


def test_where(t):
	test = t.where('points', 2)
	assert_equal(test, """\
	letter | count | points
	b      | 3     | 2
	c      | 3     | 2
	""")


def test_where_conditions(t):
	t['totals'] = t['points'] * t['count']
	test = t.where(t['totals'] > 8)
	assert_equal(test, """\
	letter | count | points | totals
	a      | 9     | 1      | 9
	z      | 1     | 10     | 10
	""")


def test_sort(t):
	test = t.sort('points')
	assert_equal(test, """\
	letter | count | points | totals
	a      | 9     | 1      | 9
	b      | 3     | 2      | 6
	c      | 3     | 2      | 6
	z      | 1     | 10     | 10
	""")


def test_sort_args(t):
	test = t.sort('points', decreasing=True, distinct=True)
	assert_equal(test, """\
	letter | count | points | totals
	z      | 1     | 10     | 10
	b      | 3     | 2      | 6
	a      | 9     | 1      | 9
	""")


def test_sort_syntax(t):
	test = t.sort(-t['totals'])
	assert_equal(test, """\
	letter | count | points | totals
	z      | 1     | 10     | 10
	a      | 9     | 1      | 9
	b      | 3     | 2      | 6
	c      | 3     | 2      | 6
	""")


def test_group(t):
	test = t.group('points')
	assert_equal(test, """\
	points | letter    | count | totals
	1      | ['a']     | [9]   | [9]
	2      | ['b' 'c'] | [3 3] | [6 6]
	10     | ['z']     | [1]   | [10]
	""")


def test_group_with_func(t):
	test = t.group('points', sum)
	assert_equal(test, """\
	points | letter sum | count sum | totals sum
	1      |            | 9         | 9
	2      |            | 6         | 12
	10     |            | 1         | 10
	""")


def test_groups(t):
	t = t.copy()
	t.append(('e', 12, 1, 12))
	t['early'] = t['letter'] < 'd'
	test = t.groups(['points', 'early'])
	assert_equal(test, """\
    points | early | letter    | count | totals
    1      | False | ['e']     | [12]  | [12]
    1      | True  | ['a']     | [9]   | [9]
    2      | True  | ['b' 'c'] | [3 3] | [6 6]
    10     | False | ['z']     | [1]   | [10]
	""")

def test_groups_collect(t):
	t = t.copy()
	t.append(('e', 12, 1, 12))
	t['early'] = t['letter'] < 'd'
	test = t.select(['points', 'early', 'count']).groups(['points', 'early'], sum)
	assert_equal(test, """\
    points | early | count sum
    1      | False | 12
    1      | True  | 9
    2      | True  | 6
    10     | False | 1
	""")

def test_join(t, u):
	"""Tests that join works, not destructive"""
	test = t.join('points', u)
	assert_equal(test, """\
	points | letter | count | totals | names
	1      | a      | 9     | 9      | one
	2      | b      | 3     | 6      | two
	""")
	assert_equal(u, """\
	points  | names
	1       | one
	2       | two
	3       | three
	""")
	assert_equal(t, """\
	letter | count | points | totals
	a      | 9     | 1      | 9
	b      | 3     | 2      | 6
	c      | 3     | 2      | 6
	z      | 1     | 10     | 10
	""")


def test_pivot(t):
	t = t.copy()
	t.append(('e', 12, 1, 12))
	t['early'] = t['letter'] < 'd'
	t['exists'] = 1
	test = t.pivot('points', 'early', 'exists')
	assert_equal(test, """\
    early | 1 exists | 2 exists | 10 exists
    False | [1]      | None     | [1]
    True  | [1]      | [1 1]    | None
	""")


def test_pivot_sum(t):
	t = t.copy()
	t.append(('e', 12, 1, 12))
	t['early'] = t['letter'] < 'd'
	t['exists'] = 1
	test = t.pivot('points', 'early', 'exists', sum)
	assert_equal(test, """\
	early | 1 exists | 2 exists | 10 exists
    False | 1        | 0        | 1
    True  | 1        | 2        | 0
	""")


########
# Init #
########


def test_tuples(t, u):
	"""Tests that different-sized tuples are allowed."""
	different = [((5, 1), (1, 2, 2, 10)), ('short', 'long')]
	t = Table(different, ['tuple', 'size'])
	assert_equal(t, """\
	tuple         | size
    (5, 1)        | short
    (1, 2, 2, 10) | long
	""")
	same = [((5, 4, 3, 1), (1, 2, 2, 10)), ('long', 'long')]
	u = Table(same, ['tuple', 'size'])
	assert_equal(u, """\
	tuple         | size
    [5 4 3 1]     | long
    [ 1  2  2 10] | long
	""")


def test_keys_and_values():
	"""Tests that a table can be constructed from keys and values."""
	d = {1: 2, 3: 4}
	t = Table([d.keys(), d.values()], ['keys', 'values'])
	assert_equal(t, """\
	keys | values
	1    | 2
	3    | 4
	""")


##########
# Modify #
##########


def test_move_to_start(table):
	assert table.column_labels == ('letter', 'count', 'points')
	table.move_to_start('points')
	assert table.column_labels == ('points', 'letter', 'count')


def test_move_to_end(table):
	assert table.column_labels == ('letter', 'count', 'points')
	table.move_to_end('letter')
	assert table.column_labels == ('count', 'points', 'letter')


def test_append_row(table):
	row = ['g', 2, 2]
	table.append(row)
	assert_equal(table, """\
	letter | count | points
	a      | 9     | 1
	b      | 3     | 2
	c      | 3     | 2
	z      | 1     | 10
	g      | 2     | 2
	""")


def test_append_table(table):
	table.append(table)
	assert_equal(table, """\
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


def test_append_bad_table(table, u):
	with pytest.raises(AssertionError):
		table.append(u)


def test_relabel():
	table = Table([(1, 2, 3), (12345, 123, 5123)], ['points', 'id'])
	table.relabel('id', 'yolo')
	assert_equal(table, """\
	points | yolo
	1      | 12345
	2      | 123
	3      | 5123
	""")


def test_relabel_with_chars(table):
	assert_equal(table, """\
	letter | count | points
	a      | 9     | 1
	b      | 3     | 2
	c      | 3     | 2
	z      | 1     | 10
	""")
	table.relabel('points', 'minions')
	assert_equal(table, """\
	letter | count | minions
	a      | 9     | 1
	b      | 3     | 2
	c      | 3     | 2
	z      | 1     | 10
	""")


##########
# Create #
##########


def test_from_rows():
	letters = [('a', 9, 1), ('b', 3, 2), ('c', 3, 2), ('z', 1, 10)]
	t = Table.from_rows(letters, ['letter', 'count', 'points'])
	assert_equal(t, """\
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
	t = Table(tuples, ['tuples', 'ints'])
	assert_equal(t, """\
	tuples        | ints
	(5, 1)        | 3
	(1, 2, 2, 10) | 3
	(1, 2, 2, 10) | 1
	""")
	table = t.group('tuples')
	assert_equal(table, """\
	tuples        | ints
	(1, 2, 2, 10) | [3 1]
	(5, 1)        | [3]
	""")


def test_group_no_new_column(table):
	table.group(table.columns[1])
	assert_equal(table, """\
	letter | count | points
	a      | 9     | 1
	b      | 3     | 2
	c      | 3     | 2
	z      | 1     | 10
	""")


def test_stack(table):
	test = table.stack(key='letter')
	assert_equal(test, """\
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
	test = table.stack(key='letter', column_labels=['count'])
	assert_equal(test, """\
	letter | column | value
	a      | count  | 9
	b      | count  | 3
	c      | count  | 3
	z      | count  | 1
	""")


def test_join_basic(table, table2):
	table['totals'] = table['points'] * table['count']
	test = table.join('points', table2)
	assert_equal(test, """\
	points | letter | count | totals | names
	1      | a      | 9     | 9      | one
	2      | b      | 3     | 6      | two
	""")


def test_join_with_booleans(table, table2):
	table['totals'] = table['points'] * table['count']
	table['points'] = table['points'] > 1
	table2['points'] = table2['points'] > 1

	test = table.join('points', table2)
	assert_equal(test, """\
	points | letter | count | totals | names
	False  | a      | 9     | 9      | one
	True   | b      | 3     | 6      | two
	""")


def test_join_with_self(table):
	test = table.join('count', table)
	assert_equal(test, """\
	count | letter | points | letter_2 | points_2
	1     | z      | 10     | z        | 10
	3     | b      | 2      | b        | 2
	9     | a      | 1      | a        | 1
	""")


def test_join_with_strings(table):
	test = table.join('letter', table)
	assert_equal(test, """\
	letter | count | points | count_2 | points_2
	a      | 9     | 1      | 9       | 1
	b      | 3     | 2      | 3       | 2
	c      | 3     | 2      | 3       | 2
	z      | 1     | 10     | 1       | 10
	""")


##################
# Export/Display #
##################



#############
# Visualize #
#############



###########
# Support #
###########
