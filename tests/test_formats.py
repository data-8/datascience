import doctest
import re
import datascience as ds
from datascience import formats


def assert_equal(string1, string2):
    string1, string2 = str(string1), str(string2)
    whitespace = re.compile('\s')
    purify = lambda s: whitespace.sub('', s)
    assert purify(string1) == purify(string2), "\n%s\n!=\n%s" % (string1, string2)


def test_doctests():
    results = doctest.testmod(formats, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.failed == 0


def test_default_format():
    fmt = ds.default_formatter.format_value
    assert_equal(fmt(1.23456789), '1.23457')
    assert_equal(fmt(123456789), '123456789')
    assert_equal(fmt(123456789**5), '28679718602997181072337614380936720482949')
    assert_equal(fmt(123.456789**5), '2.86797e+10')
    assert_equal(fmt(True), 'True')
    assert_equal(fmt(False), 'False')
    assert_equal(fmt('hello'), 'hello')
    assert_equal(fmt((1, 2)), '(1, 2)')


def test_number_format():
    for fmt in [ds.NumberFormatter(2), ds.NumberFormatter]:
        us = ['1,000', '12,000']
        vs = ['1,000', '12,000.346']
        t = ds.Table().with_columns('u', us, 'v', vs)
        t.set_format(['u', 'v'], fmt)
        assert_equal(t, """
        u      | v
        1,000  | 1,000.00
        12,000 | 12,000.35
        """)


def test_currency_format():
    vs = ['$60', '$162.5', '$1,625']
    t = ds.Table().with_columns('num', vs, 'str', vs)
    t.set_format('num', ds.CurrencyFormatter('$', int_to_float=True))
    assert_equal(t, """
    num       | str
    $60.00    | $60
    $162.50   | $162.5
    $1,625.00 | $1,625
    """)
    assert_equal(t.sort('num'), """
    num       | str
    $60.00    | $60
    $162.50   | $162.5
    $1,625.00 | $1,625
    """)
    assert_equal(t.sort('str'), """
    num       | str
    $1,625.00 | $1,625
    $162.50   | $162.5
    $60.00    | $60
    """)


def test_currency_format_int():
    t = ds.Table().with_column('money', [1, 2, 3])
    t.set_format(['money'], ds.CurrencyFormatter)
    assert_equal(t, """
    money
    $1
    $2
    $3
    """)


def test_date_format():
    vs = ['2015-07-01 22:39:44.900351']
    t = ds.Table().with_column('time', vs)
    t.set_format('time', ds.DateFormatter("%Y-%m-%d %H:%M:%S.%f"))
    assert isinstance(t['time'][0], float)


def test_percent_formatter():
    vs = [0.1, 0.11111, 0.199999, 10]
    t = ds.Table().with_column('percent', vs)
    t.set_format('percent', ds.PercentFormatter(1))
    assert_equal(t, """
    percent
    10.0%
    11.1%
    20.0%
    1000.0%
    """)

def test_distribution_formatter():
    counts = [9, 10, 18, 23]
    t = ds.Table().with_column('count', counts)
    t.set_format('count', ds.DistributionFormatter)
    assert_equal(t, """
    count
    15.00%
    16.67%
    30.00%
    38.33%
    """)
