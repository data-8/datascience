import collections.abc
import pytest
import re
import datascience as ds


def assert_equal(string1, string2):
    string1, string2 = str(string1), str(string2)
    whitespace = re.compile('\s')
    purify = lambda s: whitespace.sub('', s)
    assert purify(string1) == purify(string2), "\n%s\n!=\n%s" % (string1, string2)


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
        t = ds.Table([us, vs], ['u', 'v'])
        t.set_format(['u', 'v'], fmt)
        assert_equal(t, """
        u      | v
        1,000  | 1,000.00
        12,000 | 12,000.35
        """)


def test_currency_format():
    vs = ['$60', '$162.5']
    t = ds.Table([vs, vs, vs], ['num1', 'num2', 'str'])
    t.set_format(['num1', 'num2'], ds.CurrencyFormatter('$'))
    assert_equal(t, """
    num1    | num2    | str
    $60.00  | $60.00  | $60
    $162.50 | $162.50 | $162.5
    """)
    assert_equal(t.sort('num1'), """
    num1    | num2    | str
    $60.00  | $60.00  | $60
    $162.50 | $162.50 | $162.5
    """)
    assert_equal(t.sort('str'), """
    num1    | num2    | str
    $162.50 | $162.50 | $162.5
    $60.00  | $60.00  | $60
    """)


def test_date_format():
    vs = ['2015-07-01 22:39:44.900351']
    t = ds.Table([vs], ['time'])
    t.set_format('time', ds.DateFormatter("%Y-%m-%d %H:%M:%S.%f"))
    assert isinstance(t['time'][0], float)


def test_percent_formatter():
    vs = [0.1, 0.11111, 0.199999, 10]
    t = ds.Table([vs], ['percent'])
    t.set_format('percent', ds.PercentFormatter(1))
    assert_equal(t, """
    percent
    10.0%
    11.1%
    20.0%
    1000.0%
    """)
