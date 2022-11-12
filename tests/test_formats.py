import doctest
import re
import datascience as ds
from datascience import formats
import random


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
    assert_equal(fmt(-1), '-1')  # edge case
    assert_equal(fmt(0), '0')  # edge case
    assert_equal(fmt(1.23456789), '1.23457')
    assert_equal(fmt(123456789), '123456789')
    assert_equal(fmt(123456789 ** 5), '28679718602997181072337614380936720482949')
    assert_equal(fmt(123456789 ** 10),
                 '822526259147102579504761143661535547764137892295514168093701699676416207799736601')  # edge case
    assert_equal(fmt(123.456789 ** 5), '2.86797e+10')
    assert_equal(fmt(123 ** 5), '28153056843')  # edge case
    assert_equal(fmt(123.5 ** 2), '15252.2')  # edge case
    assert_equal(fmt(123.1 ** 2), '15153.6')  # edge case
    assert_equal(fmt(True), 'True')
    assert_equal(fmt(False), 'False')
    assert_equal(fmt('hello'), 'hello')
    assert_equal(fmt((1, 2)), '(1, 2)')
    assert_equal(fmt((1, 2, 3, 4, 5, 6, 7)), '(1, 2, 3, 4, 5, 6, 7)')  # edge case
    assert_equal(
        fmt('this is a very long string with spaces and more than 60 characters, but who is counting?'),
        'this is a very long string with spaces and more than 60 characters, but who is counting?')  # edge case

    # randomize floating point number rounding test
    test_cases = 1000000

    for i in range(test_cases):

        # generate a random floating point number
        num = random.uniform(-999999999999.9, 999999999999.9)

        # cast to string
        num_str = str(num)

        neg_flag = False

        # check for negative
        if num_str[:1] == "-":
            neg_flag = True
            num_str = num_str[1:]
            num = num * -1  # need num to be positive for some of the calculations below

        # split string before and after the decimal point
        # returns an array with 2 elements -> integer part at 0 and decimal part at 1
        split_str = num_str.split(".")

        # count lengths
        int_len = len(split_str[0])
        dec_len = len(split_str[1])

        # if integer + decimal part <= 6, return it as is
        if int_len + dec_len < 6:
            # string to test is unchanged
            rand_result = num_str

        # round decimal portion if integer portion < 6 digits long
        elif int_len < 6:
            rand_result = str(round(num, 6 - int_len))

        # simple whole number round if integer portion is 6 digits long
        elif int_len == 6:
            rand_result = str(round(num))

        # truncate the integer part if it is > 6
        else:
            # shift the decimal to the second character in the string
            test_str = split_str[0] + split_str[1]
            test_str = test_str[:1] + "." + test_str[1:]

            # round up
            test_str = str(round(float(test_str), 5))

            # if the second character in test_str is NOT a decimal point, the number was rounded up one magnitude
            if test_str[1:2] != ".":
                test_str = test_str[:1]
                # power increases by one
                power = str(int_len)

            else:
                # find power and cast to string normally
                power = str(int_len - 1)

            # if test_str is a whole number, remove the decimal portion
            if len(test_str) == 3 and test_str[2:3] == "0":
                test_str = test_str[:1]

            # power leads with a 0 if < 10
            if len(power) == 1:
                power = "0" + power

            rand_result = test_str + "e+" + power

        if neg_flag:
            rand_result = "-" + rand_result
            num = num * -1  # revert to negative for fmt() result

        # test if the randomized result is the same as the formatted result
        assert_equal(fmt(num), rand_result)



def test_number_format():
    for fmt in [ds.NumberFormatter(2), ds.NumberFormatter]:
        # edge cases: 0, 1000000
        us = ['0', '1,000', '12,000', '1,000,000', '1,000,000,000']
        vs = ['0', '1,000', '12,000.346', '1,000,000', '1,000,000,000']
        t = ds.Table().with_columns('u', us, 'v', vs)
        t.set_format(['u', 'v'], fmt)
        assert_equal(t, """
        u      | v
        0      | 0.00
        1,000  | 1,000.00
        12,000 | 12,000.35
        1,000,000 | 1,000,000.00
        1,000,000,000 | 1,000,000,000.00
        """)


def test_currency_format():
    # edge cases: 0, one non-zero digit: 6, five digits: -16,257, 2 decimal places: -16,257.55
    # duplicate values: 0 and 60
    vs = ['$0', '$0', '$6', '$60', '$60', '$162.5', '$1,625', '$-16,257.55']
    t = ds.Table().with_columns('num', vs, 'str', vs)
    t.set_format('num', ds.CurrencyFormatter('$', int_to_float=True))
    # check that table was created correctly
    assert_equal(t, """
    num       | str
    $0.00     | $0
    $0.00     | $0
    $6.00     | $6
    $60.00    | $60
    $60.00    | $60
    $162.50   | $162.5
    $1,625.00 | $1,625
    $-16,257.55| $-16,257.55
    """)
    # sort via num -> sorts numerically
    assert_equal(t.sort('num'), """
    num       | str
    $-16,257.55| $-16,257.55
    $0.00     | $0
    $0.00     | $0
    $6.00     | $6
    $60.00    | $60
    $60.00    | $60
    $162.50   | $162.5
    $1,625.00 | $1,625
    """)
    # sort via num with DISTINCT values-> sorts numerically
    assert_equal(t.sort('num', distinct=True), """
    num       | str
    $-16,257.55| $-16,257.55
    $0.00     | $0
    $6.00     | $6
    $60.00    | $60
    $162.50   | $162.5
    $1,625.00 | $1,625
    """)
    # sort via num DESCENDING -> sorts numerically
    assert_equal(t.sort('num', descending=True), """
    num       | str
    $1,625.00 | $1,625
    $162.50   | $162.5
    $60.00    | $60
    $60.00    | $60
    $6.00     | $6
    $0.00     | $0
    $0.00     | $0
    $-16,257.55| $-16,257.55
    """)
    # sort via str -> sorts alphabetically
    assert_equal(t.sort('str'), """
    num       | str
    $-16,257.55| $-16,257.55
    $0.00     | $0
    $0.00     | $0
    $1,625.00 | $1,625
    $162.50   | $162.5
    $6.00     | $6
    $60.00    | $60
    $60.00    | $60
    """)


def test_currency_format_int():
    # edge cases: 0, 10, 100, -999, 1000, 10000, -1000000
    # all rows >10 in the table are omitted in the assert_equal comparison, so we can't test a larger table
    t = ds.Table().with_column('money', [0, 1, 2, 3, 10, 100, -999, 1000, 10000, -1000000])
    t.set_format(['money'], ds.CurrencyFormatter)
    assert_equal(t, """
    money
    $0
    $1
    $2
    $3
    $10
    $100
    $-999
    $1,000
    $10,000
    $-1,000,000
    """)


def test_date_format():
    vs = ['2015-07-01 22:39:44.900351']
    t = ds.Table().with_column('time', vs)
    t.set_format('time', ds.DateFormatter("%Y-%m-%d %H:%M:%S.%f"))
    assert isinstance(t['time'][0], float)


def test_percent_formatter():
    # edge cases: 0.0, 0.499, 0.4994, 0.4995, -0.4999, 1.0
    vs = [0.0, 0.1, 0.11111, 0.199999, 0.499, 0.4994, 0.4995, -0.4999, 1.0, 10]
    t = ds.Table().with_column('percent', vs)
    t.set_format('percent', ds.PercentFormatter(1))
    assert_equal(t, """
    percent
    0.0%
    10.0%
    11.1%
    20.0%
    49.9%
    49.9%
    50.0%
    -50.0%
    100.0%
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
