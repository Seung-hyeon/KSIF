"""

"""

import os
import re
import decorator
import numpy as np
import pandas as pd
import datetime
import time

try:
    import cPickle as pickle
except ImportError:
    import pickle

__author__ = 'Seung Hyeon Yu'
__email__ = 'rambor12@business.kaist.ac.kr'


def _memoize(func, *args, **kw):
    # should we refresh the cache?
    refresh = False
    refresh_kw = func.mrefresh_keyword

    # kw is not always set - check args
    if refresh_kw in func.__code__.co_varnames:
        if args[func.__code__.co_varnames.index(refresh_kw)]:
            refresh = True

    # check in kw if not already set above
    if not refresh and refresh_kw in kw:
        if kw[refresh_kw]:
            refresh = True

    key = pickle.dumps(args, 1) + pickle.dumps(kw, 1)

    cache = func.mcache
    if not refresh and key in cache:
        return cache[key]
    else:
        cache[key] = result = func(*args, **kw)
        return result


def memoize(f, refresh_keyword='mrefresh'):
    """
    Memoize decorator. The refresh keyword is the keyword
    used to bypass the cache (in the function call).
    """
    f.mcache = {}
    f.mrefresh_keyword = refresh_keyword
    return decorator.decorator(_memoize, f)


def parse_arg(arg):
    """
    Parses arguments for convenience. Argument can be a
    csv list ('a,b,c'), a string, a list, a tuple.

    Returns a list.
    """
    # handle string input
    if type(arg) == str:
        arg = arg.strip()
        # parse csv as tickers and create children
        if ',' in arg:
            arg = arg.split(',')
            arg = [x.strip() for x in arg]
        # assume single string - create single item list
        else:
            arg = [arg]

    return arg


def clean_ticker(ticker):
    """
    Cleans a ticker for easier use throughout MoneyTree

    Splits by space and only keeps first bit. Also removes
    any characters that are not letters. Returns as lowercase.

    >>> clean_ticker('^VIX')
    'vix'
    >>> clean_ticker('SPX Index')
    'spx'
    """
    pattern = re.compile('[\W_]+')
    res = pattern.sub('', ticker.split(' ')[0])
    return res.lower()


def clean_tickers(tickers):
    """
    Maps clean_ticker over tickers.
    """
    return [clean_ticker(x) for x in tickers]


def fmtp(number):
    """
    Formatting helper - percent
    """
    if np.isnan(number):
        return '-'
    return format(number, '.2%')


def fmtpn(number):
    """
    Formatting helper - percent no % sign
    """
    if np.isnan(number):
        return '-'
    return format(number * 100, '.2f')


def fmtn(number):
    """
    Formatting helper - float
    """
    if np.isnan(number):
        return '-'
    return format(number, '.2f')


def get_period_name(period):
    period = period.upper()
    periods = {
        'B': 'business day',
        'C': 'custom business day',
        'D': 'daily',
        'W': 'weekly',
        'M': 'monthly',
        'BM': 'business month end',
        'CBM': 'custom business month end',
        'MS': 'month start',
        'BMS': 'business month start',
        'CBMS': 'custom business month start',
        'Q': 'quarterly',
        'BQ': 'business quarter end',
        'QS': 'quarter start',
        'BQS': 'business quarter start',
        'Y': 'yearly',
        'A': 'yearly',
        'BA': 'business year end',
        'AS': 'year start',
        'BAS': 'business year start',
        'H': 'hourly',
        'T': 'minutely',
        'S': 'secondly',
        'L': 'milliseonds',
        'U': 'microseconds'}

    if period in periods:
        return periods[period]
    else:
        return None


def scale(val, src, dst):
    """
    Scale value from src range to dst range.
    If value outside bounds, it is clipped and set to
    the low or high bound of dst.

    Ex:
        scale(0, (0.0, 99.0), (-1.0, 1.0)) == -1.0
        scale(-5, (0.0, 99.0), (-1.0, 1.0)) == -1.0

    """
    if val < src[0]:
        return dst[0]
    if val > src[1]:
        return dst[1]

    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]


def as_percent(self, digits=2):
    return as_format(self, '.%s%%' % digits)


def as_format(item, format_str='.2f'):
    """
    Map a format string over a pandas object.
    """
    if isinstance(item, pd.Series):
        return item.map(lambda x: format(x, format_str))
    elif isinstance(item, pd.DataFrame):
        return item.applymap(lambda x: format(x, format_str))


DECIMAL = 2


def prettyfloat(number):
    return "{:9.2f}".format(number)


def to_numeric(string):
    """
    numeric if parsing succeeded. Otherwise, str itself.
        Return type depends on input.
    :type string: str
    """

    if pd.isnull(string) or isinstance(string, float):
        return string
    else:
        try:
            return float(string.replace(',', ''))
        except:
            return string


def get_form(date):
    """
    Get Form from the date

    :param date:
    :return: form
    """
    if isinstance(date, str):
        if len(date) == 8:
            form = "%Y%m%d"
        elif '-' in date:
            form = "%Y-%m-%d"
        elif '/' in date:
            form = "%Y/%m/%d"
        elif '.' in date:
            form = "%Y.%m.%d"
        else:
            raise NotImplementedError

    return form


def date_to_numeric(date):
    """
    Return Unix Time which is total elapsed nanoseconds from 1970-01-01

    :param date: any time format
    :return: int total elapsed nanoseconds from 1970-01-01
    """
    if isinstance(date, pd.tslib.Timestamp):
        return date.value
    elif isinstance(date, (pd.datetime, np.datetime64)):
        return pd.Timestamp(date).value
    elif isinstance(date, str):
        return int(time.mktime(str_to_date(date).timestamp()))


def date_to_str(date, form="%Y-%m-%d"):
    """
    Return Date String

    :param date: date
    :param form: format of return
    :return: str formatted date time
    """
    if isinstance(date, str):
        return date
    elif isinstance(date, (pd.tslib.Timestamp, pd.datetime)):
        return date.strftime(form)


def str_to_date(date, form=None):
    """
    Return Date with datetime format

    :param form:
    :param date: str date
    :return: datetime date
    """
    if form is None:
        form = get_form(date)
    return datetime.datetime.strptime(date, form)


def to_list(*args):
    """
    Return list

    :return: list of listed values
    """
    result = []
    for arg in args:
        if isinstance(arg, (str, int, float)):
            result.append([arg])
        elif isinstance(arg, (list, pd.Series, np.ndarray)):
            result.append(list(arg))
        elif arg is None:
            result.append(arg)
        else:
            raise NotImplementedError
    return result


def ranking(data: pd.DataFrame, group, key, option='relative'):
    """
    Return a rank of variable in a group

    For example, if you have data as
    a : 67
    b : 23
    c : 1
    d : 6
    e : 7

    Then the absolute ranks are
    a : 5
    b : 4
    c : 1
    d : 2
    e : 3

    And the relative ranks are
    a : 1
    b : 4/5
    c : 1/5
    d : 2/5
    e : 3/5

    So you have to notice that the relative ranks have a range of [1/n, 1]

    :param data: (pd.DataFrame)
    :param group: (list)
    :param key: (str) the variable you want to rank
    :param option: (str)
    :return: (pd.Series) ranked Series
    """

    temp = data
    temp[key + 'r'] = temp.groupby(group)[key].rank(ascending=True)
    if option == 'absolute':
        return temp[key + 'r']

    temp = temp.groupby(group)
    if option == 'relative':
        return temp[key + 'r'].apply(lambda x: x / len(x))
    else:
        print(" Need ranking Option ! Choose {'relative', 'absolute'}")
        return None


def cleanse(data, selector, group=None, pos_filter=True):
    """
    Cleanse data without redundant columns.
    In this part, you can rank the variables using 'r' at the end of the column name
    (e.g.)
    'PERr' means the relative ranks of 'PER'
    'EBITDAr' means the relative ranks of 'EBITDA'

    :param data: (pd.DataFrame)
    :param selector: (list) the list of column names you want to save
    :param group: (list) the list of group when you calculate ranks
    :param pos_filter: (Bool) whether you ranks only positive parts or not
    :return: (pd.DataFrame) cleansed data
    """

    if group is None:
        group = ['DATE', 'FNSECTCODE']
    keys = []

    for key in selector:
        if key[-1] == 'r':
            keys.append(key[:-1])
        else:
            keys.append(key)

    data = data[list(set(group + keys))]  # removing duplicates in list

    for key in selector:
        if key[-1] == 'r':
            if pos_filter is True:
                filtered = data[data[key[:-1]] > 0]
            data[key] = ranking(filtered, group=group, key=key[:-1])

    return data


def curpath():
    pth, _ = os.path.split(os.path.abspath(__file__))

    return pth
