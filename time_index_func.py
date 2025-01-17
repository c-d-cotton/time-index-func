#!/usr/bin/env python3
"""
To import everything:
sys.path.append(str(__projectdir__ / Path('submodules/time-index-func/')))
from time_index_func import *

"""

import os
from pathlib import Path
import sys


import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import pytz


# Definitions:{{{1
listoffreqs = ['y', 'q', 'm', 'w', 'd', 'H', 'M', 'S']

# General Functions for Time:{{{1
def convertmytimetodatetime(mytime):
    try:
        freq = mytime[-1]
        if freq == 'y':
            asdatetime = datetime.datetime.strptime(mytime[: -1], '%Y')
        elif freq == 'q':
            asdatetime = datetime.datetime(int(mytime[: 4]), int(mytime[4]) * 3, 1)
        elif freq == 'm':
            asdatetime = datetime.datetime.strptime(mytime[: 6], '%Y%m')
        elif freq == 'w' or freq == 'd':
            asdatetime = datetime.datetime.strptime(mytime[: 8], '%Y%m%d')
        elif freq == 'H':
            asdatetime = datetime.datetime.strptime(mytime[: 11], '%Y%m%d_%H')
        elif freq == 'M':
            asdatetime = datetime.datetime.strptime(mytime[: 13], '%Y%m%d_%H%M')
        elif freq == 'S':
            asdatetime = datetime.datetime.strptime(mytime[: 15], '%Y%m%d_%H%M%S')
        else:
            raise ValueError('Frequency not exist')
    except Exception:
        raise ValueError('Failed convertmytimetodatetime for: ' + mytime + '.')

    return(asdatetime)


def convertmytimetodate(mytime):
    freq = mytime[-1]
    if freq == 'y':
        asdatetime = datetime.date(int(mytime[0:4]), 1, 1)
    elif freq == 'q':
        asdatetime = datetime.date(int(mytime[0:4]), int(mytime[4])*3, 1)
    elif freq == 'm':
        asdatetime = datetime.date(int(mytime[0:4]), int(mytime[4:6]), 1)
    elif freq == 'w' or freq == 'd':
        asdatetime = datetime.date(int(mytime[0:4]), int(mytime[4:6]), int(mytime[6: 8]))
    else:
        raise ValueError('Frequency not exist')

    return(asdatetime)


def convertmytimetodatetime_test():
    """
    Takes a date in the format I use and convert it to datetime.
    """
    print(convertmytimetodatetime('2020y'))
    print(convertmytimetodatetime('20201q'))
    print(convertmytimetodatetime('202001m'))
    print(convertmytimetodatetime('20200102w'))
    print(convertmytimetodatetime('20200102d'))
    print(convertmytimetodatetime('20200102_03H'))
    print(convertmytimetodatetime('20200102_0306M'))
    print(convertmytimetodatetime('20200102_030609S'))


def convertdatetimetomytime(dt, freq):
    """
    Takes a datetime and converts it to the format I use for my indexes.
    """
    if freq == 'y':
        asmytime = dt.strftime('%Yy')
    elif freq == 'q':
        asmytime = str(dt.year) + str((dt.month + 2) // 3) + 'q'
    elif freq == 'm':
        asmytime = dt.strftime('%Y%mm')
    elif freq == 'w':
        asmytime = dt.strftime('%Y%m%dw')
    elif freq == 'd':
        asmytime = dt.strftime('%Y%m%dd')
    elif freq == 'H':
        asmytime = dt.strftime('%Y%m%d_%HH')
    elif freq == 'M':
        asmytime = dt.strftime('%Y%m%d_%H%MM')
    elif freq == 'S':
        asmytime = dt.strftime('%Y%m%d_%H%M%SS')
    else:
        raise ValueError('freq not defined. freq: ' + str(freq) + '.')

    return(asmytime)


def convertdatetimetomytime_test():
    dt = datetime.datetime(2020, 1, 2, 3, 6, 9)
    print(convertdatetimetomytime(dt, 'y'))
    print(convertdatetimetomytime(dt, 'q'))
    print(convertdatetimetomytime(dt, 'm'))
    print(convertdatetimetomytime(dt, 'w'))
    print(convertdatetimetomytime(dt, 'd'))
    print(convertdatetimetomytime(dt, 'H'))
    print(convertdatetimetomytime(dt, 'M'))
    print(convertdatetimetomytime(dt, 'S'))


def getallpointsbetween(mytime1, mytime2):
    """
    Get all periods between two of my datetimes
    Returns as mytime unless asmytime is False
    """
    freq = mytime1[-1]

    if mytime1 > mytime2:
        raise ValueError('mytime1 should be less than mytime2.')

    times = []

    if freq == 'y':
        times = list(range(int(mytime1[0: 4]), int(mytime2[0: 4]) + 1))
        times = [str(time) + 'y' for time in times]
    elif freq == 'q':
        years = list(range(int(mytime1[0: 4]), int(mytime2[0: 4]) + 1))
        for year in years:
            for quarter in list(range(1, 5)):
                times.append( str(year) + str(quarter) + 'q' )
        times = [time for time in times if time >= mytime1 and time <= mytime2]
    elif freq == 'm':
        years = list(range(int(mytime1[0: 4]), int(mytime2[0: 4]) + 1))
        months = []
        for year in years:
            for month in list(range(1, 13)):
                times.append( str(year) + str(month).zfill(2) + 'm' )
        times = [time for time in times if time >= mytime1 and time <= mytime2]
    elif freq in ['d', 'H', 'M', 'S']:
        
        # get dates
        dt1 = convertmytimetodate(mytime1[0: 8] + 'd')
        dt2 = convertmytimetodate(mytime2[0: 8] + 'd')
        delta = dt2 - dt1
        days = []
        for i in range(delta.days + 1):
            days.append( convertdatetimetomytime(dt1 + datetime.timedelta(days=i), 'd') )

        if freq == 'd':
            times = days
        elif freq == 'H':
            for day in days:
                for hour in list(range(0, 24)):
                    times.append( day[0: 8] + '_' + str(hour).zfill(2) + 'H' )
            times = [time for time in times if time >= mytime1 and time <= mytime2]
        elif freq == 'M':
            for day in days:
                for hour in list(range(0, 24)):
                    for minute in list(range(0, 60)):
                        times.append( day[0: 8] + '_' + str(hour).zfill(2) + str(minute).zfill(2) + 'M' )
            times = [time for time in times if time >= mytime1 and time <= mytime2]
        elif freq == 'S':
            for day in days:
                for hour in list(range(0, 24)):
                    for minute in list(range(0, 60)):
                        for second in list(range(0, 60)):
                            times.append( day[0: 8] + '_' + str(hour).zfill(2) + str(minute).zfill(2) + str(second).zfill(2) + 'S' )
            times = [time for time in times if time >= mytime1 and time <= mytime2]
        else:
            raise ValueError('freq misspecified: ' + str(freq) + '.')
    else:
        raise ValueError('freq misspecified: ' + str(freq) + '.')

    return(times)


def getallpointsbetween_test():
    print(getallpointsbetween('2001y', '2005y'))
    print(getallpointsbetween('200101m', '200105m'))
    print(getallpointsbetween('20010101d', '20010105d'))
    print(getallpointsbetween('20010101_00H', '20010101_04H'))
    print(getallpointsbetween('20010101_0000M', '20010101_0004M'))
    print(getallpointsbetween('20010101_000000S', '20010101_000004S'))


def countperiods(mytime1, mytime2, checkiflt = False):
    if checkiflt is False:
        if mytime1 > mytime2:
            mytime1temp = mytime2
            mytime2temp = mytime1
            mytime1 = mytime1temp
            mytime2 = mytime2temp
    between = getallpointsbetween(mytime1, mytime2)
    return(len(between) - 1)

def addperiods(mytime, pointstoadd):
    freq = mytime[-1]
    if freq == 'y':
        num = int(mytime[0: 4]) + pointstoadd
        return(str(num) + 'y')
    elif freq == 'q':
        num = int(mytime[0: 4]) * 4 + int(mytime[4]) - 1 + pointstoadd
        return(str(num // 4) + str(num % 4 + 1) + 'q')
    elif freq == 'm':
        num = int(mytime[0: 4]) * 12 + int(mytime[4: 6]) - 1 + pointstoadd
        return(str(num // 12) + str(num % 12 + 1).zfill(2) + 'm')
    elif freq == 'd':
        thisdate = convertdatetimetomytime(convertmytimetodatetime(mytime) + datetime.timedelta(days = pointstoadd), 'd')
        return(thisdate)
    elif freq == 'H':
        thisdatetime = convertdatetimetomytime(convertmytimetodatetime(mytime) + datetime.timedelta(seconds = pointstoadd * 60 * 60), 'H')
        return(thisdatetime)
    elif freq == 'M':
        thisdatetime = convertdatetimetomytime(convertmytimetodatetime(mytime) + datetime.timedelta(seconds = pointstoadd * 60), 'M')
        return(thisdatetime)
    elif freq == 'S':
        thisdatetime = convertdatetimetomytime(convertmytimetodatetime(mytime) + datetime.timedelta(seconds = pointstoadd), 'S')
        return(thisdatetime)
    else:
        raise ValueError('Freq not defined: ' + freq + '.')


def addperiods_test():
    """
    Add one point to first of 2010
    """
    print(addperiods("2010y", 1))
    print(addperiods("20101q", 1))
    print(addperiods("201001m", 1))
    print(addperiods("20100101d", 1))
    print(addperiods("20100101_00H", 1))
    print(addperiods("20100101_0000M", 1))
    print(addperiods("20100101_000000S", 1))


# Weekdays:{{{1
def getdayofweek(index_mytime):
    """
    0 = Monday
    6 = Sunday
    """
    dayofweek = [convertmytimetodatetime(mytime).weekday() for mytime in index_mytime]

    return(dayofweek)


def getdayofweek_test():
    df = pd.DataFrame({'interestrate': [1.01, 1.02, 1.03, 1.04, 1.05]}, index = ['20100701d', '20100702d', '20100703d', '20100704d', '20100705d'])
    df['dayofweek'] = getdayofweek(df.index)
    print(df)


def get_weekend_fri_sat():
    # based on https://www.diversityresources.com/holidays-and-work-schedule/
    weekend_fri_sat = []
    weekend_fri_sat.append('AFG') # Afghanistan
    weekend_fri_sat.append('ARE') # United Arab Emirates
    weekend_fri_sat.append('BHR') # Bahrain
    weekend_fri_sat.append('DZA') # Algeria
    weekend_fri_sat.append('DJI') # Djibouti
    weekend_fri_sat.append('EGY') # Egypt
    weekend_fri_sat.append('IRN') # Iran
    weekend_fri_sat.append('IRQ') # Iraq
    weekend_fri_sat.append('ISR') # Israel
    weekend_fri_sat.append('JOR') # Jordan
    weekend_fri_sat.append('KWT') # Kuwait
    weekend_fri_sat.append('LBY') # Libya
    weekend_fri_sat.append('MRT') # Mauritania
    weekend_fri_sat.append('OMN') # Oman
    weekend_fri_sat.append('QAT') # Qatar
    weekend_fri_sat.append('SAU') # Saudi Arabia
    weekend_fri_sat.append('SDN') # Sudan
    weekend_fri_sat.append('SYR') # Syria
    weekend_fri_sat.append('PSE') # Palestine
    return(weekend_fri_sat)


def weekdaysonly(df):
    df = df[[convertmytimetodatetime(mytime).weekday() in [0, 1, 2, 3, 4] for mytime in df.index]]

    return(df)


def weekdaysonly_test():
    df = pd.DataFrame({'interestrate': [1.01, 1.02, 1.03, 1.04, 1.05]}, index = ['20100701d', '20100702d', '20100703d', '20100704d', '20100705d'])
    df = weekdaysonly(df)
    print(df)


# Fill DataFrame:{{{1
def filltime(df):
    startdate = df.index[0]
    enddate = df.index[-1]

    dates = getallpointsbetween(startdate, enddate)

    df = df.reindex(dates)

    return(df)


def filltime_test():
    df = pd.DataFrame([[100], [101], [103]], columns = ['gdp'], index = ['20001q', '20002q', '20004q'])
    print(filltime(df))


# Adjusting Frequencies of Data:{{{1
def raisefreq(df, newfreq, howfill = 'sameval'):
    """
    Works for:
    - y -> q/m
    - q -> m

    howfill options:
    - 'sameval' (default): select every value at higher frequency to be the same as in the period for the lower frequency so 201001m/201002m/201003m take the same value as 20101q
    - 'none': Only fill in last value in the higher frequency period i.e. 201003m for 20101q
    - 'interpolate': interpolate the values so 201001m/201002m take values in between 20094q and 20101q

    """
    df = df.copy()

    firstdate = df.index[0]
    lastdate = df.index[-1]

    oldfreq = firstdate[-1]

    if oldfreq not in ['y', 'q']:
        raise ValueError('oldfreq not a frequency I use. oldfreq: ' + oldfreq + '.')
    if newfreq not in ['q', 'm']:
        raise ValueError('newfreq not a frequency I use. newfreq: ' + newfreq + '.')

    # verify newfreq is a higher frequency than oldfreq
    if listoffreqs.index(newfreq) <= listoffreqs.index(oldfreq):
        raise ValueError('newfreq does not have a higher frequency than oldfreq. newfreq: ' + newfreq + '. oldfreq: ' + oldfreq + '.')

    if oldfreq == 'y':
        if newfreq == 'q':
            df.index = df.index.str.slice(0,4) + '4q'
        elif newfreq == 'm':
            df.index = df.index.str.slice(0,4) + '4q'
        else:
            raise ValueError('new freq not defined: ' + str(newfreq) + '.')
    elif oldfreq == 'q':
        if newfreq == 'm':
            df.index = df.index.str.slice(0,4) + (df.index.str.slice(4, 5).astype(int)*3).astype(str).str.zfill(2) + 'm'
        else:
            raise ValueError('new freq not defined: ' + str(newfreq) + '.')
    
    if len(df.dropna(how = 'any')) != len(df) and howfill == 'interpolate':
        print('There are missing values in the lower frequency data that will be filled in with interpolate.')

    # fill gaps
    df = filltime(df)

    # add interpolation
    if howfill == 'none' or howfill is None:
        None
    elif howfill == 'sameval':
        # limit how far backfill to avoid filling in missing values
        if oldfreq == 'y' and newfreq == 'q':
            bfillnum = 3
        elif oldfreq == 'y' and newfreq == 'm':
            bfillnum = 11
        elif oldfreq == 'q' and newfreq == 'm':
            bfillnum = 2
        df = df.bfill(limit = bfillnum)
    elif howfill == 'interpolate':
        df = df.interpolate(limit_area = 'inside')
    else:
        raise ValueError('howfill misspecified')

    return(df)


def raisefreq_test():
    df = pd.DataFrame([[1], [2], [3]], columns = ['var1'], index = ['20101q', '20102q', '20103q'])
    print(raisefreq(df, 'm'))
    print(raisefreq(df, 'm', howfill = 'none'))
    print(raisefreq(df, 'm', howfill = 'interpolate'))

    # ensure not filling missing values
    df = pd.DataFrame([[1], [np.nan], [3]], columns = ['var1'], index = ['20101q', '20102q', '20103q'])
    print(raisefreq(df, 'm'))

# Strip NA Start/End:{{{1
def stripnarows_startend(df, start = True, end = True):
    """
    This function removes every na row at the start/end of a dataset until a row is reached where there is a non-na value.
    Note this is useful with time series data.

    To turn of start: start = False
    To turn of end: end = False

    PROBABLY A BETTER WAY!
    """
    if start is not True and end is not True:
        raise ValueError('At least one of start and end must be specified to be True')
    numrows = len(df)

    # turn into list to make iteration faster
    valueslist = df.values.tolist()

    if start is True:
        starti = None
        for i in range(numrows):
            for element in valueslist[i]:
                if pd.isnull(element) is False:
                    starti = i
                    break
            if starti is not None:
                break
    else:
        starti = 0

    if end is True:
        endi = None
        for i in reversed(range(numrows)):
            for element in valueslist[i]:
                if pd.isnull(element) is False:
                    endi = i
                    break
            if endi is not None:
                break
    else:
        endi = numrows - 1

    if starti is None or endi is None:
        raise ValueError('All rows appear to only contain na values.')

    df = df.iloc[starti: endi + 1, :].copy()

    return(df)

                    
def stripnarows_startend_test():
    df = pd.DataFrame([[np.nan], [100], [102], [np.nan], [np.nan]], index = ['2000y', '2001y', '2002y', '2003y', '2004y'], columns = ['gdp'])
    
    print(stripnarows_startend(df))
    print(stripnarows_startend(df, start = False, end = True))
    print(stripnarows_startend(df, start = True, end = False))

# Difference Between Periods:{{{1
def getdifferenceinperiods(list_mytime, includefirst = True):
    """
    Get difference between periods.
    No easy way to do this for months/quarters/years and it's unclear that I would want an easy way since I may want to capture the fact that the difference between Feb 1st and March 1st is different to March 1st and April 1st
    So just use days for years, quarters and months
    Every other measurement use same measure of difference as index
    """
    freq = list_mytime[0][-1]
    
    list_dt = [convertmytimetodatetime(mytime) for mytime in list_mytime]
    diff_dt = [list_dt[i] - list_dt[i - 1] for i in range(1, len(list_dt))]
    
    if freq == 'y' or freq == 'q' or freq == 'm' or freq == 'd':
        diff_per = [dt.days for dt in diff_dt]
    elif freq == 'w':
        diff_per = [dt.days / 7 for dt in diff_dt]
    elif freq == 'H':
        diff_per = [dt.seconds / 3600 for dt in diff_dt]
    elif freq == 'M':
        diff_per = [dt.seconds / 60 for dt in diff_dt]
    elif freq == 'S':
        diff_per = [dt.seconds for dt in diff_dt]
    else:
        raise ValueError('freq not defined. freq: ' + str(freq) + '.')

    if includefirst is True:
        # add in first element which is na since I can't subtract the time period before
        diff_per = [np.nan] + diff_per

    return(diff_per)


def getdifferenceinperiods_test():
    print( getdifferenceinperiods(['2000y', '2001y', '2002y', '2004y']) )
    print( getdifferenceinperiods(['20001q', '20002q', '20003q','20011q']) )
    print( getdifferenceinperiods(['200001m', '200002m', '200003m','200005m']) )
    print( getdifferenceinperiods(['20000101d', '20000102d', '20000103d','20000105d']) )
    print( getdifferenceinperiods(['20000101_00H', '20000101_01H', '20000101_02H','20000101_04H']) )
    print( getdifferenceinperiods(['20000101_0000M', '20000101_0001M', '20000101_0002M','20000101_0004M']) )
    print( getdifferenceinperiods(['20000101_000000S', '20000101_000001S', '20000101_000002S','20000101_000004S']) )
# Timezone Convert:{{{1
def tzconvert_single(dt, oldtimezone, newtimezone):
    """
    Convert dt from oldtimezone to newtimezone
    Get list of timezones using pytz.common_timezones
    Basic ones:
    GMT
    America/New_York
    America/Los_Angeles
    """
    dt2 = pytz.timezone(oldtimezone).localize(dt).astimezone(pytz.timezone(newtimezone))
    return(dt2)


def tzconvert_list(dtlist, oldtimezone, newtimezone):
    dtlist2 = [tzconvert_single(dt, oldtimezone, newtimezone) for dt in dtlist]
    return(dtlist2)
