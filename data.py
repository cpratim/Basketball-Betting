import pandas as pd
import os
from controls import *
import inspect
from math import isnan
import numpy as np

data_dir = 'data/raw'


team_stats = read_json('data/cleaned/json/team_stats.json')
standings = read_json('data/cleaned/json/standings.json')

def psqrt(n):
    return np.sign(n) * abs(n) ** .5

def remove_outliers(x, y, out_tol=1.5):

    nX, nY = [], []
    q1, q3 = np.percentile(sorted(y), [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (out_tol * iqr)
    upper_bound = q3 + (out_tol * iqr)

    for i in range(len(y)):
        if y[i] < upper_bound and y[i] > lower_bound:
            nX.append(x[i])
            nY.append(y[i])
    return np.array(nX), np.array(nY)


def load_xy(ret_odds=True, y_target='total', odds_target='over_under'):
    X, y = [], []
    failed = 0
    odds = []
    match_table = read_json('data/cleaned/json/match_table.json')
    for g in match_table:
        m = match_table[g]
        try:
            _x = m['home_standings'] + m['away_standings'] + m['home_avg_stats'] + m['away_avg_stats'] + m['home_odds'] + m['away_odds']
            odds.append(m[odds_target])
            X.append(_x)
            y.append(m[y_target])
        except:
            failed += 1
            continue
    #_X, _y = remove_outliers(X, y)
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0), np.nan_to_num(y, nan=0, posinf=0, neginf=0), odds

