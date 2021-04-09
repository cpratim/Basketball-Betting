from controls import *
import inspect
import pandas as pd
import numpy as np

team_stats = read_json('data/cleaned/json/team_stats.json')
standings = read_json('data/cleaned/json/standings.json')

def get_values(c, ignored=[]):
    values = []
    nms = []
    for i in inspect.getmembers(c): 
        nm = i[0]
        if not nm.startswith('_'): 
            if nm in ignored:
                continue
            v = i[1]
            try:
                if isnan(v):
                    v = 0
                values.append(float(v))
            except:
                pass
    return values

def concat(sw):
    dfs = []
    for fl in os.listdir(data_dir):
        for f in os.listdir(f'{data_dir}/{fl}'):
            if f == sw:
                df = pd.read_csv(f'{data_dir}/{fl}/{f}')
                dfs.append(df)
    cdf = pd.concat(dfs)
    cdf.to_csv(f'data/cleaned/{sw}')


def get_avg_stats(teamid, gameid, back=3):
    ts = team_stats[teamid]
    games = sorted(list(ts.keys()))
    ind = games.index(gameid)
    near = games[ind-3:ind]
    stats = []
    for g in near:
        _s = np.array(ts[g])
        if len(stats) == 0:
            stats = _s
        else:
            stats += _s
    return list(stats/back)

def load_standings():
    df = pd.read_csv('data/cleaned/csv/raw_scores.csv')
    otc = [f'PTS_OT{i}' for i in range(2, 11)]
    df = df.drop(otc, axis=1)
    standings = {}
    for row in df.itertuples():
        gameid = row.GAME_ID
        teamid = row.TEAM_ID
        wl = [int(i) for i in row.TEAM_WINS_LOSSES.split('-')]
        if gameid not in standings:
            standings[gameid] = {}
        standings[gameid][teamid] = wl
    dump_json('data/cleaned/json/standings.json', standings)


def load_team_stats():
    df = pd.read_csv('data/cleaned/csv/raw_scores.csv')
    otc = [f'PTS_OT{i}' for i in range(2, 11)]
    df = df.drop(otc, axis=1)
    team_stats = {}
    for row in df.itertuples():
        gameid = row.GAME_ID
        teamid = row.TEAM_ID
        stats = get_values(row, ignored=['GAME_ID', 'TEAM_ID', 'Index'])
        if teamid not in team_stats:
            team_stats[teamid] = {}
        team_stats[teamid][gameid] = stats
    dump_json('data/cleaned/json/team_stats.json', team_stats)

def load_match_table():
    df = pd.read_csv('data/cleaned/csv/vegas.csv')
    
    match_table = {}
    failed = 0
    for row in df.itertuples():
        gameid = row.GameId
        teamid = row.TeamId
        loc = row.Location
        spread = row.Spread
        stats = get_values(row, ignored=['Spread', 'Pts', 'Total', 'Index', 'GameId', 'TeamId'])
        try:
            ts = get_avg_stats(str(teamid), str(gameid))
        except Exception as e:
            failed += 1
            continue
        if gameid not in match_table:
            match_table[gameid] = {}
        if loc == 'away':
            stn = standings[str(gameid)][str(teamid)]
            if spread > 0:
                stn[0] -= 1
            else:
                stn[1] -= 1
            match_table[gameid]['away_id'] = teamid
            match_table[gameid]['away_odds'] = stats
            match_table[gameid]['away_standings'] = stn
            match_table[gameid]['away_avg_stats'] = ts
        else:

            stn = standings[str(gameid)][str(teamid)]
            if spread > 0:
                stn[0] -= 1
            else:
                stn[1] -= 1
            match_table[gameid]['home_standings'] = stn
            match_table[gameid]['home_id'] = teamid
            match_table[gameid]['home_odds'] = stats
            match_table[gameid]['home_avg_stats'] = ts
            match_table[gameid]['spread'] = row.Spread
            match_table[gameid]['total'] = row.Average_Line_OU - row.Total
            match_table[gameid]['over_under'] = row.Average_Line_OU 
            match_table[gameid]['ml'] = row.Average_Line_ML
            match_table[gameid]['odds_spread'] = row.Average_Line_Spread


    dump_json('data/cleaned/json/match_table.json', match_table)

load_match_table()