"""Standalone fetch job for live WNBA data, run on a schedule outside Streamlit Cloud.

Streamlit Community Cloud's outbound IPs are blocked by stats.wnba.com/stats.nba.com,
so live nba_api calls can no longer happen inside the deployed app. This script mirrors
the same calls main.py used to make live (load_data()'s LeagueGameLog calls and
get_scoreboard()/build_player_list()'s ScoreboardV3/CommonTeamRoster calls) and writes
the results to CSV files that the app reads instead.
"""
import sys
import time
import datetime

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV3, commonteamroster

WNBA_LEAGUE_ID = '10'
WNBA_CURRENT_SEASON = '2026'

BOX_SCORE_CSV = 'current_season_box.csv'
MATCHUPS_CSV = 'todays_matchups.csv'


def fetch_current_season_box() -> pd.DataFrame:
    playerbox = LeagueGameLog(
        player_or_team_abbreviation='P',
        season_type_all_star='Regular Season',
        season=WNBA_CURRENT_SEASON,
        league_id=WNBA_LEAGUE_ID
    ).get_data_frames()[0]
    time.sleep(1)
    try:
        playoffs = LeagueGameLog(
            player_or_team_abbreviation='P',
            season_type_all_star='Playoffs',
            season=WNBA_CURRENT_SEASON,
            league_id=WNBA_LEAGUE_ID
        ).get_data_frames()[0]
        if not playoffs.empty:
            playerbox = pd.concat([playerbox, playoffs], ignore_index=True)
    except Exception:
        pass
    return playerbox


def fetch_scoreboard() -> list:
    today = datetime.date.today().strftime('%Y-%m-%d')
    board = ScoreboardV3(game_date=today, league_id=WNBA_LEAGUE_ID)
    game_header = board.game_header.get_data_frame()
    line_scores = board.line_score.get_data_frame()
    matchups = []
    for _, row in game_header.iterrows():
        game_id = row['gameId']
        team_codes = row['gameCode'].split('/')[1]
        away_tricode = team_codes[:3]
        home_tricode = team_codes[3:]
        game_lines = line_scores[line_scores['gameId'] == game_id]
        away_id = game_lines[game_lines['teamTricode'] == away_tricode]['teamId'].values
        home_id = game_lines[game_lines['teamTricode'] == home_tricode]['teamId'].values
        if len(away_id) and len(home_id):
            matchups.append([away_id[0], away_tricode, home_id[0], home_tricode])
    return matchups


def fetch_todays_matchups() -> pd.DataFrame:
    matchups = fetch_scoreboard()
    playeroutput = pd.DataFrame()
    for awayid, awayabb, homeid, homeabb in matchups:
        time.sleep(1)
        awayroster = commonteamroster.CommonTeamRoster(
            team_id=awayid, season=WNBA_CURRENT_SEASON, league_id_nullable=WNBA_LEAGUE_ID
        )
        time.sleep(1)
        homeroster = commonteamroster.CommonTeamRoster(
            team_id=homeid, season=WNBA_CURRENT_SEASON, league_id_nullable=WNBA_LEAGUE_ID
        )
        away_df = pd.DataFrame(awayroster.get_data_frames()[0].PLAYER_ID)
        away_df['Home'] = False
        away_df['OPP'] = str([homeabb])
        away_df['TEAM'] = str([awayabb])
        home_df = pd.DataFrame(homeroster.get_data_frames()[0].PLAYER_ID)
        home_df['Home'] = True
        home_df['OPP'] = str([awayabb])
        home_df['TEAM'] = str([homeabb])
        playeroutput = pd.concat([playeroutput, away_df, home_df]).drop_duplicates()
    return playeroutput


def main() -> int:
    exit_code = 0

    try:
        box = fetch_current_season_box()
        box.to_csv(BOX_SCORE_CSV, index=False)
        print(f"Wrote {len(box)} rows to {BOX_SCORE_CSV}")
    except Exception as e:
        print(f"FAILED to fetch box scores, leaving {BOX_SCORE_CSV} untouched: {e}", file=sys.stderr)
        exit_code = 1

    try:
        matchups = fetch_todays_matchups()
        matchups.to_csv(MATCHUPS_CSV, index=False)
        print(f"Wrote {len(matchups)} rows to {MATCHUPS_CSV}")
    except Exception as e:
        print(f"FAILED to fetch today's matchups, leaving {MATCHUPS_CSV} untouched: {e}", file=sys.stderr)
        exit_code = 1

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
