import time
import math
import difflib
import datetime

import nba_api.stats.library.data
import numpy as np
import matplotlib.pyplot as plt
import requests
import streamlit as st
import pandas
from scipy.stats import poisson as scipy_poisson
from nba_api.stats.library.parameters import SeasonTypeAllStar, PlayerOrTeamAbbreviation
from sklearn.cluster import KMeans
from nba_api.stats.endpoints import LeagueGameLog, SynergyPlayTypes, PlayerDashPtShots, shotchartdetail, \
    ShotChartDetail, LeagueHustleStatsPlayer, synergyplaytypes, LeagueDashPtStats, ScoreboardV3
from nba_api.stats.static.players import get_players
from nba_api.stats.endpoints import commonteamroster
from sklearn.preprocessing import StandardScaler
import pandas as pd
from FanduelScrape import getnbaprops, reformat_api, get_team_totals

WNBA_LEAGUE_ID = '10'
WNBA_CURRENT_SEASON = '2026'  # Most recent completed WNBA season
WNBA_SEASONS = ['2023', '2024', '2025','2026']

# WNBA teams: id, abbreviation, full name
WNBA_TEAMS = [
    {'id': 1611661313, 'abbreviation': 'NYL', 'full_name': 'New York Liberty'},
    {'id': 1611661316, 'abbreviation': 'LVA', 'full_name': 'Las Vegas Aces'},
    {'id': 1611661317, 'abbreviation': 'MIN', 'full_name': 'Minnesota Lynx'},
    {'id': 1611661319, 'abbreviation': 'CON', 'full_name': 'Connecticut Sun'},
    {'id': 1611661320, 'abbreviation': 'PHX', 'full_name': 'Phoenix Mercury'},
    {'id': 1611661321, 'abbreviation': 'CHI', 'full_name': 'Chicago Sky'},
    {'id': 1611661322, 'abbreviation': 'ATL', 'full_name': 'Atlanta Dream'},
    {'id': 1611661323, 'abbreviation': 'LAS', 'full_name': 'Los Angeles Sparks'},
    {'id': 1611661324, 'abbreviation': 'IND', 'full_name': 'Indiana Fever'},
    {'id': 1611661325, 'abbreviation': 'DAL', 'full_name': 'Dallas Wings'},
    {'id': 1611661326, 'abbreviation': 'SEA', 'full_name': 'Seattle Storm'},
    {'id': 1611661327, 'abbreviation': 'WAS', 'full_name': 'Washington Mystics'},
    {'id': 1611661328, 'abbreviation': 'GSV', 'full_name': 'Golden State Valkyries'},
    {'id': 1611661329, 'abbreviation': 'TOR', 'full_name': 'Toronto Tempo'},   # verify id
    {'id': 1611661330, 'abbreviation': 'POR', 'full_name': 'Portland Fire'},    # verify id
]

# Per-stat half-life values (games) derived from backtest.
# Weight of a game N games ago = 0.5 ^ (N / half_life). 999 ≈ no decay.
STAT_HALF_LIFE = {
    # Offensive — volume/shooting: no meaningful decay
    'PTS':  999,
    'FGM':  999,
    'FGA':  999,
    'FG3M': 999,
    'FG3A': 999,
    'FTM':  999,
    'FTA':  999,
    'TOV':  999,
    # Playmaking — modest decay (backtest optimum: 15)
    'AST':  15,
    # Rebounding — moderate decay; recent matchups matter but don't overreact (optimum: 1, using 10)
    'OREB': 10,
    'DREB': 10,
    # Defensive activity — moderate decay (optimum: 10)
    'STL':  10,
    'BLK':  10,
    'PF':   10,
    # Composite — handled as OREB+DREB components, but keep a safe default
    'REB':  999,
}

# FanDuel category → component stats used for projection
_PROP_STAT_COMPONENTS = {
    'PTS': ['PTS'],
    'REB': ['OREB', 'DREB'],
    'AST': ['AST'],
    'STL': ['STL'],
    'BLK': ['BLK'],
    '3PM': ['FG3M'],
    'PA':  ['PTS', 'AST'],
    'PR':  ['PTS', 'OREB', 'DREB'],
    'PRA': ['PTS', 'OREB', 'DREB', 'AST'],
    'RA':  ['OREB', 'DREB', 'AST'],
}

_OFFENSIVE_STATS = ['PTS', 'AST', 'OREB', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV']
_DEFENSIVE_STATS = ['DREB', 'REB', 'STL', 'BLK', 'PF']


@st.cache_data(ttl=21600)
def load_fanduel_props():
    try:
        raw = getnbaprops()
        if not raw:
            return pd.DataFrame()
        return reformat_api(raw)
    except Exception:
        return pd.DataFrame()


# Maps Odds API full team names → WNBA abbreviations
_ODDS_TEAM_TO_ABBREV = {
    'Atlanta Dream': 'ATL',
    'Chicago Sky': 'CHI',
    'Connecticut Sun': 'CON',
    'Dallas Wings': 'DAL',
    'Golden State Valkyries': 'GSV',
    'Indiana Fever': 'IND',
    'Las Vegas Aces': 'LVA',
    'Los Angeles Sparks': 'LAS',
    'Minnesota Lynx': 'MIN',
    'New York Liberty': 'NYL',
    'Phoenix Mercury': 'PHX',
    'Seattle Storm': 'SEA',
    'Washington Mystics': 'WAS',
    'Toronto Tempo': 'TOR',
    'Portland Fire': 'POR',
}


@st.cache_data(ttl=3600)
def load_team_totals():
    """Return {team_abbrev: total_points} from FanDuel team total lines."""
    raw = get_team_totals()
    return {_ODDS_TEAM_TO_ABBREV[name]: total
            for name, total in raw.items()
            if name in _ODDS_TEAM_TO_ABBREV}


def _to_american(odds):
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return None
    odds = float(odds)
    if 1.0 <= odds < 100.0:  # decimal format
        if odds >= 2.0:
            return round((odds - 1) * 100)
        else:
            return round(-100 / (odds - 1))
    return int(odds)  # already American


def _american_to_decimal(odds):
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return None
    odds = float(odds)
    # Decimal odds are in [1.0, ~50); American are >= 100 or <= -100
    if 1.0 <= odds < 100.0:
        return odds  # already decimal
    elif odds >= 100:
        return 1 + odds / 100
    else:  # negative American (e.g. -110)
        return 1 + 100 / abs(odds)


def _calc_poisson_ev(mu, line, over_odds, under_odds):
    """Returns (ev_over_pct, ev_under_pct, p_over_pct) using Poisson(mu)."""
    if mu <= 0 or line < 0:
        return None, None, None
    k = int(line)  # floor; handles .5 lines correctly
    p_over = 1.0 - scipy_poisson.cdf(k, mu)
    p_under = scipy_poisson.cdf(k, mu)
    d_over = _american_to_decimal(over_odds)
    d_under = _american_to_decimal(under_odds)
    ev_over = (p_over * (d_over - 1) - p_under) * 100 if d_over is not None else None
    ev_under = (p_under * (d_under - 1) - p_over) * 100 if d_under is not None else None
    return ev_over, ev_under, p_over * 100


@st.cache_data(ttl=21600)
def _build_player_season_stats(merged, min_date, max_date):
    """Compute per-player season averages and last-10-game median minutes in one pass."""
    all_cols = _OFFENSIVE_STATS + _DEFENSIVE_STATS
    filtered = merged[
        (merged['GAME_DATE'] >= min_date) &
        (merged['GAME_DATE'] <= max_date) &
        (merged['MIN'] >= 5)
    ]
    player_avgs = filtered.groupby('PLAYER_NAME')[all_cols + ['MIN']].mean()
    player_last10_min = (
        filtered.sort_values('GAME_DATE', ascending=False)
        .groupby('PLAYER_NAME')
        .head(10)
        .groupby('PLAYER_NAME')['MIN']
        .median()
    )
    return player_avgs, player_last10_min


def bayesian_pct_diff(game_diffs: np.ndarray, stat: str,
                      prior_mean: float = 0.0, prior_weight: float = 3.0) -> float:
    """Bayesian sequential update with per-stat games-ago exponential decay."""
    half_life = STAT_HALF_LIFE.get(stat, 999)
    n         = len(game_diffs)
    estimate  = prior_mean
    total_wt  = prior_weight
    for i, diff in enumerate(game_diffs):
        games_ago   = n - 1 - i          # 0 = most recent game
        game_weight = 0.5 ** (games_ago / half_life)
        estimate    = (total_wt * estimate + game_weight * diff) / (total_wt + game_weight)
        total_wt   += game_weight
    return estimate


@st.cache_data(ttl=1800)
def get_inactive_players() -> set:
    """
    Return a set of player names (displayName) who are Out or Doubtful today.
    Source: ESPN public injuries API.
    """
    INACTIVE_STATUSES = {'out', 'doubtful'}
    try:
        url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries'
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        inactive = set()
        for team_block in data.get('injuries', []):
            for injury in team_block.get('injuries', []):
                status = injury.get('status', '').lower().strip()
                if status in INACTIVE_STATUSES:
                    athlete = injury.get('athlete', {})
                    name = athlete.get('displayName', '').strip()
                    if name:
                        inactive.add(name)
        return inactive
    except Exception as e:
        print(f"Injury report fetch failed: {e}")
        return set()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_todays_matchups(merged,min_date, max_date):
    """Load and calculate today's matchup data with caching"""

    try:
        # Get today's games and players
        todays_players = build_player_list()

        if todays_players.empty:
            return None

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. NBA API is slow or unreachable.")
        return None

    except requests.exceptions.ConnectionError:
        st.error("🌐 Connection error. Check your internet connection.")
        return None

    except requests.exceptions.HTTPError as e:
        st.error(f"❌ NBA API error: {str(e)}")
        return None

    except Exception as e:
        st.error(f"❌ Error loading today's games: {str(e)}")
        st.info("This could mean: no games today, API is down, or rate limiting.")
        return None

    try:
        # Clean up the OPP and TEAM columns (stored as str(list) e.g. "['LAL']")
        todays_players['OPP'] = todays_players['OPP'].str.strip("[]'\"")
        todays_players['TEAM'] = todays_players['TEAM'].str.strip("[]'\"")

        # Merge with cluster data
        todays_players_merged = todays_players.merge(
            merged[['PLAYER_ID', 'PLAYER_NAME', 'OffCluster', 'DefCluster']].drop_duplicates('PLAYER_ID'),
            on='PLAYER_ID',
            how='inner'
        )

        if todays_players_merged.empty:
            return None

    except KeyError as e:
        st.error(f"❌ Data format error: Missing column {str(e)}")
        return None

    except Exception as e:
        st.error(f"❌ Error merging player data: {str(e)}")
        return None

    try:
        # Calculate matchup scores
        matchup_scores = []
        MIN_THRESHOLD = 5
        offensive_stats = ['PTS', 'AST', 'OREB', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV']
        defensive_stats = ['DREB', 'REB', 'STL', 'BLK', 'PF']

        # Pre-filter merged data by date range once
        merged_filtered = merged[
            (merged['GAME_DATE'] >= min_date) &
            (merged['GAME_DATE'] <= max_date)
            ].copy()

        if merged_filtered.empty:
            st.warning("⚠️ No historical data in selected date range")
            return None

        # Pre-calculate cluster averages for all clusters
        cluster_stats_cache = {}

        for idx, player_row in todays_players_merged.iterrows():
            try:
                player_id = player_row['PLAYER_ID']
                player_name = player_row['PLAYER_NAME']
                opponent = player_row['OPP']
                player_team = player_row.get('TEAM', '')
                off_cluster = player_row['OffCluster']
                def_cluster = player_row['DefCluster']
                is_home = player_row['Home']

                if pd.isna(off_cluster) or pd.isna(def_cluster):
                    continue

                # Use cached cluster data if available
                cache_key_off = f"off_{off_cluster}"
                cache_key_def = f"def_{def_cluster}"

                if cache_key_off not in cluster_stats_cache:
                    df_off_cluster = merged_filtered[merged_filtered['OffCluster'] == off_cluster]
                    df_off_valid = df_off_cluster[df_off_cluster['MIN'] >= MIN_THRESHOLD]

                    if df_off_valid.empty:
                        continue

                    df_off_per36 = df_off_valid[offensive_stats].div(df_off_valid['MIN'], axis=0) * 36
                    season_avg_per36_off = df_off_per36.mean()
                    df_off_cluster['OPP_TEAM'] = df_off_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                    cluster_stats_cache[cache_key_off] = {
                        'df': df_off_cluster,
                        'season_avg': season_avg_per36_off
                    }

                if cache_key_def not in cluster_stats_cache:
                    df_def_cluster = merged_filtered[merged_filtered['DefCluster'] == def_cluster]
                    df_def_valid = df_def_cluster[df_def_cluster['MIN'] >= MIN_THRESHOLD]

                    if df_def_valid.empty:
                        continue

                    df_def_per36 = df_def_valid[defensive_stats].div(df_def_valid['MIN'], axis=0) * 36
                    season_avg_per36_def = df_def_per36.mean()
                    df_def_cluster['OPP_TEAM'] = df_def_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                    cluster_stats_cache[cache_key_def] = {
                        'df': df_def_cluster,
                        'season_avg': season_avg_per36_def
                    }

                # Get from cache
                df_off_cluster = cluster_stats_cache[cache_key_off]['df']
                season_avg_per36_off = cluster_stats_cache[cache_key_off]['season_avg']
                df_def_cluster = cluster_stats_cache[cache_key_def]['df']
                season_avg_per36_def = cluster_stats_cache[cache_key_def]['season_avg']

                if df_off_cluster.empty or df_def_cluster.empty:
                    continue

                # Filter by opponent
                df_off_vs_team = df_off_cluster[df_off_cluster['OPP_TEAM'] == opponent]
                df_def_vs_team = df_def_cluster[df_def_cluster['OPP_TEAM'] == opponent]

                df_off_vs_team_valid = df_off_vs_team[df_off_vs_team['MIN'] >= MIN_THRESHOLD]
                df_def_vs_team_valid = df_def_vs_team[df_def_vs_team['MIN'] >= MIN_THRESHOLD]

                # Calculate Bayesian percentage differences
                prior_mean = 0.0
                prior_weight = 3.0

                pct_diff_off = pd.Series(0.0, index=offensive_stats)
                pct_diff_def = pd.Series(0.0, index=defensive_stats)
                has_off_history = False
                has_def_history = False

                # Offensive cluster Bayesian calculation
                if not df_off_vs_team_valid.empty:
                    has_off_history = True
                    df_off_vs_team_valid_sorted = df_off_vs_team_valid.sort_values('GAME_DATE')
                    df_off_vs_team_per36 = df_off_vs_team_valid_sorted[offensive_stats].div(
                        df_off_vs_team_valid_sorted['MIN'], axis=0) * 36

                    game_pct_diffs_off = (
                            (df_off_vs_team_per36 - season_avg_per36_off[offensive_stats]) /
                            season_avg_per36_off[offensive_stats] * 100
                    )

                    for stat in offensive_stats:
                        pct_diff_off[stat] = bayesian_pct_diff(game_pct_diffs_off[stat].values, stat)
                    # Projected 3PM = Projected 3PA × player season 3P%
                    if 'FG3A' in pct_diff_off.index:
                        pct_diff_off['FG3M'] = pct_diff_off['FG3A']

                # Defensive cluster Bayesian calculation
                if not df_def_vs_team_valid.empty:
                    has_def_history = True
                    df_def_vs_team_valid_sorted = df_def_vs_team_valid.sort_values('GAME_DATE')
                    df_def_vs_team_per36 = df_def_vs_team_valid_sorted[defensive_stats].div(
                        df_def_vs_team_valid_sorted['MIN'], axis=0) * 36

                    game_pct_diffs_def = (
                            (df_def_vs_team_per36 - season_avg_per36_def[defensive_stats]) /
                            season_avg_per36_def[defensive_stats] * 100
                    )

                    for stat in defensive_stats:
                        pct_diff_def[stat] = bayesian_pct_diff(game_pct_diffs_def[stat].values, stat)

                # Only include players with historical data vs this opponent
                if has_off_history or has_def_history:
                    player_rows = merged[merged['PLAYER_ID'] == player_id]
                    off_cluster_name = player_rows['OffClusterName'].iloc[0] if not player_rows.empty else ''
                    def_cluster_name = player_rows['DefClusterName'].iloc[0] if not player_rows.empty else ''
                    if pd.isna(off_cluster_name): off_cluster_name = ''
                    if pd.isna(def_cluster_name): def_cluster_name = ''
                    matchup_scores.append({
                        'PLAYER_NAME': player_name,
                        'PLAYER_ID': player_id,
                        'TEAM': player_team,
                        'Opponent': opponent,
                        'Home': '🏠' if is_home else '✈️',
                        'Off Cluster': int(off_cluster),
                        'Off Cluster Name': off_cluster_name,
                        'Def Cluster': int(def_cluster),
                        'Def Cluster Name': def_cluster_name,
                        'Has Off History': has_off_history,
                        'Has Def History': has_def_history,
                        'PTS %': pct_diff_off['PTS'] if has_off_history else 0,
                        'AST %': pct_diff_off['AST'] if has_off_history else 0,
                        'OREB %': pct_diff_off['OREB'] if has_off_history else 0,
                        'DREB %': pct_diff_def['DREB'] if has_def_history else 0,
                        'FG3M %': pct_diff_off['FG3M'] if has_off_history else 0,
                        'FG3A %': pct_diff_off['FG3A'] if has_off_history else 0,
                        'STL %': pct_diff_def['STL'] if has_def_history else 0,
                        'BLK %': pct_diff_def['BLK'] if has_def_history else 0,
                        'Off Games': len(df_off_vs_team_valid),
                        'Def Games': len(df_def_vs_team_valid),
                        'pct_diff_off': pct_diff_off,
                        'pct_diff_def': pct_diff_def
                    })

            except Exception as e:
                # Skip individual players that cause errors
                print(f"Error processing player {player_row.get('PLAYER_NAME', 'Unknown')}: {e}")
                continue

        if not matchup_scores:
            return None

        matchups_df = pd.DataFrame(matchup_scores)
        matchups_df = matchups_df.sort_values('PTS %', ascending=False)

        return matchups_df

    except KeyError as e:
        st.error(f"❌ Missing required data column: {str(e)}")
        st.info("Check that your merged data has all required stats columns")
        return None

    except Exception as e:
        st.error(f"❌ Error calculating matchups: {str(e)}")
        st.exception(e)  # Shows full traceback in Streamlit
        return None

def get_scoreboard():
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


def build_player_list():
    matchups = get_scoreboard()
    playeroutput = pd.DataFrame()
    for matchup in matchups:
        awayid, awayabb, homeid, homeabb = matchup[0], matchup[1], matchup[2], matchup[3]
        time.sleep(1)
        awayroster = commonteamroster.CommonTeamRoster(team_id=awayid, season=WNBA_CURRENT_SEASON, league_id_nullable=WNBA_LEAGUE_ID)
        time.sleep(1)
        homeroster = commonteamroster.CommonTeamRoster(team_id=homeid, season=WNBA_CURRENT_SEASON, league_id_nullable=WNBA_LEAGUE_ID)
        awayroster = pd.DataFrame(awayroster.get_data_frames()[0].PLAYER_ID)
        awayroster['Home'] = False
        awayroster['OPP'] = str([homeabb])
        awayroster['TEAM'] = str([awayabb])
        homeroster = pd.DataFrame(homeroster.get_data_frames()[0].PLAYER_ID)
        homeroster['Home'] = True
        homeroster['OPP'] = str([awayabb])
        homeroster['TEAM'] = str([homeabb])
        playeroutput = pd.concat([playeroutput, awayroster, homeroster]).drop_duplicates()
    return playeroutput


def get_playtype_stats(season=WNBA_CURRENT_SEASON):
    """Get synergy play type stats. NOTE: Synergy data is NBA-only; returns None for WNBA."""

    # Define play types to pull
    offensive_types = ['PRBallHandler', 'PRRollman', 'Postup', 'Spotup', 'OffScreen', 'Cut']
    defensive_types = ['PRBallHandler', 'PRRollman', 'Postup']

    # Storage for all raw dataframes
    offensive_data = {}
    defensive_data = {}

    max_retries = 3

    # Get all offensive stats first
    for play_type in offensive_types:
        for attempt in range(max_retries):
            try:
                time.sleep(1)
                print(f"Getting offensive {play_type} stats for {season}... (attempt {attempt + 1}/{max_retries})")

                offensive_data[play_type] = SynergyPlayTypes(
                    season=season,
                    season_type_all_star='Regular Season',
                    per_mode_simple='PerGame',
                    play_type_nullable=play_type,
                    player_or_team_abbreviation='P',
                    type_grouping_nullable='offensive'
                ).get_data_frames()[0]
                break
            except Exception as e:
                print(f"Error getting offensive {play_type} stats (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to get offensive {play_type} stats after all retries")
                    return None, None, None
                print("Waiting 60 seconds before retry...")
                time.sleep(60)

    # Get all defensive stats
    for play_type in defensive_types:
        for attempt in range(max_retries):
            try:
                time.sleep(1)
                print(f"Getting defensive {play_type} stats for {season}... (attempt {attempt + 1}/{max_retries})")

                defensive_data[play_type] = SynergyPlayTypes(
                    season=season,
                    season_type_all_star='Regular Season',
                    per_mode_simple='PerGame',
                    play_type_nullable=play_type,
                    player_or_team_abbreviation='P',
                    type_grouping_nullable='defensive'
                ).get_data_frames()[0]
                break
            except Exception as e:
                print(f"Error getting defensive {play_type} stats (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to get defensive {play_type} stats after all retries")
                    return None, None, None
                print("Waiting 60 seconds before retry...")
                time.sleep(60)

    # Now extract just the frequencies we need
    try:
        all_stats = []

        # Extract offensive frequencies
        for play_type, df in offensive_data.items():
            stats = df[['PLAYER_ID', 'POSS_PCT']].copy()
            stats.rename(columns={'POSS_PCT': f'OFF_{play_type.upper()}_FREQ'}, inplace=True)
            all_stats.append(stats)

        # Extract defensive frequencies
        for play_type, df in defensive_data.items():
            stats = df[['PLAYER_ID', 'POSS_PCT']].copy()
            stats.rename(columns={'POSS_PCT': f'DEF_{play_type.upper()}_FREQ'}, inplace=True)
            all_stats.append(stats)

        # Merge all frequencies together
        combined = all_stats[0]
        for stats in all_stats[1:]:
            combined = combined.merge(stats, on='PLAYER_ID', how='outer')

        # Return both the combined frequencies AND the raw data if you need it later
        return combined, offensive_data, defensive_data

    except Exception as e:
        print(f"Error merging play type stats: {e}")
        return None, None, None
def get_shot_chart_data(player_id, season=WNBA_CURRENT_SEASON, max_retries=3):
    """Get detailed shot chart data for a player."""
    for attempt in range(max_retries):
        try:
            time.sleep(5)
            player_id = str(int(float(player_id)))
            print(f"Getting shot chart for {player_id} (attempt {attempt + 1}/{max_retries})")

            shot_chart = ShotChartDetail(
                player_id=player_id,
                team_id=0,
                season_nullable=WNBA_CURRENT_SEASON,
                season_type_all_star='Regular Season',
                context_measure_simple='FGA',
                league_id=WNBA_LEAGUE_ID
            ).get_data_frames()[0]

            # Calculate zone-based metrics
            shot_profile = {
                'PLAYER_ID': player_id,
                'SEASON_ID': season
            }

            # Get actual games played
            games_played = shot_chart['GAME_ID'].nunique() if len(shot_chart) > 0 else 0

            if games_played == 0:
                print(f"No games found for player {player_id}")
                return None

            # Paint shots
            paint_shots = shot_chart[shot_chart['SHOT_ZONE_BASIC'].isin(['Restricted Area', 'In The Paint (Non-RA)'])]
            shot_profile['paint_shots_per_game'] = len(paint_shots) / games_played
            shot_profile['paint_fg_pct'] = paint_shots['SHOT_MADE_FLAG'].mean() if len(paint_shots) > 0 else 0

            # Corner 3s
            corner_3s = shot_chart[shot_chart['SHOT_ZONE_BASIC'].isin(['Corner 3', 'Left Corner 3', 'Right Corner 3'])]
            shot_profile['corner_3_per_game'] = len(corner_3s) / games_played
            shot_profile['corner_3_pct'] = corner_3s['SHOT_MADE_FLAG'].mean() if len(corner_3s) > 0 else 0

            # Mid-range
            midrange = shot_chart[shot_chart['SHOT_ZONE_BASIC'] == 'Mid-Range']
            shot_profile['midrange_per_game'] = len(midrange) / games_played
            shot_profile['midrange_pct'] = midrange['SHOT_MADE_FLAG'].mean() if len(midrange) > 0 else 0

            # Above break 3s
            above_break_3s = shot_chart[shot_chart['SHOT_ZONE_BASIC'] == 'Above the Break 3']
            shot_profile['above_break_3_per_game'] = len(above_break_3s) / games_played
            shot_profile['above_break_3_pct'] = above_break_3s['SHOT_MADE_FLAG'].mean() if len(
                above_break_3s) > 0 else 0

            # Distance metrics
            shot_profile['avg_shot_distance'] = shot_chart['SHOT_DISTANCE'].mean()
            shot_profile['max_shot_distance'] = shot_chart['SHOT_DISTANCE'].max()

            # Shot clock analysis
            if 'SHOT_CLOCK' in shot_chart.columns:
                valid_shot_clock = shot_chart['SHOT_CLOCK'].notna()
                shot_profile['avg_shot_clock'] = shot_chart.loc[valid_shot_clock, 'SHOT_CLOCK'].mean()
                shot_profile['early_shot_clock_freq'] = (
                    len(shot_chart[shot_chart['SHOT_CLOCK'] >= 15]) / len(shot_chart)
                    if len(shot_chart) > 0 else 0
                )

            # Pressure shots
            if 'PERIOD' in shot_chart.columns and 'MINUTES_REMAINING' in shot_chart.columns:
                clutch_shots = shot_chart[
                    (shot_chart['PERIOD'] >= 4) &
                    (shot_chart['MINUTES_REMAINING'] <= 5)
                    ]
                shot_profile['clutch_fg_pct'] = clutch_shots['SHOT_MADE_FLAG'].mean() if len(clutch_shots) > 0 else 0

            # Add metadata
            shot_profile['games_played'] = games_played
            shot_profile['total_shots'] = len(shot_chart)

            return pd.DataFrame([shot_profile])

        except Exception as e:
            print(f"Error on attempt {attempt + 1} for player {player_id}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 60  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts for player {player_id}")
                return None


def get_hustle_stats(season=WNBA_CURRENT_SEASON):
    """Get hustle stats for all players - contested shots and deflections."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            print(f"Getting hustle stats... (attempt {attempt + 1}/{max_retries})")

            hustle = LeagueHustleStatsPlayer(
                season=season,
                season_type_all_star='Regular Season',
                league_id=WNBA_LEAGUE_ID,
            ).get_data_frames()[0]

            # Select relevant columns
            hustle_stats = hustle[[
                'PLAYER_ID',
                'PLAYER_NAME',
                'CONTESTED_SHOTS_2PT',
                'CONTESTED_SHOTS_3PT',
                'DEFLECTIONS',
                'G',
                'MIN'  # Games played for reference
            ]].copy()

            hustle_stats['CONTESTED_SHOTS_2PT_PER36'] = (hustle_stats['CONTESTED_SHOTS_2PT'] / hustle_stats['MIN']) * 36
            hustle_stats['CONTESTED_SHOTS_3PT_PER36'] = (hustle_stats['CONTESTED_SHOTS_3PT'] / hustle_stats['MIN']) * 36
            hustle_stats['DEFLECTIONS_PER36'] = (hustle_stats['DEFLECTIONS'] / hustle_stats['MIN']) * 36

            # Select final columns
            hustle_stats_per36 = hustle_stats[[
                'PLAYER_ID',
                'PLAYER_NAME',
                'CONTESTED_SHOTS_2PT_PER36',
                'CONTESTED_SHOTS_3PT_PER36',
                'DEFLECTIONS_PER36'
            ]]

            return hustle_stats_per36

        except Exception as e:
            print(f"Error getting hustle stats (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Failed to get hustle stats after all retries")
                return None
            print("Waiting 60 seconds before retry...")
            time.sleep(60)
def get_tracking_stats(season=WNBA_CURRENT_SEASON):
    """Get tracking stats - speed, distance, catch & shoot, passing, drives, and rebounding."""

    # Speed/Distance tracking
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            print(f"Getting speed/distance stats... (attempt {attempt + 1}/{max_retries})")
            tracking = LeagueDashPtStats(
                season=season,
                per_mode_simple='PerGame',
                pt_measure_type='SpeedDistance',
                player_or_team='Player',
                league_id=WNBA_LEAGUE_ID
            ).get_data_frames()[0]

            tracking_stats = tracking[[
                'PLAYER_ID',
                'PLAYER_NAME',
                'AVG_SPEED_OFF',
                'AVG_SPEED_DEF',
            ]].copy()
            break
        except Exception as e:
            print(f"Error getting speed/distance stats (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Failed to get speed/distance stats after all retries")
                return None
            print("Waiting 60 seconds before retry...")
            time.sleep(60)

    # Catch & Shoot
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            print(f"Getting catch & shoot stats... (attempt {attempt + 1}/{max_retries})")
            cs = LeagueDashPtStats(
                season=season,
                per_mode_simple='PerGame',
                pt_measure_type='CatchShoot',
                player_or_team='Player',
                league_id=WNBA_LEAGUE_ID
            ).get_data_frames()[0]

            csstats = cs[[
                'PLAYER_ID',
                'CATCH_SHOOT_FGA',
                'CATCH_SHOOT_FG3A',
                'MIN'
            ]].copy()

            csstats['CATCHANDSHOOTPER36'] = (csstats['CATCH_SHOOT_FGA'] / csstats['MIN']) * 36
            csstats['CATCHANDSHOOT3PAPER36'] = (csstats['CATCH_SHOOT_FG3A'] / csstats['MIN']) * 36
            csstats['CATCHANDSHOOT3PAPER36'] = csstats['CATCHANDSHOOT3PAPER36'].fillna(0)
            break
        except Exception as e:
            print(f"Error getting catch & shoot stats (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Failed to get catch & shoot stats after all retries")
                return None
            print("Waiting 60 seconds before retry...")
            time.sleep(60)

    # Passing
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            print(f"Getting passing stats... (attempt {attempt + 1}/{max_retries})")
            passing = LeagueDashPtStats(
                season=season,
                per_mode_simple='PerGame',
                pt_measure_type='Passing',
                player_or_team='Player',
                league_id=WNBA_LEAGUE_ID
            ).get_data_frames()[0]

            passingstats = passing[[
                'PLAYER_ID',
                'PASSES_MADE',
                'PASSES_RECEIVED',
                'MIN'
            ]].copy()

            passingstats['PASSESMADEPER36'] = (passingstats['PASSES_MADE'] / passingstats['MIN']) * 36
            passingstats['PASSESRECEIVEDPER36'] = (passingstats['PASSES_RECEIVED'] / passingstats['MIN']) * 36
            break
        except Exception as e:
            print(f"Error getting passing stats (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Failed to get passing stats after all retries")
                return None
            print("Waiting 60 seconds before retry...")
            time.sleep(60)

    # Drives
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            print(f"Getting drives stats... (attempt {attempt + 1}/{max_retries})")
            drives = LeagueDashPtStats(
                season=season,
                per_mode_simple='PerGame',
                pt_measure_type='Drives',
                player_or_team='Player',
                league_id=WNBA_LEAGUE_ID
            ).get_data_frames()[0]

            drivestats = drives[[
                'PLAYER_ID',
                'DRIVES',
                'MIN',
                'DRIVE_PTS_PCT',
                'DRIVE_AST_PCT',
            ]].copy()
            drivestats['DRIVESPER36'] = (drivestats['DRIVES'] / drivestats['MIN']) * 36
            break
        except Exception as e:
            print(f"Error getting drives stats (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Failed to get drives stats after all retries")
                return None
            print("Waiting 60 seconds before retry...")
            time.sleep(60)

    # Rebounding (both offensive and defensive)
    for attempt in range(max_retries):
        try:
            time.sleep(1)
            print(f"Getting rebounding stats... (attempt {attempt + 1}/{max_retries})")
            rebounding = LeagueDashPtStats(
                season=season,
                per_mode_simple='PerGame',
                pt_measure_type='Rebounding',
                player_or_team='Player',
                league_id=WNBA_LEAGUE_ID
            ).get_data_frames()[0]

            reboundingstats = rebounding[[
                'PLAYER_ID',
                'AVG_OREB_DIST',
                'AVG_DREB_DIST'
            ]].copy()
            break
        except Exception as e:
            print(f"Error getting rebounding stats (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("Failed to get rebounding stats after all retries")
                return None
            print("Waiting 60 seconds before retry...")
            time.sleep(60)

    # Merge all tracking stats together
    try:
        combined = tracking_stats.merge(
            csstats[['PLAYER_ID', 'CATCHANDSHOOTPER36', 'CATCHANDSHOOT3PAPER36']],
            on='PLAYER_ID',
            how='left'
        ).merge(
            passingstats[['PLAYER_ID', 'PASSESMADEPER36', 'PASSESRECEIVEDPER36']],
            on='PLAYER_ID',
            how='left'
        ).merge(
            drivestats[['PLAYER_ID', 'DRIVE_PTS_PCT', 'DRIVESPER36', 'DRIVE_AST_PCT']],
            on='PLAYER_ID',
            how='left'
        ).merge(
            reboundingstats[['PLAYER_ID', 'AVG_OREB_DIST', 'AVG_DREB_DIST']],
            on='PLAYER_ID',
            how='left'
        )

        return combined
    except Exception as e:
        print(f"Error merging tracking stats: {e}")
        return None
def enhance_player_data(player_averages, min_games=10):
    """Add shot chart, tracking, hustle, and play type data to player averages."""

    all_shot_data = pd.DataFrame()
    qualified_players = player_averages

    total_players = len(qualified_players)
    print(f"Processing {total_players} qualified players...")

    # GET SHOT CHART DATA (existing code - unchanged)
    all_shot_data = []
    failed_players = []

    for idx, player in qualified_players.iterrows():
        print(f"\nProcessing player {idx + 1}/{total_players}: {player['PLAYER_NAME']} (ID: {player['PLAYER_ID']})")

        shot_data = get_shot_chart_data(player['PLAYER_ID'], player['SEASON_ID'])

        if shot_data is not None:
            all_shot_data.append(shot_data)
        else:
            failed_players.append({
                'PLAYER_ID': player['PLAYER_ID'],
                'PLAYER_NAME': player['PLAYER_NAME']
            })

        time.sleep(1)

    # Combine all data
    if all_shot_data:
        all_shot_data = pd.concat(all_shot_data, ignore_index=True)
        print(f"\nSuccessfully processed {len(all_shot_data)} players")
    else:
        all_shot_data = pd.DataFrame()
        print("\nNo shot data collected")

    # Report failures
    if failed_players:
        print(f"\nFailed to get data for {len(failed_players)} players:")
        for p in failed_players[:10]:  # Show first 10
            print(f"  - {p['PLAYER_NAME']} ({p['PLAYER_ID']})")

    # Season ID mapping: WNBA seasons use single-year format
    season_id_map = {s: int('1' + s) for s in WNBA_SEASONS}  # e.g. '2025' → 12025

    # ============ GET TRACKING STATS FOR EACH SEASON ============
    print("\nGetting tracking stats...")
    tracking_frames = []
    for season in WNBA_SEASONS:
        ts = get_tracking_stats(season=season)
        if ts is not None:
            ts['SEASON_ID'] = season_id_map[season]
            tracking_frames.append(ts)
    tracking_stats = pd.concat(tracking_frames, ignore_index=True) if tracking_frames else None
    if tracking_stats is not None:
        print(f"✓ Combined tracking stats: {len(tracking_stats)} records")
    else:
        print("⚠ Tracking stats not available for WNBA — skipping")

    # ============ GET HUSTLE STATS FOR EACH SEASON ============
    print("\nGetting hustle stats...")
    hustle_frames = []
    for season in WNBA_SEASONS:
        hs = get_hustle_stats(season=season)
        if hs is not None:
            hs['SEASON_ID'] = season_id_map[season]
            hustle_frames.append(hs)
    hustle_stats = pd.concat(hustle_frames, ignore_index=True) if hustle_frames else None
    if hustle_stats is not None:
        print(f"✓ Combined hustle stats: {len(hustle_stats)} records")
    else:
        print("⚠ Hustle stats not available for WNBA — skipping")

    # ============ GET PLAY TYPE STATS FOR EACH SEASON ============
    print("\nGetting play type stats...")
    playtype_frames = []
    for season in WNBA_SEASONS:
        result = get_playtype_stats(season=season)
        if result[0] is not None:
            pt = result[0]
            pt['SEASON_ID'] = season_id_map[season]
            playtype_frames.append(pt)
    playtype_stats = pd.concat(playtype_frames, ignore_index=True) if playtype_frames else None
    if playtype_stats is not None:
        print(f"✓ Combined play type stats: {len(playtype_stats)} records")
    else:
        print("⚠ Play type stats not available for WNBA — skipping")

    # ============ MERGE EVERYTHING TOGETHER ============
    qualified_players = qualified_players.fillna(0)
    qualified_players[['PLAYER_ID', 'SEASON_ID']] = qualified_players[['PLAYER_ID', 'SEASON_ID']].astype(int)

    # Merge shot chart data
    if not all_shot_data.empty:
        all_shot_data[['PLAYER_ID', 'SEASON_ID']] = all_shot_data[['PLAYER_ID', 'SEASON_ID']].astype(int)
        enhanced_data = qualified_players.merge(all_shot_data, on=['PLAYER_ID', 'SEASON_ID'], how='left')
    else:
        enhanced_data = qualified_players.copy()

    # Merge tracking stats
    if tracking_stats is not None:
        tracking_stats['PLAYER_ID'] = tracking_stats['PLAYER_ID'].astype(int)
        tracking_stats['SEASON_ID'] = tracking_stats['SEASON_ID'].astype(int)
        enhanced_data = enhanced_data.merge(
            tracking_stats.drop(columns=['PLAYER_NAME'], errors='ignore'),
            on=['PLAYER_ID', 'SEASON_ID'], how='left'
        )
        print(f"✓ Merged tracking stats")

    # Merge hustle stats
    if hustle_stats is not None:
        hustle_stats['PLAYER_ID'] = hustle_stats['PLAYER_ID'].astype(int)
        hustle_stats['SEASON_ID'] = hustle_stats['SEASON_ID'].astype(int)
        enhanced_data = enhanced_data.merge(
            hustle_stats.drop(columns=['PLAYER_NAME'], errors='ignore'),
            on=['PLAYER_ID', 'SEASON_ID'], how='left'
        )
        print(f"✓ Merged hustle stats")

    # Merge play type stats
    if playtype_stats is not None:
        playtype_stats['PLAYER_ID'] = playtype_stats['PLAYER_ID'].astype(int)
        playtype_stats['SEASON_ID'] = playtype_stats['SEASON_ID'].astype(int)
        enhanced_data = enhanced_data.merge(
            playtype_stats, on=['PLAYER_ID', 'SEASON_ID'], how='left'
        )
        print(f"✓ Merged play type stats")

    print(f"\n✓ Final enhanced dataset: {len(enhanced_data)} records with {len(enhanced_data.columns)} features")

    return enhanced_data


def cluster_players_off(data, n_clusters):
    """Perform k-means clustering on players based on their stats and shooting profiles."""
    data = data[(data['MIN'] > 10) & (data['SEASON_ID'].astype(str).str.startswith('2'))].copy()

    required_features = [
        'PTS_per36', 'AST_per36', 'OREB_per36', 'DREB_per36', 'TOV_per36',
        'FG%', 'FG3%', 'FT%',
        'WEIGHT', 'Height_IN',
    ]
    optional_features = [
        'paint_shots_per_game_per36', 'paint_fg_pct',
        'corner_3_per_game_per36', 'corner_3_pct',
        'midrange_per_game_per36', 'midrange_pct',
        'above_break_3_per_game_per36', 'above_break_3_pct',
        'avg_shot_distance',
        'AVG_SPEED_OFF',
        'CATCHANDSHOOTPER36', 'CATCHANDSHOOT3PAPER36',
        'PASSESMADEPER36', 'PASSESRECEIVEDPER36',
        'DRIVESPER36', 'DRIVE_PTS_PCT', 'DRIVE_AST_PCT',
        'AVG_OREB_DIST',
        'OFF_PRBALLHANDLER_FREQ', 'OFF_PRROLLMAN_FREQ', 'OFF_POSTUP_FREQ',
        'OFF_SPOTUP_FREQ', 'OFF_OFFSCREEN_FREQ', 'OFF_CUT_FREQ',
    ]

    data['Height_IN'] = data['HEIGHT'].str.split('-').apply(lambda x: int(x[0]) * 12 + int(x[1]) if isinstance(x, list) and len(x) == 2 else None)

    minutes_factor = 36 / data['MIN'].clip(lower=1)
    per36_cols = ['PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'paint_shots_per_game', 'corner_3_per_game',
                  'midrange_per_game', 'above_break_3_per_game']
    for col in per36_cols:
        if col in data.columns:
            data[f'{col}_per36'] = data[col] * minutes_factor

    # Use only optional features that are actually present in the data
    available_optional = [f for f in optional_features if f in data.columns]
    features = required_features + available_optional

    clean_data = data.dropna(subset=features)

    print(f"\n{'=' * 60}")
    print(f"OFFENSIVE CLUSTERING FEATURES ({len(features)} total)")
    print(f"{'=' * 60}")
    print("Required: PTS, AST, OREB, DREB, TOV, FG%, FG3%, FT%, Height, Weight")
    print(f"Optional present: {available_optional}")
    print(f"{'=' * 60}\n")

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_data[features])
    if n_clusters is None:
        from sklearn.metrics import silhouette_score

        # Initialize plots for both methods
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        min_clusters = 6
        max_clusters = 12
        # Lists to store results
        inertia_values = []
        silhouette_scores = []

        print("Determining optimal number of clusters...")

        # Try different numbers of clusters
        for k in range(min_clusters, max_clusters + 1):
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)

            # Get inertia (for Elbow Method)
            inertia_values.append(kmeans.inertia_)

            # Get silhouette score
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            print(f"  Clusters: {k}, Inertia: {kmeans.inertia_:.0f}, Silhouette Score: {silhouette_avg:.3f}")

        # Plot Elbow Method
        ax1.plot(range(min_clusters, max_clusters + 1), inertia_values, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')

        # Calculate the "elbow" point using second derivatives
        deltas = np.diff(inertia_values)
        second_deltas = np.diff(deltas)
        elbow_index = np.argmax(second_deltas) + 1  # +1 due to double differentiation
        elbow_value = min_clusters + elbow_index

        ax1.axvline(x=elbow_value, color='r', linestyle='--', alpha=0.7)
        ax1.text(elbow_value + 0.1, ax1.get_ylim()[0] + 0.7 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                 f'Elbow point: {elbow_value}', color='r')

        # Plot Silhouette Analysis
        ax2.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')

        # Find optimal clusters from silhouette
        min_silhouette_improvement = 0.01  # Only add clusters if silhouette improves by at least 0.02

        # Find optimal with improvement threshold
        optimal_clusters_sil = min_clusters
        peak_score = silhouette_scores[0]

        for i, score in enumerate(silhouette_scores[1:], start=1):
            if score > peak_score + min_silhouette_improvement:
                optimal_clusters_sil = min_clusters + i
                peak_score = score

        # Update visualization to show threshold-adjusted optimal
        ax2.axvline(x=optimal_clusters_sil, color='r', linestyle='--', alpha=0.7)
        ax2.text(optimal_clusters_sil + 0.1, ax2.get_ylim()[0] + 0.7 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]),
                 f'Optimal: {optimal_clusters_sil}', color='r')

        plt.tight_layout()
        plt.show()

        # For interpretability in basketball, prefer fewer clusters when close
        if abs(optimal_clusters_sil - elbow_value) <= 1:
            n_clusters = min(optimal_clusters_sil, elbow_value)
            print(f"Elbow and Silhouette methods agree (within 1): choosing {n_clusters} clusters")
        else:
            n_clusters = optimal_clusters_sil
            print(f"Note: Elbow method suggests {elbow_value} clusters, "
                  f"while Silhouette suggests {optimal_clusters_sil}. "
                  f"Choosing {n_clusters} based on Silhouette score with improvement threshold.")

        print(f"\nFinal selection: {n_clusters} clusters")
        print(f"Optimal number of clusters: {n_clusters}")

    # Perform k-means clustering with the determined number of clusters

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add cluster labels to the dataset
    result_data = clean_data.copy()
    result_data['Cluster'] = clusters
    result_data = result_data[result_data['SEASON_ID'].astype(str).str.startswith('2')]
    latest_season = result_data['SEASON_ID'].max()
    result_data = result_data[result_data['SEASON_ID'] == latest_season]
    # Create cluster visualization
    plt.figure(figsize=(12, 8))

    # Create a scatter plot using the first two principal components
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis')
    plt.title('Player Clusters Based on Performance and Shooting Profile')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Add player names for some key points
    for idx, player in enumerate(result_data['PLAYER_NAME']):
        if result_data.iloc[idx]['PTS'] > 20:  # Label high scorers
            plt.annotate(player, (scaled_features[idx, 0], scaled_features[idx, 1]))

    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    plt.figure(figsize=(12, 8))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    plt.title('Player Clusters Using PCA (Performance and Shooting Profile)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

    # Add player names for key players
    for idx, player in enumerate(result_data['PLAYER_NAME']):
        if result_data.iloc[idx]['PTS'] > 20 or result_data.iloc[idx]['AST'] > 7:
            plt.annotate(player, (principal_components[idx, 0], principal_components[idx, 1]))

    plt.tight_layout()
    plt.show()

    # Print cluster summaries with more detailed information
    print("\nCluster Summaries:")
    for i in range(n_clusters):
        cluster_players = result_data[result_data['Cluster'] == i]

        print(f"\nCluster {i} ({len(cluster_players)} players):")

        # Top players by different metrics
        print("Top Scorers:", ", ".join(cluster_players.nlargest(3, 'PTS')['PLAYER_NAME'].tolist()))
        if 'AST' in cluster_players.columns:
            print("Top Playmakers:", ", ".join(cluster_players.nlargest(3, 'AST')['PLAYER_NAME'].tolist()))

        # Shooting profile
        print(f"Avg Points: {cluster_players['PTS'].mean():.1f}")
        print(f"Avg 3P%: {cluster_players['FG3%'].mean():.3f}")

        # Shot distribution
        if 'paint_shots_per_game' in cluster_players.columns and 'above_break_3_per_game' in cluster_players.columns:
            paint_pct = cluster_players['paint_shots_per_game'].mean() / (
                    cluster_players['paint_shots_per_game'].mean() +
                    cluster_players['midrange_per_game'].mean() +
                    cluster_players['above_break_3_per_game'].mean() +
                    cluster_players['corner_3_per_game'].mean()) * 100
            print(f"Paint Shot %: {paint_pct:.1f}%")

    return result_data


def cluster_players_def(data, n_clusters):
    """Perform k-means clustering on players based on their defensive stats."""
    data = data[(data['MIN'] > 10) & (data['SEASON_ID'].astype(str).str.startswith('2'))].copy()

    required_features = [
        'DREB_per36',
        'STL_per36',
        'BLK_per36',
        'WEIGHT',
        'Height_IN',
    ]
    optional_features = [
        'CONTESTED_SHOTS_2PT_PER36',
        'CONTESTED_SHOTS_3PT_PER36',
        'DEFLECTIONS_PER36',
        'AVG_SPEED_DEF',
        'AVG_DREB_DIST',
        'DEF_PRBALLHANDLER_FREQ',
        'DEF_PRROLLMAN_FREQ',
        'DEF_POSTUP_FREQ',
    ]

    data['Height_IN'] = data['HEIGHT'].str.split('-').apply(lambda x: int(x[0]) * 12 + int(x[1]) if isinstance(x, list) and len(x) == 2 else None)

    minutes_factor = 36 / data['MIN'].clip(lower=1)
    per36_cols = ['PTS', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'paint_shots_per_game', 'corner_3_per_game',
                  'midrange_per_game', 'above_break_3_per_game']
    for col in per36_cols:
        if col in data.columns:
            data[f'{col}_per36'] = data[col] * minutes_factor

    available_optional = [f for f in optional_features if f in data.columns]
    features = required_features + available_optional

    clean_data = data.dropna(subset=features)

    print(f"\n{'=' * 60}")
    print(f"DEFENSIVE CLUSTERING FEATURES ({len(features)} total)")
    print(f"{'=' * 60}")
    print("Required: DREB, STL, BLK, Height, Weight")
    print(f"Optional present: {available_optional}")
    print(f"{'=' * 60}\n")

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_data[features])
    if n_clusters is None:
        from sklearn.metrics import silhouette_score

        # Initialize plots for both methods
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        min_clusters = 6
        max_clusters = 12
        # Lists to store results
        inertia_values = []
        silhouette_scores = []

        print("Determining optimal number of clusters...")

        # Try different numbers of clusters
        for k in range(min_clusters, max_clusters + 1):
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)

            # Get inertia (for Elbow Method)
            inertia_values.append(kmeans.inertia_)

            # Get silhouette score
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            print(f"  Clusters: {k}, Inertia: {kmeans.inertia_:.0f}, Silhouette Score: {silhouette_avg:.3f}")

        # Plot Elbow Method
        ax1.plot(range(min_clusters, max_clusters + 1), inertia_values, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')

        # Calculate the "elbow" point using second derivatives
        deltas = np.diff(inertia_values)
        second_deltas = np.diff(deltas)
        elbow_index = np.argmax(second_deltas) + 1  # +1 due to double differentiation
        elbow_value = min_clusters + elbow_index

        ax1.axvline(x=elbow_value, color='r', linestyle='--', alpha=0.7)
        ax1.text(elbow_value + 0.1, ax1.get_ylim()[0] + 0.7 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                 f'Elbow point: {elbow_value}', color='r')

        # Plot Silhouette Analysis
        ax2.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')

        # Find optimal clusters from silhouette
        optimal_clusters_sil = min_clusters + np.argmax(silhouette_scores)
        ax2.axvline(x=optimal_clusters_sil, color='r', linestyle='--', alpha=0.7)
        ax2.text(optimal_clusters_sil + 0.1, ax2.get_ylim()[0] + 0.7 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]),
                 f'Optimal: {optimal_clusters_sil}', color='r')

        plt.tight_layout()
        plt.show()

        # Choose the optimal number of clusters
        # If silhouette and elbow methods disagree, prefer silhouette
        if abs(optimal_clusters_sil - elbow_value) <= 1:
            n_clusters = optimal_clusters_sil
        else:
            # If they disagree by more than 1, take silhouette with a note
            n_clusters = min(optimal_clusters_sil,elbow_value)
            print(f"Note: Elbow method suggests {elbow_value} clusters, "
                  f"while Silhouette suggests {optimal_clusters_sil}. "
                  f"Choosing {n_clusters} based on Silhouette score.")

        print(f"Optimal number of clusters: {n_clusters}")

    # Perform k-means clustering with the determined number of clusters

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add cluster labels to the dataset
    result_data = clean_data.copy()
    result_data['Cluster'] = clusters
    result_data = result_data[result_data['SEASON_ID'].astype(str).str.startswith('2')]
    latest_season = result_data['SEASON_ID'].max()
    result_data = result_data[result_data['SEASON_ID'] == latest_season]
    # Create cluster visualization
    plt.figure(figsize=(12, 8))

    # Create a scatter plot using the first two principal components
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis')
    plt.title('Player Clusters Based on Performance and Shooting Profile')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Add player names for some key points
    for idx, player in enumerate(result_data['PLAYER_NAME']):
        if result_data.iloc[idx]['PTS'] > 20:  # Label high scorers
            plt.annotate(player, (scaled_features[idx, 0], scaled_features[idx, 1]))

    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    plt.figure(figsize=(12, 8))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis')
    plt.title('Player Clusters Using PCA (Performance and Shooting Profile)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

    # Add player names for key players
    for idx, player in enumerate(result_data['PLAYER_NAME']):
        if result_data.iloc[idx]['PTS'] > 20 or result_data.iloc[idx]['AST'] > 7:
            plt.annotate(player, (principal_components[idx, 0], principal_components[idx, 1]))

    plt.tight_layout()
    plt.show()

    # Print cluster summaries with more detailed information
    print("\nCluster Summaries:")
    for i in range(n_clusters):
        cluster_players = result_data[result_data['Cluster'] == i]

        print(f"\nCluster {i} ({len(cluster_players)} players):")

        # Top players by different metrics
        print("Top Steals:", ", ".join(cluster_players.nlargest(3, 'STL')['PLAYER_NAME'].tolist()))
        if 'AST' in cluster_players.columns:
            print("Top Blocks:", ", ".join(cluster_players.nlargest(3, 'BLK')['PLAYER_NAME'].tolist()))

        # Shooting profile
        print(f"Avg Blocks: {cluster_players['BLK'].mean():.1f}")
        print(f"Avg Steals: {cluster_players['STL'].mean():.3f}")

        # Shot distribution
        if 'paint_shots_per_game' in cluster_players.columns and 'above_break_3_per_game' in cluster_players.columns:
            paint_pct = cluster_players['paint_shots_per_game'].mean() / (
                    cluster_players['paint_shots_per_game'].mean() +
                    cluster_players['midrange_per_game'].mean() +
                    cluster_players['above_break_3_per_game'].mean() +
                    cluster_players['corner_3_per_game'].mean()) * 100
            print(f"Paint Shot %: {paint_pct:.1f}%")

    return result_data


def get_player_box():
    season_types = ['Regular Season', 'Playoffs']
    frames = []
    for season in WNBA_SEASONS:
        for stype in season_types:
            time.sleep(1)
            try:
                df = LeagueGameLog(
                    player_or_team_abbreviation='P',
                    season_type_all_star=stype,
                    season=season,
                    league_id=WNBA_LEAGUE_ID
                ).get_data_frames()[0]
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"Failed to fetch {stype} {season}: {e}")
    playerbox = pandas.concat(frames, ignore_index=True)
    print(playerbox.columns)
    return playerbox


def box_to_avg(data):
    player_averages = data[['PLAYER_NAME', 'PLAYER_ID', 'SEASON_ID', 'MIN', 'FGM',
                            'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                            'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].groupby(
        ['PLAYER_NAME', 'PLAYER_ID', 'SEASON_ID']).mean().reset_index()
    player_averages['FG%'] = player_averages['FGM'] / player_averages['FGA']
    player_averages['FT%'] = player_averages['FTM'] / player_averages['FTA']
    player_averages['FG3%'] = player_averages['FG3M'] / player_averages['FG3A']
    return player_averages


def add_height_weight_pos(data):
    index = nba_api.stats.endpoints.playerindex.PlayerIndex(
        league_id=WNBA_LEAGUE_ID, historical_nullable=1
    ).get_data_frames()[0]
    index = index[['PERSON_ID', 'POSITION', 'HEIGHT', 'WEIGHT']].drop_duplicates('PERSON_ID')
    data = data.merge(index, left_on='PLAYER_ID', right_on='PERSON_ID', how='left')
    return data


def add_shot_loc_eff(data):
    print(SynergyPlayTypes(
        player_or_team_abbreviation='P',
        season=WNBA_CURRENT_SEASON,
        season_type_all_star='Regular Season'
    ).get_data_frames()[0])
    return


def add_advanced(data):  # ftr usage to ratio
    return


def basic_clustering(data):
    # Step 1: Generate or Load Data
    # Replace this with your actual dataset
    np.random.seed(42)
    data = np.random.rand(100, 2)  # 100 samples with 2 features

    # Step 2: Define Number of Clusters (k)
    num_clusters = 10

    # Step 3: Initialize and Fit the KMeans Model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)

    # Step 4: Access the Results
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Step 5: Visualize the Clusters (Optional for 2D data)
    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x', s=100, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    # Step 6: (Optional) Evaluate or Save Results
    # Print the cluster centers
    print("Cluster Centers:")
    print(cluster_centers)

    # Save the labels or other results as needed


def create_clusters():
    # data = get_player_box()
    # print("got data")
    # player_averages = box_to_avg(data)
    # print("converted to avg")
    # print(player_averages)
    # player_averages = add_height_weight_pos(player_averages)
    # print("added height weight")
    #
    # # ============ UPDATED: Now includes tracking and hustle stats ============
    # enhanced_data = enhance_player_data(player_averages)

    # Save the enhanced data with ALL features
    # enhanced_data.to_csv('player_data.csv', index=False)
    # print("\n✓ Saved enhanced player data to 'player_data.csv'")

    enhanced_data = pd.read_csv("player_data.csv").fillna(0)
    # Perform clustering with new features
    print("\n" + "=" * 60)
    print("STARTING OFFENSIVE CLUSTERING")
    print("=" * 60)
    clustered_data = cluster_players_off(enhanced_data, None)
    clustered_data.to_csv('player_clusters_detailed.csv', index=False)
    print("✓ Saved offensive clusters to 'player_clusters_detailed.csv'")

    print("\n" + "=" * 60)
    print("STARTING DEFENSIVE CLUSTERING")
    print("=" * 60)
    defcluster = cluster_players_def(enhanced_data, None)
    defcluster.to_csv('def_player_clusters_detailed.csv', index=False)
    print("✓ Saved defensive clusters to 'def_player_clusters_detailed.csv'")

    print("\n" + "=" * 60)
    print("CLUSTERING COMPLETE!")
    print("=" * 60)
    print(f"Total players clustered: {len(clustered_data)}")
    print(f"Offensive clusters created: {clustered_data['Cluster'].nunique()}")
    print(f"Defensive clusters created: {defcluster['Cluster'].nunique()}")

    return


st.set_page_config(page_title="WNBA Archetype Scout", layout="wide")


@st.cache_data
def load_data():
    # Populate after running WNBA clustering — add misclassified WNBA players here:
    DEF_CLUSTER_OVERRIDES = {
        # 'Player Name': cluster_id,
    }
    OFF_CLUSTER_NAMES = {
        0: 'Scoring Guard',
        2: 'Paint Big',
        3: 'Playmaking Guard',
        4: 'Star Forward',
        5: 'Fringe Player',
        6: 'Role Player',
    }
    DEF_CLUSTER_NAMES = {
        0: 'Perimeter Disruptor',
        1: 'Rebounding Big',
        2: 'Two-Way Rebounder',
        3: 'Versatile Defender',
        4: 'Active Guard Defender',
        5: 'Offensive-First Guard',
        6: 'Elite Shot Blocker',
        7: 'Role Defender',
    }
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

    offcluster = pd.read_csv('player_clusters_detailed.csv')[['PLAYER_ID', 'Cluster']].rename(
        columns={'Cluster': 'OffCluster'})
    defcluster = pd.read_csv('def_player_clusters_detailed.csv')[['PLAYER_ID', 'Cluster']].rename(
        columns={'Cluster': 'DefCluster'})

    merged = (
        playerbox
        .merge(offcluster, on='PLAYER_ID', how='left')
        .merge(defcluster, on='PLAYER_ID', how='left')
    )
    merged['GAME_DATE'] = pd.to_datetime(merged['GAME_DATE'])

    # Apply manual DefCluster overrides by player name
    override_mask = merged['PLAYER_NAME'].isin(DEF_CLUSTER_OVERRIDES)
    merged.loc[override_mask, 'DefCluster'] = merged.loc[override_mask, 'PLAYER_NAME'].map(DEF_CLUSTER_OVERRIDES)
    merged['OffClusterName'] = merged['OffCluster'].map(
        lambda x: OFF_CLUSTER_NAMES.get(int(x)) if pd.notna(x) else None
    )
    merged['DefClusterName'] = merged['DefCluster'].map(
        lambda x: DEF_CLUSTER_NAMES.get(int(x)) if pd.notna(x) else None
    )

    return merged


def main():
    merged = load_data()
    st.title("WNBA Archetype Scout")

    # Build cluster name maps from merged data
    off_cluster_name_map = (
        merged.dropna(subset=['OffCluster', 'OffClusterName'])
        .drop_duplicates('OffCluster')
        .set_index('OffCluster')['OffClusterName']
        .to_dict()
    )
    def_cluster_name_map = (
        merged.dropna(subset=['DefCluster', 'DefClusterName'])
        .drop_duplicates('DefCluster')
        .set_index('DefCluster')['DefClusterName']
        .to_dict()
    )

    def format_off_cluster(x):
        if x == "All":
            return "All"
        name = off_cluster_name_map.get(x, '')
        return f"{name} ({int(x)})" if name else f"Cluster {int(x)}"

    def format_def_cluster(x):
        if x == "All":
            return "All"
        name = def_cluster_name_map.get(x, '')
        return f"{name} ({int(x)})" if name else f"Cluster {int(x)}"

    st.sidebar.header("Filters")

    # Initialize session state for filters if not exists
    if 'team_filter' not in st.session_state:
        st.session_state.team_filter = "All"
    if 'opp_filter' not in st.session_state:
        st.session_state.opp_filter = "All"
    if 'player_filter' not in st.session_state:
        st.session_state.player_filter = "All"
    if 'off_cluster_filter' not in st.session_state:
        st.session_state.off_cluster_filter = "All"
    if 'def_cluster_filter' not in st.session_state:
        st.session_state.def_cluster_filter = "All"

    # Function to clear all filters (used as callback)
    def clear_all_filters():
        st.session_state.team_filter = "All"
        st.session_state.opp_filter = "All"
        st.session_state.player_filter = "All"
        st.session_state.off_cluster_filter = "All"
        st.session_state.def_cluster_filter = "All"

    # Handle individual clear button clicks BEFORE creating widgets
    if 'clear_team' in st.session_state and st.session_state.clear_team:
        st.session_state.team_filter = "All"
    if 'clear_opp' in st.session_state and st.session_state.clear_opp:
        st.session_state.opp_filter = "All"
    if 'clear_player' in st.session_state and st.session_state.clear_player:
        st.session_state.player_filter = "All"
    if 'clear_off' in st.session_state and st.session_state.clear_off:
        st.session_state.off_cluster_filter = "All"
    if 'clear_def' in st.session_state and st.session_state.clear_def:
        st.session_state.def_cluster_filter = "All"

    # Clear all filters button (BEFORE widgets, using on_click callback)
    st.sidebar.button("🔄 Clear All Filters", on_click=clear_all_filters)

    # Team filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        teams = sorted(merged['TEAM_ABBREVIATION'].unique())
        selected_team = st.selectbox("Team", ["All"] + teams, key='team_filter')
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_team")

    # Opponent filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        opponents = sorted(merged['TEAM_ABBREVIATION'].unique())
        selected_opp = st.selectbox("Opponent", ["All"] + opponents, key='opp_filter')
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_opp")

    # Player filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        players = sorted(merged['PLAYER_NAME'].unique())
        selected_player = st.selectbox("Player", ["All"] + players, key='player_filter')
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_player")

    # Offensive Cluster filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        off_clusters = sorted(merged['OffCluster'].dropna().unique())
        selected_off_cluster = st.selectbox(
            "Offensive Cluster", ["All"] + off_clusters,
            key='off_cluster_filter', format_func=format_off_cluster
        )
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_off")

    # Defensive Cluster filter with clear button
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        def_clusters = sorted(merged['DefCluster'].dropna().unique())
        selected_def_cluster = st.selectbox(
            "Defensive Cluster", ["All"] + def_clusters,
            key='def_cluster_filter', format_func=format_def_cluster
        )
    with col2:
        st.write("")  # Spacing
        st.button("✕", key="clear_def")

    min_date, max_date = merged['GAME_DATE'].min(), merged['GAME_DATE'].max()
    selected_date = st.sidebar.date_input("Game Date Range", [min_date, max_date])

    # --- Apply Filters once ---
    df_filtered = merged.copy()

    if selected_team != "All":
        df_filtered = df_filtered[df_filtered['TEAM_ABBREVIATION'] == selected_team]
    if selected_opp != "All":
        df_filtered = df_filtered[df_filtered['MATCHUP'].apply(lambda x: x.split()[-1]) == selected_opp]
    if selected_player != "All":
        df_filtered = df_filtered[df_filtered['PLAYER_NAME'] == selected_player]
    if selected_off_cluster != "All":
        df_filtered = df_filtered[df_filtered['OffCluster'] == selected_off_cluster]
    if selected_def_cluster != "All":
        df_filtered = df_filtered[df_filtered['DefCluster'] == selected_def_cluster]
    if len(selected_date) == 2:
        df_filtered = df_filtered[
            (df_filtered['GAME_DATE'] >= pd.to_datetime(selected_date[0])) &
            (df_filtered['GAME_DATE'] <= pd.to_datetime(selected_date[1]))
            ]

    # =====================================================
    # 🔹 Tabs that share this filtered dataset
    # =====================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Player Stats", "🧠 Clustering Overview", "Team Best/Worst Cluster Performance", "🔥 Today's Best Matchup Opportunities", "💰 FanDuel EV"])

    # =====================================================
    # 🟦 TAB 1 — Player Stats
    # =====================================================
    with tab1:
        st.markdown("### Player Stats + Archetypes")
        if df_filtered.empty:
            st.info("No games match your filters.")
        else:
            st.dataframe(
                df_filtered[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_DATE',
                             'MATCHUP', 'MIN','PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'FTA', 'OffClusterName', 'DefClusterName']]
                .sort_values('GAME_DATE', ascending=False)
                .reset_index(drop=True)
            )

            # --- Summary metrics ---
            st.markdown("### Summary Stats")
            avg_pts = df_filtered['PTS'].mean()
            avg_reb = df_filtered['REB'].mean()
            avg_ast = df_filtered['AST'].mean()
            avg_3pa = df_filtered['FG3A'].mean()
            avg_FTA = df_filtered['FTA'].mean()

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Average Points", f"{avg_pts:.1f}")
            col2.metric("Average Rebounds", f"{avg_reb:.1f}")
            col3.metric("Average Assists", f"{avg_ast:.1f}")
            col4.metric("Average 3PA", f"{avg_3pa:.1f}")
            col5.metric("Average FTA", f"{avg_FTA:.1f}")

    # =====================================================
    # 🟩 TAB 2 — Cluster Overview (with % Diff vs Team)
    # =====================================================
    with tab2:
        st.subheader("Cluster Performance Overview")
        default_off_cluster = None
        default_def_cluster = None

        # Auto-select clusters based on selected player
        if selected_player != "All":
            player_data = merged[merged['PLAYER_NAME'] == selected_player]
            if not player_data.empty:
                player_data = player_data.iloc[0]
                default_off_cluster = player_data['OffCluster'] if pd.notna(player_data['OffCluster']) else None
                default_def_cluster = player_data['DefCluster'] if pd.notna(player_data['DefCluster']) else None
            else:
                default_off_cluster = None
                default_def_cluster = None
        if len(selected_date) == 2:
            merged = merged[
                (merged['GAME_DATE'] >= pd.to_datetime(selected_date[0])) &
                (merged['GAME_DATE'] <= pd.to_datetime(selected_date[1]))
                ]

        # Allow user to select both offensive and defensive clusters
        col1, col2 = st.columns(2)

        with col1:
            off_cluster_options = sorted(merged['OffCluster'].dropna().unique())
            if default_off_cluster is not None and default_off_cluster in off_cluster_options:
                default_off_index = off_cluster_options.index(default_off_cluster)
            else:
                default_off_index = 0
            selected_off_analysis = st.selectbox(
                "Offensive Cluster for Analysis",
                off_cluster_options,
                index=default_off_index,
                key=f"off_analysis_{selected_player}",
                format_func=format_off_cluster
            )

        with col2:
            def_cluster_options = sorted(merged['DefCluster'].dropna().unique())
            if default_def_cluster is not None and default_def_cluster in def_cluster_options:
                default_def_index = def_cluster_options.index(default_def_cluster)
            else:
                default_def_index = 0
            selected_def_analysis = st.selectbox(
                "Defensive Cluster for Analysis",
                def_cluster_options,
                index=default_def_index,
                key=f"def_analysis_{selected_player}",
                format_func=format_def_cluster
            )

        # Get data for both clusters
        df_off_cluster = merged[(merged['OffCluster'] == selected_off_analysis) &
                                (merged['GAME_DATE'] >= min_date) &
                                (merged['GAME_DATE'] <= max_date)]

        df_def_cluster = merged[(merged['DefCluster'] == selected_def_analysis) &
                                (merged['GAME_DATE'] >= min_date) &
                                (merged['GAME_DATE'] <= max_date)]

        if df_off_cluster.empty or df_def_cluster.empty:
            st.warning("No games match the selected clusters within your filters.")
        else:
            st.markdown(f"### 📈 Cluster Analysis")
            off_analysis_name = off_cluster_name_map.get(selected_off_analysis, '')
            off_analysis_label = f"{off_analysis_name} ({int(selected_off_analysis)})" if off_analysis_name else f"Cluster {int(selected_off_analysis)}"
            def_analysis_name = def_cluster_name_map.get(selected_def_analysis, '')
            def_analysis_label = f"{def_analysis_name} ({int(selected_def_analysis)})" if def_analysis_name else f"Cluster {int(selected_def_analysis)}"
            st.markdown(
                f"**Offensive Cluster {off_analysis_label}** | **Defensive Cluster {def_analysis_label}**")

            # Show example players from both clusters
            off_players = df_off_cluster['PLAYER_NAME'].unique()[:5]
            def_players = df_def_cluster['PLAYER_NAME'].unique()[:5]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Example Players (Offensive):**")
                st.markdown("- " + "\n- ".join(off_players))
            with col2:
                st.markdown("**Example Players (Defensive):**")
                st.markdown("- " + "\n- ".join(def_players))

            # Define stat categories
            offensive_stats = ['PTS', 'AST','OREB',  'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA','TOV']
            defensive_stats = ['DREB', 'REB', 'STL', 'BLK',  'PF']
            all_counting_stats = offensive_stats + defensive_stats

            MIN_THRESHOLD = 5

            # Calculate per-36 averages for OFFENSIVE cluster (all players in this cluster)
            df_off_valid = df_off_cluster[df_off_cluster['MIN'] >= MIN_THRESHOLD]
            df_off_per36 = df_off_valid[offensive_stats].div(df_off_valid['MIN'], axis=0) * 36
            season_avg_per36_off = df_off_per36.mean()

            # Calculate per-36 averages for DEFENSIVE cluster (all players in this cluster)
            df_def_valid = df_def_cluster[df_def_cluster['MIN'] >= MIN_THRESHOLD]
            df_def_per36 = df_def_valid[defensive_stats].div(df_def_valid['MIN'], axis=0) * 36
            season_avg_per36_def = df_def_per36.mean()

            # If opponent is selected, calculate % differences
            if selected_opp != "All":
                # Add opponent team column
                df_off_cluster.loc[:, 'OPP_TEAM'] = df_off_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                df_def_cluster.loc[:, 'OPP_TEAM'] = df_def_cluster['MATCHUP'].apply(lambda x: x.split()[-1])

                # Filter by opponent
                df_off_vs_team = df_off_cluster[df_off_cluster['OPP_TEAM'] == selected_opp]
                df_def_vs_team = df_def_cluster[df_def_cluster['OPP_TEAM'] == selected_opp]

                # Filter by minutes
                df_off_vs_team_valid = df_off_vs_team[df_off_vs_team['MIN'] >= MIN_THRESHOLD]
                df_def_vs_team_valid = df_def_vs_team[df_def_vs_team['MIN'] >= MIN_THRESHOLD]

                # Helper: compute Bayesian % diffs for a given vs-opponent time slice
                def _compute_pct_diffs(df_off_vs, df_def_vs):
                    p_off = pd.Series(0.0, index=offensive_stats)
                    p_def = pd.Series(0.0, index=defensive_stats)
                    h_off = not df_off_vs.empty
                    h_def = not df_def_vs.empty
                    if h_off:
                        sorted_off = df_off_vs.sort_values('GAME_DATE')
                        per36_off = sorted_off[offensive_stats].div(sorted_off['MIN'], axis=0) * 36
                        diffs_off = ((per36_off - season_avg_per36_off[offensive_stats])
                                     / season_avg_per36_off[offensive_stats] * 100)
                        for s in offensive_stats:
                            p_off[s] = bayesian_pct_diff(diffs_off[s].values, s)
                        p_off['FG3M'] = p_off['FG3A']
                    if h_def:
                        sorted_def = df_def_vs.sort_values('GAME_DATE')
                        per36_def = sorted_def[defensive_stats].div(sorted_def['MIN'], axis=0) * 36
                        diffs_def = ((per36_def - season_avg_per36_def[defensive_stats])
                                     / season_avg_per36_def[defensive_stats] * 100)
                        for s in defensive_stats:
                            p_def[s] = bayesian_pct_diff(diffs_def[s].values, s)
                    combined = pd.Series(index=all_counting_stats, dtype=float)
                    for s in offensive_stats:
                        combined[s] = p_off[s]
                    for s in defensive_stats:
                        combined[s] = p_def[s]
                    # REB = adjusted OREB (off cluster) + adjusted DREB (def cluster)
                    reb_season = season_avg_per36_off['OREB'] + season_avg_per36_def['DREB']
                    if reb_season > 0:
                        reb_proj = (season_avg_per36_def['DREB'] * (1 + p_def['DREB'] / 100) +
                                    season_avg_per36_off['OREB'] * (1 + p_off['OREB'] / 100))
                        combined['REB'] = ((reb_proj / reb_season) - 1) * 100
                    combined['FG3M'] = combined['FG3A']
                    return p_off, p_def, h_off, h_def, combined

                # Full season
                pct_diff_off, pct_diff_def, has_off_history, has_def_history, avg_pct_diff_combined = \
                    _compute_pct_diffs(df_off_vs_team_valid, df_def_vs_team_valid)

                # Last 3 months
                cutoff_3m = max_date - pd.DateOffset(months=3)
                _, _, _, _, avg_pct_diff_3m = _compute_pct_diffs(
                    df_off_vs_team_valid[df_off_vs_team_valid['GAME_DATE'] >= cutoff_3m],
                    df_def_vs_team_valid[df_def_vs_team_valid['GAME_DATE'] >= cutoff_3m]
                )

                # Last month
                cutoff_1m = max_date - pd.DateOffset(months=1)
                _, _, _, _, avg_pct_diff_1m = _compute_pct_diffs(
                    df_off_vs_team_valid[df_off_vs_team_valid['GAME_DATE'] >= cutoff_1m],
                    df_def_vs_team_valid[df_def_vs_team_valid['GAME_DATE'] >= cutoff_1m]
                )

                # Show comparison table if we have any historical data
                if has_off_history or has_def_history:
                    st.markdown(f"### 📊 Stats Comparison vs {selected_opp}")

                    if not has_off_history:
                        st.info("⚠️ No offensive cluster history vs this opponent. Offensive stats reflect prior only.")
                    if not has_def_history:
                        st.info("⚠️ No defensive cluster history vs this opponent. Defensive stats reflect prior only.")

                    season_avg_row = pd.concat([
                        season_avg_per36_off[offensive_stats],
                        season_avg_per36_def[defensive_stats]
                    ])

                    comparison_data = {
                        "Season Avg (per 36)": season_avg_row,
                        "% Diff - Full Season": avg_pct_diff_combined,
                        "% Diff - Last 3 Mo.": avg_pct_diff_3m,
                        "% Diff - Last Month": avg_pct_diff_1m,
                    }

                    comparison_df = pd.DataFrame(comparison_data).T

                    def color_cells(val, row_name):
                        if '% Diff' in row_name:
                            try:
                                if val > 0:
                                    return 'background-color: #28A745; color: white'
                                elif val < 0:
                                    return 'background-color: #DC3545; color: white'
                            except:
                                pass
                        return ''

                    styled_df = comparison_df.style.format("{:.2f}").apply(
                        lambda row: [color_cells(val, row.name) for val in row], axis=1
                    )

                    st.dataframe(styled_df)
                else:
                    st.info(
                        f"⚠️ No historical data for either cluster vs {selected_opp} with {MIN_THRESHOLD}+ minutes. Showing season averages without adjustments.")

                # =====================================================
                # 🟨 Projected Stats Section - ALWAYS SHOW
                # =====================================================
                st.markdown("---")
                st.markdown(f"### 🎯 Projected Stats vs {selected_opp}")

                if has_off_history and has_def_history:
                    st.markdown("*Season averages adjusted by cluster's historical % difference vs this opponent*")
                elif has_off_history:
                    st.markdown(
                        "*Offensive stats adjusted by cluster history. Defensive stats show season averages (no cluster history).*")
                elif has_def_history:
                    st.markdown(
                        "*Defensive stats adjusted by cluster history. Offensive stats show season averages (no cluster history).*")
                else:
                    st.markdown("*Showing season averages (no historical cluster data vs this opponent available)*")

                st.caption(
                    f"**Note:** Offensive stats use Offensive Cluster {off_analysis_label}, Defensive stats use Defensive Cluster {def_analysis_label}")

                # Get ALL unique players from the season who have the required clusters
                # We need to aggregate from merged dataset to get season averages per player

                # Group by player to get season averages for offensive stats
                # Group by player to get season averages for offensive stats
                off_players_season = merged[
                    (merged['OffCluster'] == selected_off_analysis) &
                    (merged['GAME_DATE'] >= min_date) &
                    (merged['GAME_DATE'] <= max_date) &
                    (merged['MIN'] >= MIN_THRESHOLD)
                    ].groupby('PLAYER_NAME').agg({
                    'MIN': 'mean',
                    **{stat: 'mean' for stat in offensive_stats}
                }).reset_index()

                # Group by player to get season averages for defensive stats
                def_players_season = merged[
                    (merged['DefCluster'] == selected_def_analysis) &
                    (merged['GAME_DATE'] >= min_date) &
                    (merged['GAME_DATE'] <= max_date) &
                    (merged['MIN'] >= MIN_THRESHOLD)
                    ].groupby('PLAYER_NAME').agg({
                    'MIN': 'mean',
                    **{stat: 'mean' for stat in defensive_stats}
                }).reset_index()

                # NEW: Get player season averages WITHOUT minutes filter for projections
                off_players_no_min_filter = merged[
                    (merged['OffCluster'] == selected_off_analysis) &
                    (merged['GAME_DATE'] >= min_date) &
                    (merged['GAME_DATE'] <= max_date)
                    # NO MIN_THRESHOLD here
                    ].groupby('PLAYER_NAME').agg({
                    'MIN': 'mean',
                    **{stat: 'mean' for stat in offensive_stats}
                }).reset_index()

                def_players_no_min_filter = merged[
                    (merged['DefCluster'] == selected_def_analysis) &
                    (merged['GAME_DATE'] >= min_date) &
                    (merged['GAME_DATE'] <= max_date)
                    # NO MIN_THRESHOLD here
                    ].groupby('PLAYER_NAME').agg({
                    'MIN': 'mean',
                    **{stat: 'mean' for stat in defensive_stats}
                }).reset_index()

                # Merge to get players who have EITHER cluster (outer join)
                # This way each player brings their own cluster's stats
                all_players = off_players_season.merge(
                    def_players_season,
                    on='PLAYER_NAME',
                    how='outer',
                    suffixes=('_off', '_def')
                )

                # NEW: Merge the no-filter versions for projections
                all_players_for_projection = off_players_no_min_filter.merge(
                    def_players_no_min_filter,
                    on='PLAYER_NAME',
                    how='outer',
                    suffixes=('_off', '_def')
                )

                # Time-windowed player averages (for Season column in single-player view)
                def _build_player_avgs(date_cutoff):
                    off = merged[
                        (merged['OffCluster'] == selected_off_analysis) &
                        (merged['GAME_DATE'] >= date_cutoff) &
                        (merged['GAME_DATE'] <= max_date)
                    ].groupby('PLAYER_NAME').agg(
                        {'MIN': 'mean', **{s: 'mean' for s in offensive_stats}}
                    ).reset_index()
                    def_ = merged[
                        (merged['DefCluster'] == selected_def_analysis) &
                        (merged['GAME_DATE'] >= date_cutoff) &
                        (merged['GAME_DATE'] <= max_date)
                    ].groupby('PLAYER_NAME').agg(
                        {'MIN': 'mean', **{s: 'mean' for s in defensive_stats}}
                    ).reset_index()
                    return off.merge(def_, on='PLAYER_NAME', how='outer', suffixes=('_off', '_def'))

                all_players_3m = _build_player_avgs(cutoff_3m)
                all_players_1m = _build_player_avgs(cutoff_1m)

                # Index by player name for fast lookup
                proj_lookup = {
                    "Full Season": all_players_for_projection.set_index('PLAYER_NAME'),
                    "Last 3 Mo.":  all_players_3m.set_index('PLAYER_NAME'),
                    "Last Month":  all_players_1m.set_index('PLAYER_NAME'),
                }

                # Calculate median minutes from last 10 games for the slider
                if selected_player != "All":
                    # Get the specific player's last 10 games
                    player_games = merged[merged['PLAYER_NAME'] == selected_player].copy()
                    player_games = player_games.sort_values('GAME_DATE', ascending=False).head(10)

                    if not player_games.empty:
                        median_min = player_games['MIN'].median()
                        default_min = round(median_min, 1)
                    else:
                        default_min = 30.0  # fallback default

                    # Add minutes slider
                    st.subheader("Adjust Minutes")
                    selected_minutes = st.slider(
                        "Minutes per game:",
                        min_value=0.0,
                        max_value=40.0,
                        value=default_min,
                        step=0.5,
                        help=f"Default is median of last 10 games ({default_min:.1f} min)"
                    )

                    # Calculate the minutes multiplier
                    minutes_multiplier = selected_minutes / default_min if default_min > 0 else 1.0
                else:
                    # For "All" players, use their season average (no adjustment)
                    minutes_multiplier = 1.0
                    selected_minutes = None

                if all_players_for_projection.empty:
                    st.warning("No players found in the selected clusters.")
                else:
                    # Single-player view: 3 rows (one per time window)
                    # All-players view: one row per player using full-season data
                    window_configs = [
                        ("Full Season", avg_pct_diff_combined),
                        ("Last 3 Mo.", avg_pct_diff_3m),
                        ("Last Month", avg_pct_diff_1m),
                    ] if selected_player != "All" else [
                        ("Full Season", avg_pct_diff_combined),
                    ]

                    players_to_project = []

                    for idx, player_row in all_players_for_projection.iterrows():
                        player_name = player_row['PLAYER_NAME']

                        # For single-player view, skip other players early
                        if selected_player != "All" and player_name != selected_player:
                            continue

                        # Verify player exists in at least the full-season data
                        if pd.isna(player_row.get('MIN_off')) and pd.isna(player_row.get('MIN_def')):
                            continue

                        # Whole-season 3P% for FG3M derivation (stable across time windows)
                        full_fg3a = player_row.get('FG3A', 0)
                        full_fg3m = player_row.get('FG3M', 0)
                        season_fg3_pct = (full_fg3m / full_fg3a
                                          if (pd.notna(full_fg3a) and pd.notna(full_fg3m) and full_fg3a > 0)
                                          else 0)

                        for window_label, pct_diff in window_configs:
                            # Use time-windowed player averages for Season column and MPG
                            lookup = proj_lookup[window_label]
                            w_row = lookup.loc[player_name] if player_name in lookup.index else player_row

                            has_off = pd.notna(w_row.get('MIN_off'))
                            has_def = pd.notna(w_row.get('MIN_def'))

                            # MPG shows the window's actual average minutes
                            w_base_min = w_row['MIN_off'] if has_off else w_row['MIN_def']
                            display_min = w_base_min

                            if selected_player != "All" and player_name == selected_player:
                                actual_multiplier = selected_minutes / w_base_min if w_base_min > 0 else 1.0
                            else:
                                actual_multiplier = 1.0

                            projected_row = {
                                'PLAYER_NAME': player_name,
                                'Window': window_label,
                                'OffClusterName': off_analysis_label if has_off else None,
                                'DefClusterName': def_analysis_label if has_def else None,
                                'AVG_MIN': display_min,
                            }

                            if has_off:
                                for stat in offensive_stats:
                                    if stat == 'FG3M':
                                        # FG3M = FG3A (window) × whole-season 3P%
                                        fg3a_w = w_row.get('FG3A', 0) if pd.notna(w_row.get('FG3A')) else 0
                                        fg3a_pct = pct_diff['FG3A'] if 'FG3A' in pct_diff.index else 0
                                        fg3a_proj = fg3a_w * (1 + fg3a_pct / 100) * actual_multiplier
                                        season_avg = fg3a_w * season_fg3_pct
                                        projected_value = fg3a_proj * season_fg3_pct
                                    else:
                                        season_avg = w_row[stat]
                                        pct_change = pct_diff[stat] if stat in pct_diff.index else 0
                                        projected_value = season_avg * (1 + pct_change / 100) * actual_multiplier
                                    projected_row[f'{stat}_Season'] = season_avg
                                    projected_row[f'{stat}_Projected'] = projected_value
                                    projected_row[f'{stat}_Diff'] = projected_value - season_avg
                            else:
                                for stat in offensive_stats:
                                    projected_row[f'{stat}_Season'] = 0
                                    projected_row[f'{stat}_Projected'] = 0
                                    projected_row[f'{stat}_Diff'] = 0

                            if has_def:
                                for stat in defensive_stats:
                                    if stat == 'REB':
                                        season_avg = w_row['DREB'] + w_row['OREB']
                                        dreb_pct = pct_diff['DREB'] if 'DREB' in pct_diff.index else 0
                                        oreb_pct = pct_diff['OREB'] if 'OREB' in pct_diff.index else 0
                                        projected_value = (
                                            w_row['DREB'] * (1 + dreb_pct / 100) +
                                            w_row['OREB'] * (1 + oreb_pct / 100)
                                        ) * actual_multiplier
                                    else:
                                        season_avg = w_row[stat]
                                        pct_change = pct_diff[stat] if stat in pct_diff.index else 0
                                        projected_value = season_avg * (1 + pct_change / 100) * actual_multiplier
                                    projected_row[f'{stat}_Season'] = season_avg
                                    projected_row[f'{stat}_Projected'] = projected_value
                                    projected_row[f'{stat}_Diff'] = projected_value - season_avg
                            else:
                                for stat in defensive_stats:
                                    projected_row[f'{stat}_Season'] = 0
                                    projected_row[f'{stat}_Projected'] = 0
                                    projected_row[f'{stat}_Diff'] = 0

                            players_to_project.append(projected_row)

                    if not players_to_project:
                        st.warning("No valid players found for projections.")
                    else:
                        projected_df = pd.DataFrame(players_to_project)

                        display_stats = st.multiselect(
                            "Select stats to display:",
                            all_counting_stats,
                            default=['PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'FTA', 'STL', 'BLK']
                        )

                        if display_stats:
                            # Single-player view: Window replaces PLAYER_NAME as the row label
                            if selected_player != "All":
                                display_columns = ['Window', 'AVG_MIN']
                            else:
                                display_columns = ['PLAYER_NAME', 'AVG_MIN', 'OffClusterName', 'DefClusterName']
                            for stat in display_stats:
                                display_columns.extend([f'{stat}_Season', f'{stat}_Projected', f'{stat}_Diff'])

                            display_df = projected_df[display_columns].copy()

                            if display_df.empty:
                                st.warning(f"Player {selected_player} not found in the selected clusters.")
                            else:
                                rename_dict = {
                                    'PLAYER_NAME': 'Player',
                                    'Window': 'Window',
                                    'AVG_MIN': 'MPG',
                                    'OffClusterName': 'Off',
                                    'DefClusterName': 'Def',
                                }
                                for stat in display_stats:
                                    rename_dict[f'{stat}_Season'] = f'{stat} (Season)'
                                    rename_dict[f'{stat}_Projected'] = f'{stat} (Proj.)'
                                    rename_dict[f'{stat}_Diff'] = f'{stat} (±)'

                                display_df = display_df.rename(columns=rename_dict)

                                # Sort by projected points for all-players view
                                if selected_player == "All" and 'PTS (Proj.)' in display_df.columns:
                                    display_df = display_df.sort_values('PTS (Proj.)', ascending=False)

                                def highlight_diff(val, col_name):
                                    if '(±)' in col_name:
                                        try:
                                            if val > 0:
                                                return 'background-color: #D4EDDA; color: #155724'
                                            elif val < 0:
                                                return 'background-color: #F8D7DA; color: #721C24'
                                        except:
                                            pass
                                    return ''

                                styled_proj = display_df.style.format({
                                    col: "{:.1f}" for col in display_df.columns
                                    if col not in ['Player', 'Off', 'Def', 'Window']
                                }).apply(
                                    lambda row: [highlight_diff(val, col) for col, val in zip(display_df.columns, row)],
                                    axis=1
                                )

                                st.dataframe(styled_proj, use_container_width=True)

                                st.caption(
                                    f"💡 Offensive stats ({', '.join(offensive_stats)}) use ALL players in Off Cluster {off_analysis_label}")
                                st.caption(
                                    f"💡 Defensive stats ({', '.join(defensive_stats)}) use ALL players in Def Cluster {def_analysis_label}")
                        else:
                            st.info("Select at least one stat to display projections.")

            else:
                st.info("Select an opponent to see cluster performance analysis and projections.")
            games_off = len(df_off_cluster) if selected_opp == "All" else len(
                df_off_cluster[df_off_cluster['OPP_TEAM'] == selected_opp])
            st.caption(f"Games in offensive cluster sample: {games_off:,}")

            # For defensive cluster
            games_def = len(df_def_cluster) if selected_opp == "All" else len(
                df_def_cluster[df_def_cluster['OPP_TEAM'] == selected_opp])
            st.caption(f"Games in defensive cluster sample: {games_def:,}")


# =====================================================
# 🟦 TAB 3 — Team Cluster Matchup Analysis
# =====================================================
    with tab3:
        st.subheader("Team Cluster Matchup Analysis")

        if selected_opp == "All":
            st.warning("⚠️ Please select a specific opponent team to view cluster matchup analysis.")
        else:
            st.markdown(f"### 📊 Best & Worst Clusters vs {selected_opp}")
            st.markdown(f"*Showing which clusters perform best/worst against {selected_opp}*")

            # Define stat categories (same as Tab 2)
            offensive_stats = ['PTS', 'AST', 'OREB', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV']
            defensive_stats = ['DREB', 'REB', 'STL', 'BLK', 'PF']

            MIN_THRESHOLD = 5
            prior_mean = 0.0
            prior_weight = 3.0

            # Get all unique clusters
            all_off_clusters = sorted(merged['OffCluster'].dropna().unique())
            all_def_clusters = sorted(merged['DefCluster'].dropna().unique())

            # =====================================================
            # Calculate % diff for each offensive cluster vs selected opponent
            # =====================================================
            off_cluster_results = {}

            for cluster in all_off_clusters:
                df_cluster = merged[(merged['OffCluster'] == cluster) &
                                    (merged['GAME_DATE'] >= min_date) &
                                    (merged['GAME_DATE'] <= max_date)]

                if df_cluster.empty:
                    continue

                # Calculate season averages
                df_valid = df_cluster[df_cluster['MIN'] >= MIN_THRESHOLD]
                df_per36 = df_valid[offensive_stats].div(df_valid['MIN'], axis=0) * 36
                season_avg_per36 = df_per36.mean()

                # Filter vs opponent
                df_cluster.loc[:, 'OPP_TEAM'] = df_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                df_vs_team = df_cluster[df_cluster['OPP_TEAM'] == selected_opp]
                df_vs_team_valid = df_vs_team[df_vs_team['MIN'] >= MIN_THRESHOLD]

                if not df_vs_team_valid.empty:
                    # Sort by game date
                    df_vs_team_valid_sorted = df_vs_team_valid.sort_values('GAME_DATE')

                    # Calculate per-36 for each game
                    df_vs_team_per36 = df_vs_team_valid_sorted[offensive_stats].div(
                        df_vs_team_valid_sorted['MIN'], axis=0) * 36

                    # Calculate percentage difference for each game
                    game_pct_diffs = (
                            (df_vs_team_per36 - season_avg_per36[offensive_stats]) /
                            season_avg_per36[offensive_stats] * 100
                    )

                    # Bayesian sequential update for each stat
                    pct_diffs = {}
                    for stat in offensive_stats:
                        pct_diffs[stat] = bayesian_pct_diff(game_pct_diffs[stat].values, stat)

                    # Get player examples
                    cluster_players = df_cluster['PLAYER_NAME'].unique()
                    player_examples = list(np.random.choice(cluster_players, min(3, len(cluster_players)), replace=False))

                    off_cluster_results[cluster] = {
                        'pct_diffs': pct_diffs,
                        'players': player_examples
                    }

            # =====================================================
            # Calculate % diff for each defensive cluster vs selected opponent
            # =====================================================
            def_cluster_results = {}

            for cluster in all_def_clusters:
                df_cluster = merged[(merged['DefCluster'] == cluster) &
                                    (merged['GAME_DATE'] >= min_date) &
                                    (merged['GAME_DATE'] <= max_date)]

                if df_cluster.empty:
                    continue

                # Calculate season averages
                df_valid = df_cluster[df_cluster['MIN'] >= MIN_THRESHOLD]
                df_per36 = df_valid[defensive_stats].div(df_valid['MIN'], axis=0) * 36
                season_avg_per36 = df_per36.mean()

                # Filter vs opponent
                df_cluster.loc[:, 'OPP_TEAM'] = df_cluster['MATCHUP'].apply(lambda x: x.split()[-1])
                df_vs_team = df_cluster[df_cluster['OPP_TEAM'] == selected_opp]
                df_vs_team_valid = df_vs_team[df_vs_team['MIN'] >= MIN_THRESHOLD]

                if not df_vs_team_valid.empty:
                    # Sort by game date
                    df_vs_team_valid_sorted = df_vs_team_valid.sort_values('GAME_DATE')

                    # Calculate per-36 for each game
                    df_vs_team_per36 = df_vs_team_valid_sorted[defensive_stats].div(
                        df_vs_team_valid_sorted['MIN'], axis=0) * 36

                    # Calculate percentage difference for each game
                    game_pct_diffs = (
                            (df_vs_team_per36 - season_avg_per36[defensive_stats]) /
                            season_avg_per36[defensive_stats] * 100
                    )

                    # Bayesian sequential update for each stat
                    pct_diffs = {}
                    for stat in defensive_stats:
                        pct_diffs[stat] = bayesian_pct_diff(game_pct_diffs[stat].values, stat)

                    # Get player examples
                    cluster_players = df_cluster['PLAYER_NAME'].unique()
                    player_examples = list(np.random.choice(cluster_players, min(3, len(cluster_players)), replace=False))

                    def_cluster_results[cluster] = {
                        'pct_diffs': pct_diffs,
                        'players': player_examples
                    }


            # =====================================================
            # Helper function for heatmap coloring
            # =====================================================
            def get_heatmap_color(pct_value):
                """
                Returns RGB color string based on percentage value
                Scale from -25% (red) to +25% (green)
                """
                # Clamp value between -25 and 25
                clamped = max(-25, min(25, pct_value))

                # Normalize to 0-1 scale
                normalized = (clamped + 25) / 50

                # Red to Green gradient
                if normalized < 0.5:
                    # Red to Yellow (increase green)
                    r = 255
                    g = int(255 * (normalized * 2))
                    b = 0
                else:
                    # Yellow to Green (decrease red)
                    r = int(255 * (2 - normalized * 2))
                    g = 255
                    b = 0

                return f'background-color: rgb({r}, {g}, {b}); color: black; font-weight: bold;'


            # =====================================================
            # Build the results table
            # =====================================================
            if not off_cluster_results and not def_cluster_results:
                st.warning(f"No cluster data available vs {selected_opp} with {MIN_THRESHOLD}+ minutes.")
            else:
                # Store cluster-to-players mapping for display below table
                cluster_players_map = {}

                # Create table data
                table_rows = []
                pct_columns = []  # Track which columns contain percentages for styling

                # Process offensive stats
                for stat in offensive_stats:
                    if not off_cluster_results:
                        continue

                    # Get all clusters with this stat
                    cluster_pcts = [(cluster, data['pct_diffs'][stat], data['players'])
                                    for cluster, data in off_cluster_results.items()
                                    if stat in data['pct_diffs']]

                    if not cluster_pcts:
                        continue

                    # Sort by % diff
                    cluster_pcts_sorted = sorted(cluster_pcts, key=lambda x: x[1], reverse=True)

                    # Get top 3 best and top 3 worst
                    best_3 = cluster_pcts_sorted[:3]
                    worst_3 = cluster_pcts_sorted[-3:]
                    worst_3.reverse()  # Show worst first in order

                    # Build row
                    row = {'Stat': stat}

                    # Add best clusters
                    for i, (cluster, pct, players) in enumerate(best_3, 1):
                        cluster = int(cluster)
                        row[f'Best #{i} Cluster'] = cluster
                        row[f'Best #{i} %'] = pct
                        cluster_players_map[f"Off-{cluster}"] = players
                        if f'Best #{i} %' not in pct_columns:
                            pct_columns.append(f'Best #{i} %')

                    # Add worst clusters
                    for i, (cluster, pct, players) in enumerate(worst_3, 1):
                        cluster = int(cluster)
                        row[f'Worst #{i} Cluster'] = cluster
                        row[f'Worst #{i} %'] = pct
                        cluster_players_map[f"Off-{cluster}"] = players
                        if f'Worst #{i} %' not in pct_columns:
                            pct_columns.append(f'Worst #{i} %')

                    table_rows.append(row)

                # Display offensive stats table
                if table_rows:
                    st.markdown("### Offensive Stats")
                    off_stats_df = pd.DataFrame(table_rows)

                    # Apply styling to percentage columns
                    styled_off = off_stats_df.style.format({col: '{:+.1f}%' for col in pct_columns})
                    cluster_cols = [col for col in off_stats_df.columns if 'Cluster' in col]
                    off_stats_df[cluster_cols] = off_stats_df[cluster_cols].astype(int)

                    # Apply heatmap to percentage columns
                    for col in pct_columns:
                        styled_off = styled_off.applymap(get_heatmap_color, subset=[col])

                    st.dataframe(styled_off, use_container_width=True, height=600)

                    # Display cluster examples in expandable sections
                    st.markdown("#### 📋 Offensive Cluster Examples")
                    off_clusters_used = set()
                    for i in range(1, 4):
                        off_clusters_used.update(off_stats_df[f'Best #{i} Cluster'].values)
                        off_clusters_used.update(off_stats_df[f'Worst #{i} Cluster'].values)
                    off_clusters_used = sorted(off_clusters_used)

                    cols = st.columns(3)
                    for idx, cluster in enumerate(off_clusters_used):
                        with cols[idx % 3]:
                            off_name = off_cluster_name_map.get(cluster, '')
                            off_expander_label = f"Off Cluster {cluster}: {off_name}" if off_name else f"Offensive Cluster {cluster}"
                            with st.expander(off_expander_label):
                                players = cluster_players_map.get(f"Off-{cluster}", [])
                                st.write(", ".join(players))

                # Process defensive stats
                table_rows = []
                pct_columns = []

                for stat in defensive_stats:
                    if not def_cluster_results:
                        continue

                    # Get all clusters with this stat
                    cluster_pcts = [(cluster, data['pct_diffs'][stat], data['players'])
                                    for cluster, data in def_cluster_results.items()
                                    if stat in data['pct_diffs']]

                    if not cluster_pcts:
                        continue

                    # Sort by % diff
                    cluster_pcts_sorted = sorted(cluster_pcts, key=lambda x: x[1], reverse=True)

                    # Get top 3 best and top 3 worst
                    best_3 = cluster_pcts_sorted[:3]
                    worst_3 = cluster_pcts_sorted[-3:]
                    worst_3.reverse()  # Show worst first in order

                    # Build row
                    row = {'Stat': stat}

                    # Add best clusters
                    for i, (cluster, pct, players) in enumerate(best_3, 1):
                        cluster = int(cluster)
                        row[f'Best #{i} Cluster'] = cluster
                        row[f'Best #{i} %'] = pct
                        cluster_players_map[f"Def-{cluster}"] = players
                        if f'Best #{i} %' not in pct_columns:
                            pct_columns.append(f'Best #{i} %')

                    # Add worst clusters
                    for i, (cluster, pct, players) in enumerate(worst_3, 1):
                        cluster = int(cluster)
                        row[f'Worst #{i} Cluster'] = cluster
                        row[f'Worst #{i} %'] = pct
                        cluster_players_map[f"Def-{cluster}"] = players
                        if f'Worst #{i} %' not in pct_columns:
                            pct_columns.append(f'Worst #{i} %')

                    table_rows.append(row)

                # Display defensive stats table
                if table_rows:
                    st.markdown("---")
                    st.markdown("### Defensive Stats")
                    def_stats_df = pd.DataFrame(table_rows)

                    # Convert cluster columns to int
                    cluster_cols = [col for col in def_stats_df.columns if 'Cluster' in col]
                    def_stats_df[cluster_cols] = def_stats_df[cluster_cols].astype(int)

                    # Round percentage columns
                    def_stats_df[pct_columns] = def_stats_df[pct_columns].round(1)

                    # Apply styling to percentage columns
                    styled_def = def_stats_df.style.format({col: '{:+.1f}%' for col in pct_columns})

                    # Apply heatmap to percentage columns
                    for col in pct_columns:
                        styled_def = styled_def.applymap(get_heatmap_color, subset=[col])

                    st.dataframe(styled_def, use_container_width=True, height=400)

                    # Display cluster examples in expandable sections
                    st.markdown("#### 📋 Defensive Cluster Examples")
                    def_clusters_used = set()
                    for i in range(1, 4):
                        def_clusters_used.update(def_stats_df[f'Best #{i} Cluster'].values)
                        def_clusters_used.update(def_stats_df[f'Worst #{i} Cluster'].values)
                    def_clusters_used = sorted(def_clusters_used)

                    cols = st.columns(3)
                    for idx, cluster in enumerate(def_clusters_used):
                        with cols[idx % 3]:
                            def_name = def_cluster_name_map.get(cluster, '')
                            expander_label = f"Def Cluster {cluster}: {def_name}" if def_name else f"Defensive Cluster {cluster}"
                            with st.expander(expander_label):
                                players = cluster_players_map.get(f"Def-{cluster}", [])
                                st.write(", ".join(players))
                st.caption(
                    f"*Analysis based on games with {MIN_THRESHOLD}+ minutes. % differences calculated using Bayesian approach with game-by-game updates.*")
                st.caption(
                    f"*Heatmap scale: -25% (red) to +25% (green). Click cluster expanders below tables to see player examples.*")
    # =====================================================
    # 🟩 TAB 4 — Today's Best Matchups
    # =====================================================
    with tab4:
        st.subheader("🔥 Today's Best Matchup Opportunities")

        try:
            # Load cached matchup data
            with st.spinner("Loading today's games..."):
                matchups_df = load_todays_matchups(merged,min_date, max_date)

            if matchups_df is None:
                st.info("No games scheduled for today or no players with cluster data found.")
            else:
                st.success(f"Found {len(matchups_df)} players with matchup history")

                # Filter options
                show_negative = st.checkbox("Show negative matchups too", value=True)

                # Apply filters
                filtered_df = matchups_df.copy()
                if selected_opp != "All":
                    filtered_df=filtered_df[(filtered_df['Opponent'] == selected_opp)]

                if not show_negative:
                    filtered_df = filtered_df[filtered_df['PTS %'] > 0]

                st.markdown(f"### 📊 Top Matchup Opportunities ({len(filtered_df)} players)")

                # Display summary table
                display_cols = ['PLAYER_NAME', 'Home', 'Opponent', 'Off Cluster Name', 'Def Cluster Name',
                                'PTS %', 'AST %', 'DREB %', 'OREB %', 'FG3M %', 'FG3A %', 'STL %', 'BLK %',
                                'Off Games', 'Def Games']

                def highlight_score(val, col_name):
                    if '%' in col_name and col_name not in ['Home']:
                        try:
                            if val > 10:
                                return 'background-color: #28A745; color: white; font-weight: bold'
                            elif val > 5:
                                return 'background-color: #90EE90; color: black'
                            elif val > 0:
                                return 'background-color: #D4EDDA; color: black'
                            elif val < -5:
                                return 'background-color: #DC3545; color: white'
                            elif val < 0:
                                return 'background-color: #F8D7DA; color: black'
                        except:
                            pass
                    return ''

                styled_matchups = filtered_df[display_cols].style.format({
                    'PTS %': '{:.1f}%',
                    'AST %': '{:.1f}%',
                    'DREB %': '{:.1f}%',
                    'OREB %': '{:.1f}%',
                    'FG3M %': '{:.1f}%',
                    'FG3A %': '{:.1f}%',
                    'STL %': '{:.1f}%',
                    'BLK %': '{:.1f}%'
                }).apply(
                    lambda row: [highlight_score(val, col) for col, val in zip(display_cols, row)],
                    axis=1
                )

                st.dataframe(styled_matchups, use_container_width=True, height=400, hide_index=True)

                st.caption("🏠 = Home game | ✈️ = Away game")

                # =====================================================
                # 🎯 Detailed Projection Section
                # =====================================================
                st.markdown("---")
                st.markdown("### 🎯 Detailed Projection")

                selected_today_player = st.selectbox(
                    "Select player for full projection:",
                    options=filtered_df['PLAYER_NAME'].tolist(),
                    key="today_player_select"
                )

                if selected_today_player:
                    MIN_THRESHOLD = 5
                    offensive_stats = ['PTS', 'AST', 'OREB', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV']
                    defensive_stats = ['DREB', 'REB', 'STL', 'BLK', 'PF']

                    player_info = filtered_df[filtered_df['PLAYER_NAME'] == selected_today_player].iloc[0]
                    opponent = player_info['Opponent']
                    off_cluster = player_info['Off Cluster']
                    def_cluster = player_info['Def Cluster']
                    pct_diff_off = player_info['pct_diff_off']
                    pct_diff_def = player_info['pct_diff_def']

                    player_def_name = def_cluster_name_map.get(def_cluster, def_cluster_name_map.get(float(def_cluster), ''))
                    def_cluster_label = f"{player_def_name} ({int(def_cluster)})" if player_def_name else f"Cluster {int(def_cluster)}"
                    st.markdown(f"**{selected_today_player}** vs **{opponent}**")
                    off_cluster_label = f"{player_info['Off Cluster Name']} ({int(off_cluster)})" if player_info['Off Cluster Name'] else f"Cluster {int(off_cluster)}"
                    st.caption(f"Off Cluster {off_cluster_label} | Def Cluster {def_cluster_label}")

                    # Get player's season stats
                    merged_filtered = merged[
                        (merged['GAME_DATE'] >= min_date) &
                        (merged['GAME_DATE'] <= max_date)
                        ]

                    player_season_stats = merged_filtered[
                        (merged_filtered['PLAYER_NAME'] == selected_today_player) &
                        (merged_filtered['MIN'] >= MIN_THRESHOLD)
                        ].copy()

                    if not player_season_stats.empty:
                        # Calculate player's season averages
                        all_stats = offensive_stats + defensive_stats
                        player_avg_stats = player_season_stats[all_stats + ['MIN']].mean()

                        # Get median minutes from last 10 games
                        player_last_10 = player_season_stats.sort_values('GAME_DATE', ascending=False).head(10)
                        median_min = player_last_10['MIN'].median() if not player_last_10.empty else player_avg_stats[
                            'MIN']

                        # Minutes slider
                        selected_minutes = st.slider(
                            "Minutes per game:",
                            min_value=0.0,
                            max_value=40.0,
                            value=float(median_min),
                            step=0.5,
                            key=f"minutes_{selected_today_player}",
                            help=f"Default is median of last 10 games ({median_min:.1f} min)"
                        )

                        minutes_multiplier = selected_minutes / player_avg_stats['MIN'] if player_avg_stats[
                                                                                               'MIN'] > 0 else 1.0

                        # Build projection
                        projection_data = {'Stat': [], 'Season Avg': [], 'Projected': [], 'Difference': []}

                        display_stats_order = ['PTS', 'AST', 'OREB', 'DREB', 'REB', 'FG3A', 'FG3M', 'FTA', 'FTM', 'STL',
                                               'BLK']

                        for stat in display_stats_order:
                            if stat == 'REB':
                                # Calculate REB from OREB + DREB
                                season_avg = player_avg_stats['OREB'] + player_avg_stats['DREB']
                                oreb_pct = pct_diff_off['OREB'] if player_info['Has Off History'] else 0
                                dreb_pct = pct_diff_def['DREB'] if player_info['Has Def History'] else 0
                                projected = (
                                                    player_avg_stats['OREB'] * (1 + oreb_pct / 100) +
                                                    player_avg_stats['DREB'] * (1 + dreb_pct / 100)
                                            ) * minutes_multiplier
                            elif stat in offensive_stats:
                                season_avg = player_avg_stats[stat]
                                pct_change = pct_diff_off[stat] if player_info['Has Off History'] else 0
                                projected = season_avg * (1 + pct_change / 100) * minutes_multiplier
                            elif stat in defensive_stats:
                                season_avg = player_avg_stats[stat]
                                pct_change = pct_diff_def[stat] if player_info['Has Def History'] else 0
                                projected = season_avg * (1 + pct_change / 100) * minutes_multiplier
                            else:
                                continue

                            projection_data['Stat'].append(stat)
                            projection_data['Season Avg'].append(season_avg)
                            projection_data['Projected'].append(projected)
                            projection_data['Difference'].append(projected - season_avg)

                        proj_df = pd.DataFrame(projection_data)

                        def highlight_diff(val, col_name):
                            if col_name == 'Difference':
                                try:
                                    if val > 0:
                                        return 'background-color: #D4EDDA; color: #155724'
                                    elif val < 0:
                                        return 'background-color: #F8D7DA; color: #721C24'
                                except:
                                    pass
                            return ''

                        styled_proj = proj_df.style.format({
                            'Season Avg': '{:.1f}',
                            'Projected': '{:.1f}',
                            'Difference': '{:.1f}'
                        }).apply(
                            lambda row: [highlight_diff(val, col) for col, val in zip(proj_df.columns, row)],
                            axis=1
                        )

                        st.dataframe(styled_proj, use_container_width=True, hide_index=True)

                        st.caption(f"💡 Projections based on {selected_minutes:.1f} minutes")
                        st.caption(
                            f"📊 Using Off Cluster {off_cluster_label} and Def Cluster {def_cluster_label} historical performance vs {opponent}")
                    else:
                        st.warning(f"No season stats found for {selected_today_player}")

        except Exception as e:
            st.error(f"Error loading today's games: {str(e)}")
            st.exception(e)
            st.info("Make sure the WNBA API is accessible and there are games scheduled today.")

    # =====================================================
    # 💰 TAB 5 — FanDuel EV Analysis
    # =====================================================
    with tab5:
        st.subheader("💰 FanDuel Props — Poisson EV Analysis")

        with st.spinner("Loading FanDuel props..."):
            props_df = load_fanduel_props()

        if props_df.empty:
            st.warning("No FanDuel props loaded. The API may be rate-limited or have no games today.")
        else:
            st.success(f"Loaded {len(props_df)} props for {props_df['Player Name'].nunique()} players")

            with st.spinner("Computing projections..."):
                matchups_df_ev = load_todays_matchups(merged, min_date, max_date)
                player_avgs, player_last10_min = _build_player_season_stats(merged, min_date, max_date)

            if matchups_df_ev is None:
                st.warning("Could not load today's matchup projections. Check that NBA games are scheduled.")
            else:
                def _normalize(name):
                    return name.lower().strip().replace('.', '').replace("'", '').replace('-', ' ')

                # Hard overrides for names FanDuel spells differently than NBA API
                _FD_NAME_OVERRIDES = {
                    # 'FanDuel Name': 'NBA API Name'
                    # e.g. 'nic claxton': 'nicolas claxton',
                }

                matchup_name_map = {_normalize(n): n for n in matchups_df_ev['PLAYER_NAME'].unique()}
                # O(1) row lookup by player name
                matchup_rows = {row['PLAYER_NAME']: row for _, row in matchups_df_ev.iterrows()}

                # ── Team total scaling ──────────────────────────────────────
                # Compute each team's sum of raw projected PTS across all
                # players in today's matchups, then scale so it equals the
                # FanDuel team total line.
                team_totals_map = load_team_totals()  # {abbrev: total}

                inactive_players_ev = get_inactive_players()

                # Players with FanDuel props are confirmed active — override any
                # stale ESPN injury status.
                # Normalize names (strip dots/apostrophes, lowercase) so
                # "R.J. Barrett" and "RJ Barrett" compare equal.
                def _norm_name(n):
                    return n.lower().replace('.', '').replace("'", '').strip()

                props_norm = {_norm_name(n) for n in props_df['Player Name'].unique()}
                # Build a map from normalized ESPN name → original ESPN name
                inactive_norm = {_norm_name(n): n for n in inactive_players_ev}

                # A player is truly inactive only if no FanDuel prop exists for them
                truly_inactive_norm = {norm for norm in inactive_norm if norm not in props_norm}

                team_proj_sum = {}  # {team_abbrev: sum_of_raw_proj_pts}
                for pname, mrow in matchup_rows.items():
                    if _norm_name(pname) in truly_inactive_norm:
                        continue
                    team = mrow.get('TEAM', '')
                    if not team or pname not in player_avgs.index:
                        continue
                    avg_row = player_avgs.loc[pname]
                    if avg_row['MIN'] <= 0:
                        continue
                    mm = player_last10_min.get(pname, avg_row['MIN']) / avg_row['MIN']
                    pct = float(mrow['pct_diff_off']['PTS']) if mrow['Has Off History'] else 0.0
                    proj_pts = float(avg_row['PTS']) * (1 + pct / 100) * mm
                    team_proj_sum[team] = team_proj_sum.get(team, 0.0) + proj_pts

                pts_scalars = {}  # {team_abbrev: scale_factor}
                for team, proj_sum in team_proj_sum.items():
                    if team in team_totals_map and proj_sum > 0:
                        pts_scalars[team] = team_totals_map[team] / proj_sum
                    else:
                        pts_scalars[team] = 1.0

                if team_totals_map:
                    total_lines = ', '.join(
                        f"{t}: {v:.1f}" for t, v in sorted(team_totals_map.items())
                        if t in team_proj_sum
                    )
                    st.caption(f"📊 Team totals (FanDuel): {total_lines}")
                actually_excluded = {inactive_norm[n] for n in truly_inactive_norm}
                if actually_excluded:
                    st.caption(f"🚑 Excluded (Out/Doubtful, no props): {', '.join(sorted(actually_excluded))}")
                # ────────────────────────────────────────────────────────────

                use_team_total_adj = st.checkbox("Team total adjustment", value=True, key="ev_team_total_adj")

                ev_rows = []
                for _, prop in props_df.iterrows():
                    fd_name = prop['Player Name']
                    norm = _normalize(fd_name)

                    # 1. Check hard overrides first
                    norm_override = _FD_NAME_OVERRIDES.get(norm)
                    matched = matchup_name_map.get(norm_override) if norm_override else None
                    # 2. Exact normalized match
                    if matched is None:
                        matched = matchup_name_map.get(norm)
                    # 3. Fallback: fuzzy full-name match (handles minor spelling diffs,
                    #    but rejects similar-but-wrong names like Davion vs Donovan Mitchell)
                    if matched is None:
                        close = difflib.get_close_matches(norm, matchup_name_map.keys(), n=1, cutoff=0.85)
                        matched = matchup_name_map[close[0]] if close else None

                    if matched is None or matched not in player_avgs.index:
                        continue

                    category = prop['Category']
                    line = prop['Line']
                    over_odds = prop['Over odds']
                    under_odds = prop['Under odds']

                    if pd.isna(line):
                        continue

                    avg = player_avgs.loc[matched]
                    if avg['MIN'] <= 0:
                        continue
                    median_min = player_last10_min.get(matched, avg['MIN'])
                    mm = median_min / avg['MIN']

                    mrow = matchup_rows[matched]
                    pct_diff_off = mrow['pct_diff_off']
                    pct_diff_def = mrow['pct_diff_def']
                    has_off = mrow['Has Off History']
                    has_def = mrow['Has Def History']
                    opponent = mrow['Opponent']
                    player_team = mrow.get('TEAM', '')
                    pts_scalar = pts_scalars.get(player_team, 1.0)

                    components = _PROP_STAT_COMPONENTS.get(category, [])
                    if not components:
                        continue
                    mu = 0.0
                    mu_raw = 0.0
                    for stat in components:
                        s_avg = float(avg.get(stat, 0.0))
                        if stat in _OFFENSIVE_STATS:
                            pct = float(pct_diff_off[stat]) if (has_off and stat in pct_diff_off.index) else 0.0
                        elif stat in _DEFENSIVE_STATS:
                            pct = float(pct_diff_def[stat]) if (has_def and stat in pct_diff_def.index) else 0.0
                        else:
                            pct = 0.0
                        component_mu = s_avg * (1 + pct / 100) * mm
                        mu_raw += component_mu
                        if stat == 'PTS' and use_team_total_adj:
                            component_mu *= pts_scalar
                        mu += component_mu
                    if mu <= 0:
                        continue

                    ev_over, ev_under, p_over = _calc_poisson_ev(mu, line, over_odds, under_odds)

                    ev_rows.append({
                        'Player': matched,
                        'Opponent': opponent,
                        'Stat': category,
                        'Line': line,
                        'Raw Proj': round(mu_raw, 2),
                        'Projection': round(mu, 2),
                        'Proj - Line': round(mu - line, 2),
                        'P(Over)%': round(p_over, 1) if p_over is not None else None,
                        'Over Odds': _to_american(over_odds),
                        'Under Odds': _to_american(under_odds),
                        'EV Over%': round(ev_over, 1) if ev_over is not None else None,
                        'EV Under%': round(ev_under, 1) if ev_under is not None else None,
                    })

                if not ev_rows:
                    st.info("No players matched between FanDuel props and today's projection data.")
                else:
                    ev_df = pd.DataFrame(ev_rows)
                    ev_df['Best EV%'] = ev_df[['EV Over%', 'EV Under%']].max(axis=1)

                    # Apply sidebar opponent / player filters
                    if selected_opp != "All":
                        ev_df = ev_df[ev_df['Opponent'] == selected_opp]
                    if selected_player != "All":
                        ev_df = ev_df[ev_df['Player'] == selected_player]

                    # Tab-level filter controls
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        all_cats = sorted(ev_df['Stat'].unique()) if not ev_df.empty else []
                        cat_filter = st.multiselect("Filter by stat", options=all_cats, default=all_cats, key="ev_cat_filter")
                    with col_f2:
                        min_ev = st.slider("Min best EV%", -30.0, 20.0, 0.0, 0.5, key="ev_min_slider")

                    if cat_filter:
                        ev_df = ev_df[ev_df['Stat'].isin(cat_filter)]
                    ev_df = ev_df[ev_df['Best EV%'] >= min_ev].sort_values('Best EV%', ascending=False)

                    st.markdown(f"### 📋 {len(ev_df)} props | {ev_df['Player'].nunique()} players")

                    def _ev_color(val):
                        try:
                            v = float(val)
                            if v > 10:    return 'background-color: #28A745; color: white; font-weight: bold'
                            elif v > 5:   return 'background-color: #90EE90; color: black'
                            elif v > 0:   return 'background-color: #D4EDDA; color: black'
                            elif v < -10: return 'background-color: #DC3545; color: white'
                            elif v < 0:   return 'background-color: #F8D7DA; color: black'
                        except Exception:
                            pass
                        return ''

                    display_df = ev_df.drop(columns=['Best EV%']).reset_index(drop=True)
                    styled = display_df.style.format({
                        'Raw Proj': '{:.1f}',
                        'Projection': '{:.1f}',
                        'Proj - Line': '{:+.1f}',
                        'P(Over)%': '{:.1f}%',
                        'EV Over%': '{:+.1f}%',
                        'EV Under%': '{:+.1f}%',
                    }, na_rep='—').applymap(_ev_color, subset=['EV Over%', 'EV Under%'])

                    st.dataframe(styled, use_container_width=True, height=520, hide_index=True)
                    st.caption(
                        "EV% = expected return per $100 wagered. Projection = cluster-adjusted season avg "
                        "scaled to median of last 10 games minutes. Poisson CDF used for P(Over/Under)."
                    )


def clean_existing_csv(filepath='player_data.csv', output_path='player_data.csv'):
    """Clean existing player_data.csv by removing duplicates"""
    df = pd.read_csv(filepath)

    print(f"Original rows: {len(df)}")

    # Check for duplicates
    duplicates_mask = df.duplicated(subset=['PLAYER_ID', 'SEASON_ID'], keep=False)
    num_duplicates = duplicates_mask.sum()

    if num_duplicates > 0:
        print(f"Found {num_duplicates} duplicate rows")

        # Keep first occurrence of each player-season
        df_cleaned = df.drop_duplicates(subset=['PLAYER_ID', 'SEASON_ID'], keep='first')

        print(f"Cleaned rows: {len(df_cleaned)}")
        print(f"Removed {len(df) - len(df_cleaned)} duplicate rows")

        # Save cleaned version
        df_cleaned.to_csv(output_path, index=False)
        print(f"\n✓ Saved cleaned data to '{output_path}'")

        return df_cleaned
    else:
        print("✓ No duplicates found!")
        return df


if __name__ == '__main__':
    #create_clusters()
    main()