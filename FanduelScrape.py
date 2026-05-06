import pandas as pd
import requests
def getnbaprops():
    API_KEY = '50d7871c8a201db4df96c55ed424cd6a'  # Replace with your Odds API key
    BASE_URL = 'https://api.the-odds-api.com/v4/sports'

    def get_event_ids(sport, region='us', market=''):
        """
        Fetch all event IDs for a specific sport.

        :param sport: The sport key (e.g., 'basketball_nba').
        :param region: The betting region ('us', 'uk', 'eu', 'au').
        :param market: The betting market (e.g., 'player_props').
        :return: A list of event IDs or an error message.
        """
        url = f"{BASE_URL}/{sport}/odds"
        params = {
            'apiKey': API_KEY,
            'regions': region,
            'markets': market,
            'bookmakers': "fanduel"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            events = response.json()
            # Extract event IDs from the response
            event_ids = [event['id'] for event in events]
            return event_ids
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}

    def get_odds_for_event(sport, event_id):
        """
        Fetch player props odds for a specific event.

        :param sport: The sport key (e.g., 'basketball_nba').
        :param event_id: The event ID to fetch odds for.
        :return: JSON response with odds data or an error message.
        """
        url = f"{BASE_URL}/{sport}/events/{event_id}/odds"
        params = {
            'apiKey': API_KEY,
            'markets': 'player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_points_rebounds_assists,player_points_rebounds,player_points_assists,player_rebounds_assists',
            'bookmakers': 'fanduel',
            'odds_format': 'American'
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}

    # Example usage
    sport_key = 'basketball_wnba'  # WNBA odds

    # Step 1: Get all event IDs for the sport
    event_ids = get_event_ids(sport_key)
    result = []
    if isinstance(event_ids, dict) and 'error' in event_ids:
        print(f"Error fetching event IDs: {event_ids['error']}")
    else:
        # Step 2: Fetch odds for each event individually
        for event_id in event_ids:
            odds_data = get_odds_for_event(sport_key, event_id)
            if 'error' in odds_data:
                print(f"Error fetching odds for event {event_id}: {odds_data['error']}")
            else:
                print(f"Odds for event {event_id}:",odds_data)
                result.append(odds_data)
    return result
def reformat_api(api):
    rows = []
    category_map = { #todo add combo
        'player_points': 'PTS',
        'player_rebounds': 'REB',
        'player_assists': 'AST',
        'player_steals': 'STL',
        'player_blocks': 'BLK',
        'player_points_assists': 'PA',
        'player_points_rebounds':'PR',
        'player_points_rebounds_assists':'PRA',
        'player_rebounds_assists':'RA',
        'player_threes':'3PM'
    }
    for event in api:
        for bookmaker in event.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                category = category_map.get(market['key'], market['key'])
                outcomes = market['outcomes']
                player_lines = {}

                # Group outcomes by player and type (Over/Under)
                for outcome in outcomes:
                    player = outcome['description']
                    line = outcome['point']
                    odds = outcome['price']
                    name = outcome['name']  # Over or Under

                    if player not in player_lines:
                        player_lines[player] = {'Line': line, 'Over odds': None, 'Under odds': None}

                    if name == 'Over':
                        player_lines[player]['Over odds'] = odds
                    elif name == 'Under':
                        player_lines[player]['Under odds'] = odds

                # Create rows for the DataFrame
                for player, values in player_lines.items():
                    rows.append({
                        'Player Name': player,
                        'Category': category,
                        'Line': values['Line'],
                        'Over odds': values['Over odds'],
                        'Under odds': values['Under odds'],
                    })

    return pd.DataFrame(rows)
def get_team_totals():
    """
    Derive each team's projected point total from FanDuel game totals + spreads.
    Formula: team_total = (game_total + spread_for_team) / 2
    where spread_for_team is negative for the favorite and positive for the underdog.
    Falls back to game_total / 2 if spread is unavailable.
    """
    API_KEY = '50d7871c8a201db4df96c55ed424cd6a'
    BASE_URL = 'https://api.the-odds-api.com/v4/sports'
    sport_key = 'basketball_wnba'

    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'totals,spreads',
        'bookmakers': 'fanduel',
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        events = response.json()
    except Exception as e:
        print(f"Error fetching game totals/spreads: {e}")
        return {}

    team_totals = {}
    for event in events:
        home_team = event.get('home_team', '')
        away_team = event.get('away_team', '')
        game_total = None
        spreads = {}  # team_name -> spread point (negative = favorite)

        for bookmaker in event.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'totals':
                    for outcome in market['outcomes']:
                        if outcome['name'] == 'Over':
                            game_total = float(outcome['point'])
                elif market['key'] == 'spreads':
                    for outcome in market['outcomes']:
                        spreads[outcome['name']] = float(outcome['point'])

        if game_total is None:
            continue

        for team in (home_team, away_team):
            spread = spreads.get(team, 0.0)
            team_totals[team] = (game_total + spread) / 2

    return team_totals


# Example usage:

