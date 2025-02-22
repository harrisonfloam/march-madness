from typing import List, Callable, Dict, Any, Tuple
from math import log2
import numpy as np
import pandas as pd
import itertools


Team = Dict[str, Any]  # Team dictionary
Matchup = Tuple[Team, Team]  # Matchup of teams


class Game:
    def __init__(self, team1: str, team2: str, round_number: int, 
                 team1_details: Dict = None, team2_details: Dict = None):
        self.team1 = team1
        self.team2 = team2
        self.team1_details = team1_details or {}
        self.team2_details = team2_details or {}
        self.round_number = round_number
        self.winner = None
        self.winner_details = None
        #TODO: make this simpler... just pass the details dict, grab the 'team'
        
    def set_winner(self, winner: str):
        if winner not in {self.team1, self.team2}:
            raise ValueError(f"Winner must be either {self.team1} or {self.team2}, got '{winner}'.")
        self.winner = winner
        self.winner_details = self.team1_details if self.winner == self.team1 else self.team2_details

class Tournament:
    def __init__(self, teams: List[Team], 
                 initial_matchup_strategy: Callable[["Tournament"], List[Matchup]] = None,
                 round_matchup_strategy: Callable[["Tournament"], List[Matchup]] = None):
        """
        Initializes a tournament.
        
        Args:
            teams: A structured input containing team data (can be a DataFrame or list of dicts).
            initial_matchup_strategy: A function that determines first-round matchups.
            round_matchup_strategy: A function that determines matchups in later rounds.
        """
        self.teams = self._process_teams(teams)
        self.unplayed_games = []
        self.played_games = []
        self.winner = None
        
        # Set default matchup strategies
        self.initial_matchup_strategy = initial_matchup_strategy or ncaa_initial_matchups
        self.round_matchup_strategy = round_matchup_strategy or ncaa_round_matchups
        
        self.num_rounds = int(log2(len(teams))) # Single elimination tournament
        
        self._validate_tournament()
        self.create_initial_matchups()
        
    def _validate_tournament(self):
        """Validates tournament parameters before proceeding."""
        num_teams = len(self.teams)
        if num_teams == 0 or (num_teams & (num_teams - 1)) != 0:    # Bitwise power of 2 check
            raise ValueError(f"Invalid number of teams: {num_teams}. Must be a power of 2 for a single-elimination tournament.")

        # Ensure all teams have required fields
        required_fields = {"team"}
        if self.initial_matchup_strategy == ncaa_initial_matchups:
            required_fields.update({"seed", "region"})  # NCAA-specific fields

        for team in self.teams.values():
            missing_fields = required_fields - team.keys()
            if missing_fields:
                raise ValueError(f"Missing required fields {missing_fields} in team entry: {team}")

                
    def _process_teams(self, teams: Any) -> Dict[str, Team]:
        """Converts a DataFrame or list of dicts into a standard format."""
        if isinstance(teams, pd.DataFrame):
            teams = teams.to_dict(orient="records")  # Convert DataFrame to list of dicts
        if isinstance(teams, list) and all(isinstance(t, dict) for t in teams):
            return {team["team"]: team for team in teams}
        else:
            raise ValueError("Teams must be provided as a DataFrame or a list of dictionaries.")

    def create_initial_matchups(self):
        """Creates first-round matchups using the initial matchup strategy."""
        self.current_round = 1
        matchups = self.initial_matchup_strategy(self)
        if not matchups:
            raise ValueError("Initial matchup strategy did not return any games. Check the strategy function.")
        
        for team1, team2 in matchups:
            game = Game(team1, team2, 1)
            self.unplayed_games.append(game)

    def update_game_result(self, game: Game, winner: str, verbose: bool = False):
        """Marks the winner of a game and builds the next round when all games are complete."""
        game.set_winner(winner)
        self.unplayed_games.remove(game)
        self.played_games.append(game)
        
        # If final round, set the tournament winner
        if self.current_round == self.num_rounds:
            self.winner = winner

        # If all games for this round are completed, advance the round
        elif len(self.unplayed_games) == 0:
            self.create_next_round()

    def create_next_round(self):
        """Creates matchups for the next round using the round matchup strategy."""
        winners = [game.winner for game in self.played_games if game.winner and game.round_number == self.current_round]

        # If odd number of winners, tournament is broken.
        if len(winners) % 2 != 0:
            raise ValueError(f"Found {len(winners)} winners from round {self.current_round}; Expected even count.")

        self.current_round += 1 # Advance round counter
        matchups = self.round_matchup_strategy(self)

        # Create new games for the next round
        new_games = [Game(team1["team"], team2["team"], self.current_round, team1_details=team1, team2_details=team2) for team1, team2 in matchups]
        self.unplayed_games.extend(new_games)
        
    def get_unplayed_games(self, round_number: int = None) -> List[Game]:
        """Returns unplayed games, optionally filtering by round."""
        if round_number:
            return [game for game in self.unplayed_games if game.round_number == round_number]
        return self.unplayed_games.copy()   # Return as copy so future changes don't affect iterable use

    def print_tournament_state(self):
        pass
    
class TournamentSimulator:
    def __init__(self, num_trials: int, tournament_class: type, tournament_params: Dict[str, Any], 
                 prediction_strategy: Callable, prediction_details: bool = False):
        """
        Monte Carlo tournament simulator.

        Args:
            num_trials: Number of tournament simulations to run.
            tournament_class: The Tournament class to instantiate.
            tournament_params: Dictionary of parameters to initialize Tournament.
            prediction_strategy: Callable that predicts the winner of a game.
            prediction_details: Whether to log prediction confidence and context.
        """
        self.num_trials = num_trials
        self.tournament_class = tournament_class
        self.tournament_params = tournament_params
        self.prediction_strategy = prediction_strategy
        self.prediction_details = prediction_details
        self.results_log = []
        # TODO: seed... generate seeds for each trial
        

    def run(self):
        """Runs multiple tournament simulations and logs results."""
        for trial in range(1, self.num_trials + 1):
            print(f"Starting Trial {trial}")
            tournament = self.tournament_class(**self.tournament_params)

            while tournament.get_unplayed_games():
                for game in tournament.get_unplayed_games():
                    winner, details = self.prediction_strategy(game)
                    tournament.update_game_result(game, winner)

                    # Log each game result
                    self.results_log.append({
                        "trial": trial,
                        "round": game.round_number,
                        "game": f"{game.team1} vs {game.team2}",
                        "result": f"{winner} wins",
                        "details": details if self.prediction_details else None,
                        "tournament_state": self._get_tournament_snapshot(tournament)
                    })

    def _get_tournament_snapshot(self, tournament):
        """Returns a structured snapshot of the tournament state."""
        return [(game.round_number, game.team1, game.team2, game.winner) for r, games in tournament.rounds.items() for game in games]

    def to_dataframe(self):
        """Returns the logged results as a Pandas DataFrame."""
        return pd.DataFrame(self.results_log)

def ncaa_initial_matchups(tournament: "Tournament") -> List[Matchup]:
    """
    Generates first-round matchups using NCAA-style seeding.
    
    In each region, the highest seed plays the lowest, second-highest plays second-lowest: 
    1 vs. 16, 2 vs. 15, etc.
    """
    matchups = []
    regions = set(team["region"] for team in tournament.teams.values())  # All original regions

    for region in regions:
        region_teams = sorted(
            [team for team in tournament.teams.values() if team["region"] == region],
            key=lambda x: x["seed"]
        )
        for i in range(len(region_teams) // 2):
            matchups.append((region_teams[i], region_teams[-(i + 1)]))  # Highest vs. Lowest
    return matchups


def ncaa_round_matchups(tournament: "Tournament") -> List[Matchup]:
    """
    Generates next-round matchups using NCAA-style progression.
    
    Winners are paired in order defined by the bracket structure: 
    - Winner of (1 vs. 16) faces winner of (8 vs. 9)
    - Winner of (5 vs. 12) faces winner of (4 vs. 13), etc.
    - East plays West, South plays Midwest
    """
    # NCAA-defined bracket structure for regional rounds
    ncaa_bracket_progression = [
        1, 16, 8, 9,
        5, 12, 4, 13,
        6, 11, 3, 14,
        7, 10, 2, 15
    ]
    # Final Four region matchups
    inter_region_matchups = [
        ("East", "West"),
        ("South", "Midwest")
    ]

    # Get winners from last round
    prev_round_games = [game for game in tournament.played_games if game.winner and game.round_number == tournament.current_round-1]
    num_winners = len(prev_round_games)
    regions = set(team["region"] for team in tournament.teams.values())  # All original regions
    matchups = []

    # Regional rounds
    if num_winners > len(regions):
        for region in regions:
            region_winners = [tournament.teams[game.winner] for game in prev_round_games if game.winner_details["region"] == region]
            
            # Iterate over chunks of the bracket
            # In round 1, winners must come from seeds {16, 1, 8, 9}...
            for round_bracket in itertools.batched(ncaa_bracket_progression, (tournament.current_round)**2):
                round_bracket_set = set(round_bracket)
                matchup = tuple(team for team in region_winners if team["seed"] in round_bracket_set)
                matchups.append(tuple(matchup))
                
    # Final Four
    elif num_winners == len(regions):
        winners_by_region = {tournament.teams[game.winner]["region"]: tournament.teams[game.winner] for game in prev_round_games}
        for region1, region2 in inter_region_matchups:
            matchups.append((winners_by_region[region1], winners_by_region[region2]))
            
    # Championship
    else:
        winners = tuple(tournament.teams[game.winner] for game in prev_round_games)
        matchups.append(winners)

    return matchups