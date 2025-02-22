"""Tournament class definition."""

from typing import List, Callable, Dict, Any, Tuple
from math import log2
import numpy as np
import pandas as pd
import itertools


Team = Dict[str, Any]  # Team dictionary
Matchup = Tuple[Team, Team]  # Matchup of teams


class Game:
    def __init__(self, team1: Team, team2: Team, round_number: int):
        for team in (team1, team2):
            if "team" not in team:
                raise ValueError(f"Team dictionary must contain at least key 'team'. Found keys: {team.keys()}")
        
        self.team1 = team1["team"]
        self.team2 = team2["team"]
        self.team1_details = team1
        self.team2_details = team2
        self.round_number = round_number
        self.winner = None
        self.winner_details = None
        
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
        new_games = [Game(team1, team2, self.current_round) for team1, team2 in matchups]
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
                 prediction_strategy: Callable[["Game", "Tournament", int], Prediction], prediction_details: bool = False,
                 seed: int = 42):
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
        self.prediction_details = prediction_details    #TODO: rework this... shouldnt be bool
        
        # RNG
        self.seed = seed
        self.seeds = self._generate_seeds()
        
        self.results = []
        
    def _generate_seeds(self) -> List[int]:
        """Generate random seeds for each trial for reproducibility."""
        seed_max = 2**32 - 1
        np.random.seed(self.seed)
        return np.random.randint(0, seed_max, size=self.num_trials).tolist()

    def run(self, verbose: bool = True):
        """Runs multiple tournament simulations and logs results."""
        trials = zip(range(1, self.num_trials+1), self.seeds)
        pbar = tqdm(trials, total=self.num_trials, disable=not verbose)
        for trial, trial_seed in pbar:
            tournament = self.tournament_class(**self.tournament_params)
            rng = np.random.default_rng(trial_seed)  # One RNG per trial

            while tournament.get_unplayed_games():
                for game in tournament.get_unplayed_games():
                    winner, details = self.prediction_strategy(tournament, game, rng)
                    tournament.update_game_result(game, winner)
                    
                    result = {
                        "trial": trial,
                        "round": game.round_number,
                        "team1": game.team1,
                        "team2": game.team2,
                        "winner": winner,
                        "details": details if self.prediction_details else None,
                        "tournament_state": self._get_tournament_snapshot(tournament)
                    }
                    self.results.append(result)
            
            postfix = {
                "winner": tournament.winner
            }
            pbar.set_postfix(postfix)

    def _get_tournament_snapshot(self, tournament) -> List[Tuple[int, str, str, str]]:
        """Returns a structured snapshot of the tournament state."""
        #TODO: rework
        return [
            (game.round_number, game.team1, game.team2, game.winner)
            for game in tournament.played_games
        ]
    
    def to_dataframe(self):
        """Returns the logged results as a Pandas DataFrame."""
        return pd.DataFrame(self.results)